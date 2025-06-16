#!/usr/bin/env python3
"""
Pipeline principal de MetronIA: Genera mutaciones y analiza interpretaciones musicales.

Este script implementa el pipeline completo:
1. Crear mutaciones a partir de un MIDI de referencia
2. Convertir todas las mutaciones y el MIDI de referencia a WAV
3. Analizar cada mutación comparándola con el original (usando MusicAnalyzer)
4. Generar un CSV consolidado con resumen de todos los análisis

Nota: Los análisis individuales detallados (CSV y visualizaciones) se generan 
automáticamente por MusicAnalyzer. Este script solo crea un resumen consolidado.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from tqdm import tqdm

from mutations.manager import MutationManager
from mutations.midi_utils import load_midi_with_pretty_midi, load_midi_with_mido, save_excerpt_in_audio, save_mutation_complete, extract_tempo_from_midi
from analyzers import MusicAnalyzer

EPILOG = """Ejemplos de uso:
  # Aplicar todas las mutaciones (comportamiento por defecto)
  python mutar_y_analizar.py

  # Aplicar solo mutaciones de timing
  python mutar_y_analizar.py --categories timing_errors

  # Aplicar múltiples categorías específicas
  python mutar_y_analizar.py --categories timing_errors tempo_errors

  # Usar un archivo MIDI específico
  python mutar_y_analizar.py --midi path/to/your/file.mid

  # Combinación de opciones
  python mutar_y_analizar.py --midi midi/Acordai-100.mid --categories timing_errors pitch_errors

Categorías disponibles:
  - pitch_errors: Errores de altura de las notas
  - tempo_errors: Errores relacionados con el tempo
  - timing_errors: Errores de timing de las notas
  - duration_errors: Errores de duración de las notas
  - note_errors: Errores de presencia de notas
  - articulation_errors: Errores de articulación
        """

def metronia_arg_parser():
        # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description="Pipeline MetronIA: Genera mutaciones y analiza interpretaciones musicales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG)
    
    parser.add_argument(
        '--midi',
        type=str,
        default="midi/Acordai-100.mid",
        help='Ruta al archivo MIDI de referencia (default: midi/Acordai-100.mid)'
    )
    
    parser.add_argument(
        '--categories',
        nargs='*',
        help='Categorías específicas de mutaciones a aplicar. Si no se especifica, aplica todas.'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default="results",
        help='Directorio base para guardar resultados (default: results)'
    )
    
    parser.add_argument(
        '--list-categories',
        action='store_true',
        help='Mostrar todas las categorías disponibles y salir'
    )
    return parser.parse_args()

def save_analysis_results_to_csv(analysis_data: List[Dict[str, Any]], output_file: str):
    """
    Guarda los resultados de análisis en formato CSV.
    
    Args:
        analysis_data: Lista de diccionarios con los datos de análisis
        output_file: Ruta del archivo CSV de salida
    """
    if not analysis_data:
        print("⚠️ No hay datos de análisis para guardar en CSV")
        return
    
    # Crear DataFrame
    df = pd.DataFrame(analysis_data)
    
    # Guardar CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✅ Resultados guardados en CSV: {output_file}")
    
    return df

def obtener_audio_referencia(midi_file_path: str, midi_name):
    try:
        original_excerpt = load_midi_with_pretty_midi(midi_file_path)
        print("✅ Archivo MIDI cargado exitosamente con pretty_midi")
    except Exception as e:
        print(f"⚠️ Error con pretty_midi: {e}")
        try:
            original_excerpt = load_midi_with_mido(midi_file_path)
            print("✅ Archivo MIDI cargado exitosamente con mido")
        except Exception as e2:
            print(f"❌ Error cargando MIDI: {e2}")
            return None

    base_tempo = extract_tempo_from_midi(midi_file_path)
    print(f"✅ Tempo detectado: {base_tempo} BPM")

    try:
        reference_audio_path = save_excerpt_in_audio(
            excerpt=original_excerpt,
            save_name=f"{midi_name}_reference"
        )
        print(f"✅ Audio de referencia guardado: {reference_audio_path}")
    except Exception as e:
        print(f"❌ Error generando audio de referencia: {e}")
        return None
    
    return original_excerpt, base_tempo, reference_audio_path

def filtrar_mutaciones(categories_filter):
    mutation_manager = MutationManager()
    
    # Aplicar filtro de categorías si se especifica
    if categories_filter:
        print(f"🎯 Filtrando categorías: {', '.join(categories_filter)}")
        
        # Verificar que las categorías especificadas existen
        available_categories = list(mutation_manager.categories.keys())
        invalid_categories = [cat for cat in categories_filter if cat not in available_categories]
        
        if invalid_categories:
            print(f"❌ Categorías no válidas: {', '.join(invalid_categories)}")
            print(f"📋 Categorías disponibles: {', '.join(available_categories)}")
            return None
        
        # Filtrar las categorías
        filtered_categories = {name: category for name, category in mutation_manager.categories.items() 
                             if name in categories_filter}
        mutation_manager.categories = filtered_categories
    
    total_mutations = len(mutation_manager.get_all_mutations())
    print(f"📊 Total de mutaciones a aplicar: {total_mutations}")
    for category_name, category in mutation_manager.categories.items():
        print(f"  - {category.description}: {len(category.mutations)} mutaciones")
    return mutation_manager
    
def aplicar_mutaciones(mutation_manager, original_excerpt, base_tempo, midi_name, results_dir):
    """
    Aplica todas las mutaciones y guarda los cambios detallados en archivos.
    
    Args:
        mutation_manager: Gestor de mutaciones
        original_excerpt: Excerpt original 
        base_tempo: Tempo base del MIDI
        midi_name: Nombre base del archivo MIDI
        results_dir: Directorio de resultados
    """
    successful_mutations = []
    failed_mutations = []
    
    # Crear directorio para el conjunto de mutaciones usando el nombre del MIDI de referencia
    mutations_base_dir = results_dir / midi_name
    mutations_base_dir.mkdir(exist_ok=True)
    
    # Crear directorio para los cambios de mutaciones
    changes_dir = mutations_base_dir / "mutation_changes"
    changes_dir.mkdir(exist_ok=True)
    
    for category_name, category in mutation_manager.categories.items():
        for mutation_name, mutation in category.mutations.items():
            try:
                # Crear subdirectorio para esta mutación específica
                mutation_changes_dir = changes_dir / f"{category_name}_{mutation_name}"
                
                # Aplicar mutación con guardado automático de cambios
                success = mutation.apply(
                    original_excerpt, 
                    tempo=base_tempo,
                    save_changes=True,
                    output_dir=str(mutation_changes_dir)
                )
                
                if success and mutation.excerpt is not None:
                    # Generar archivo de audio y MIDI con tempo correcto
                    file_name = f"{midi_name}_{mutation_name}"
                    
                    # Usar save_mutation_complete para calcular automáticamente el tempo
                    audio_path, midi_path, calculated_tempo = save_mutation_complete(
                        mutation_result=mutation,
                        save_name=file_name,
                        base_tempo=base_tempo  # Usar el tempo extraído del MIDI original
                    )
                    
                    mutation.set_path(audio_path)
                    successful_mutations.append((category_name, mutation_name, mutation, audio_path))
                else:
                    failed_mutations.append((category_name, mutation_name, mutation.error or "Unknown error"))
                    
            except Exception as e:
                failed_mutations.append((category_name, mutation_name, str(e)))

    if failed_mutations:
        print(f"\n⚠️ Mutaciones fallidas:")
        for category, name, error in failed_mutations:
            print(f"  - {category}.{name}: {error}")    # 5. ANÁLISIS DE CADA MUTACIÓN VS ORIGINAL
    return successful_mutations

def analizar_mutaciones(successful_mutations, reference_audio_path, base_tempo, midi_name, results_dir):
    """
    Analiza cada mutación contra el audio de referencia.
    
    Args:
        successful_mutations: Lista de mutaciones exitosas
        reference_audio_path: Ruta del audio de referencia
        base_tempo: Tempo base del MIDI
        midi_name: Nombre base del archivo MIDI  
        results_dir: Directorio de resultados base
    """
    csv_data = []
    analyzer = MusicAnalyzer()
    
    for category_name, mutation_name, mutation, audio_path in tqdm(successful_mutations, 
                                                                   desc="Analizando mutaciones"):
        try:
            # Crear nombre único para cada análisis individual usando el formato especificado
            analysis_name = f"{midi_name}_{category_name}_{mutation_name}"
            
            # El análisis completo ya genera CSV y visualizaciones automáticamente
            # Cada análisis se guarda en results/NOMBRE_ANALISIS/
            analysis_result = analyzer.comprehensive_analysis(
                reference_path=reference_audio_path,
                live_path=audio_path,
                save_name=analysis_name,  # Nombre único para cada análisis
                reference_tempo=base_tempo,
                verbose=False
            )
            
            # Obtener datos básicos para el CSV consolidado (sin duplicar el CSV individual)
            dtw_onset_result = analysis_result.get('dtw_onsets')
            if dtw_onset_result:
                # Solo extraer datos esenciales para el resumen
                csv_row = {
                    'mutation_category': category_name,
                    'mutation_name': mutation_name,
                    'total_onsets_ref': len(dtw_onset_result.matches) + len(dtw_onset_result.missing_onsets),
                    'total_onsets_live': len(dtw_onset_result.matches) + len(dtw_onset_result.extra_onsets),
                    'correct_matches': len([m for m in dtw_onset_result.matches if m.classification.value == 'correct']),
                    'late_matches': len([m for m in dtw_onset_result.matches if m.classification.value == 'late']),
                    'early_matches': len([m for m in dtw_onset_result.matches if m.classification.value == 'early']),
                    'missing_onsets': len(dtw_onset_result.missing_onsets),
                    'extra_onsets': len(dtw_onset_result.extra_onsets),
                    'beat_spectrum_similar': analysis_result['beat_spectrum'].is_similar,
                    'beat_spectrum_max_difference': f"{analysis_result['beat_spectrum'].max_difference:.3f}",
                    'tempo_reference_bpm': f"{analysis_result['tempo'].tempo_ref:.2f}",
                    'tempo_live_bpm': f"{analysis_result['tempo'].tempo_live:.2f}",
                    'tempo_difference_bpm': f"{analysis_result['tempo'].difference:.2f}",
                    'tempo_similar': analysis_result['tempo'].is_similar,
                    'dtw_assessment': analysis_result.get('dtw_analysis', {}).get('overall_assessment', 'N/A'),
                    'audio_file_path': str(audio_path),
                    'reference_audio_path': str(reference_audio_path)
                }
                
                csv_data.append(csv_row)
            else:
                print(f"⚠️ No se obtuvieron resultados DTW para {category_name}.{mutation_name}")
                
        except Exception as e:
            print(f"❌ Error analizando {category_name}.{mutation_name}: {e}")
            # Agregar una fila con datos mínimos para no perder la información
            csv_data.append({
                'mutation_category': category_name,
                'mutation_name': mutation_name,
                'error': str(e),                'audio_file_path': str(audio_path),
                'reference_audio_path': str(reference_audio_path)
            })
        # Guardar resultados consolidados en directorio del MIDI de referencia
    if csv_data:
        # Crear directorio para el conjunto de mutaciones usando NOMBRE_REFERENCIA_MUTACION
        mutations_summary_dir = results_dir / midi_name
        mutations_summary_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar CSV consolidado con resumen de todas las mutaciones
        csv_file = mutations_summary_dir / "mutations_summary.csv"
        save_analysis_results_to_csv(csv_data, csv_file)
        
        # Crear reporte de resumen en texto
        summary_report_path = mutations_summary_dir / "summary_report.txt"
        _create_mutations_summary_report(csv_data, midi_name, summary_report_path)


def _create_mutations_summary_report(csv_data: List[Dict[str, Any]], midi_name: str, 
                                    report_path: Path) -> None:
    """
    Crea un reporte resumen de todas las mutaciones aplicadas.
    
    Args:
        csv_data: Datos de análisis consolidados
        midi_name: Nombre del archivo MIDI de referencia
        report_path: Ruta donde guardar el reporte
    """
    from datetime import datetime
    
    # Estadísticas generales
    total_mutations = len(csv_data)
    successful_analyses = len([d for d in csv_data if 'error' not in d])
    failed_analyses = total_mutations - successful_analyses
    
    # Agrupar por categorías
    categories = {}
    for data in csv_data:
        cat = data.get('mutation_category', 'unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(data)
    
    # Crear reporte
    report_content = f"""
REPORTE DE MUTACIONES - METRONIA
{'=' * 50}
Archivo de referencia: {midi_name}
Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RESUMEN GENERAL:
- Total de mutaciones aplicadas: {total_mutations}
- Análisis exitosos: {successful_analyses}
- Análisis fallidos: {failed_analyses}

DISTRIBUCIÓN POR CATEGORÍAS:
"""
    
    for category, data_list in categories.items():
        successful_in_cat = len([d for d in data_list if 'error' not in d])
        report_content += f"- {category}: {len(data_list)} mutaciones ({successful_in_cat} exitosas)\n"
    
    report_content += f"""

ARCHIVOS GENERADOS:
- CSV de resumen consolidado: mutations_summary.csv
- Análisis individuales: Cada mutación tiene su directorio en results/NOMBRE_ANALISIS/
- Cambios detallados por mutación: mutation_changes/CATEGORIA_MUTACION/
- Este reporte: summary_report.txt

ESTRUCTURA DE DIRECTORIOS:
results/
├── {midi_name}/
│   ├── mutations_summary.csv
│   ├── summary_report.txt
│   └── mutation_changes/
│       ├── [categoria]_[mutacion]/
│       │   ├── [mutacion]_changes.csv
│       │   └── [mutacion]_summary.txt
└── {midi_name}_[categoria]_[mutacion]/
    ├── onset_errors_detailed.png
    ├── analysis.csv
    ├── beat_spectrum.png
    └── timeline.png
"""
    
    # Guardar reporte
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ Reporte de resumen guardado: {report_path}")

def create_mutation_pipeline(midi_file_path: str, output_base_dir: str = "results", 
                           categories_filter: Optional[List[str]] = None):
    """
    Pipeline completo para generar mutaciones y analizar interpretaciones.
    
    Args:
        midi_file_path: Ruta al archivo MIDI de referencia
        output_base_dir: Directorio base para guardar resultados
        categories_filter: Lista de categorías específicas a aplicar. Si es None, aplica todas.
    """
    print("=" * 60)
    print("🎵 Mutaciones de MetronIA ― Sistema de Análisis de Sincronía de Ritmos Musicales en Audios")
    print("=" * 60)
    
    # Configuración
    midi_path = Path(midi_file_path)
    midi_name = midi_path.stem
    results_dir = Path(output_base_dir)
    results_dir.mkdir(exist_ok=True)
    
    original_excerpt, base_tempo, reference_audio_path = obtener_audio_referencia(midi_file_path, midi_name)
    
    mutation_manager = filtrar_mutaciones(categories_filter)

    successful_mutations = aplicar_mutaciones(mutation_manager, original_excerpt, base_tempo, midi_name, results_dir)
    
    analizar_mutaciones(successful_mutations, reference_audio_path, base_tempo, midi_name, results_dir)

def listar_categorias():
    print("📋 CATEGORÍAS DE MUTACIONES DISPONIBLES:")
    print("=" * 50)
    manager = MutationManager()
    for category_name, category in manager.categories.items():
        print(f"\n🎯 {category_name}")
        print(f"   Descripción: {category.description}")
        print(f"   Mutaciones: {len(category.mutations)}")
        for mutation_name in category.mutations.keys():
            print(f"     - {mutation_name}")
    print("\n💡 Usa --categories seguido de los nombres para filtrar.")
    print("💡 Ejemplo: --categories timing_errors tempo_errors")

def main():
    """Función principal del pipeline."""
    args = metronia_arg_parser()
    
    if args.list_categories:
        listar_categorias()
        return
    
    # Configurar archivo MIDI
    MIDI_FILE_PATH = args.midi
    
    # Verificar que el archivo existe
    if not os.path.exists(MIDI_FILE_PATH):
        print(f"❌ Error: El archivo {MIDI_FILE_PATH} no existe.")
        print("Por favor, verifica la ruta del archivo MIDI.")
        return
    
    # Mostrar configuración
    print("🎵 CONFIGURACIÓN DEL PIPELINE:")
    print(f"  📄 Archivo MIDI: {MIDI_FILE_PATH}")
    print(f"  📁 Directorio salida: {args.output}")
    if args.categories:
        print(f"  🎯 Categorías filtradas: {', '.join(args.categories)}")
    else:
        print(f"  🎯 Categorías: Todas (sin filtro)")
    
    # Ejecutar pipeline
    try:
        csv_file = create_mutation_pipeline(
            midi_file_path=MIDI_FILE_PATH,
            output_base_dir=args.output,
            categories_filter=args.categories
        )
        
    except KeyboardInterrupt:
        print(f"[X] Pipeline interrumpido por el usuario")
        
    except Exception as e:
        print(f"\n❌ Error en el pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
