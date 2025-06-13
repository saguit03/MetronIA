#!/usr/bin/env python3
"""
Pipeline principal de MetronIA: Genera mutaciones y analiza interpretaciones musicales.

Este script implementa el pipeline completo:
1. Crear mutaciones a partir de un MIDI de referencia
2. Convertir todas las mutaciones y el MIDI de referencia a WAV
3. Analizar cada mutación comparándola con el original
4. Guardar los resultados de análisis en formato CSV
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from tqdm import tqdm

# Imports del proyecto
from mutations.manager import MutationManager
from mutations.midi_utils import load_midi_with_pretty_midi, load_midi_with_mido, save_excerpt_in_audio, save_mutation_complete, extract_tempo_from_midi
from analyzers import analyze_performance
from analyzers import MusicAnalyzer, ResultVisualizer

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
    print("🎵 PIPELINE METRONIA - ANÁLISIS DE INTERPRETACIONES MUSICALES")
    print("=" * 60)
    
    # Configuración
    midi_path = Path(midi_file_path)
    midi_name = midi_path.stem
    results_dir = Path(output_base_dir)
    
    # Crear directorio de resultados
    results_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Directorio de resultados: {results_dir.absolute()}")
      # 1. CARGAR MIDI ORIGINAL
    print(f"\n🎼 Cargando MIDI original: {midi_file_path}")
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
    
    # Extraer tempo del MIDI original
    print(f"\n🎵 Extrayendo tempo del MIDI original...")
    base_tempo = extract_tempo_from_midi(midi_file_path)
    print(f"✅ Tempo detectado: {base_tempo} BPM")
    
    # 2. GENERAR AUDIO DE REFERENCIA
    print(f"\n🎧 Generando audio de referencia...")
    try:
        reference_audio_path = save_excerpt_in_audio(
            excerpt=original_excerpt,
            save_name=f"{midi_name}_reference"
        )
        print(f"✅ Audio de referencia guardado: {reference_audio_path}")
    except Exception as e:
        print(f"❌ Error generando audio de referencia: {e}")
        return None
      # 3. CREAR GESTOR DE MUTACIONES
    print(f"\n🧬 Inicializando gestor de mutaciones...")
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
    
    # 4. APLICAR MUTACIONES Y GENERAR AUDIO
    print(f"\n🎯 Aplicando mutaciones y generando archivos de audio...")
    
    successful_mutations = []
    failed_mutations = []
    for category_name, category in mutation_manager.categories.items():
        for mutation_name, mutation in category.mutations.items():
            try:
                # Aplicar mutación con el tempo correcto
                success = mutation.apply(original_excerpt, tempo=base_tempo)
                
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
    
    print(f"\n📊 Resumen de generación de mutaciones:")
    print(f"  ✅ Exitosas: {len(successful_mutations)}")
    print(f"  ❌ Fallidas: {len(failed_mutations)}")
    
    if failed_mutations:
        print(f"\n⚠️ Mutaciones fallidas:")
        for category, name, error in failed_mutations:
            print(f"  - {category}.{name}: {error}")
      # 5. ANÁLISIS DE CADA MUTACIÓN VS ORIGINAL
    print(f"\n🔬 Analizando mutaciones contra el original...")
    
    csv_data = []
    
    # Crear analizador y visualizador una sola vez
    analyzer = MusicAnalyzer()
    result_visualizer = ResultVisualizer(output_dir=results_dir / "mutations")
    
    for category_name, mutation_name, mutation, audio_path in tqdm(successful_mutations, 
                                                                   desc="Analizando mutaciones"):
        try:
            analysis_result = analyzer.comprehensive_analysis(
                reference_path=reference_audio_path,
                live_path=audio_path,
                save_name=f"{midi_name}_{mutation_name}",
                reference_tempo=base_tempo  # Pasar el tempo extraído del MIDI
            )
            
            # Obtener datos para CSV usando el método directo de OnsetDTWAnalysisResult
            dtw_onset_result = analysis_result.get('dtw_onsets')
            if dtw_onset_result:
                csv_row = dtw_onset_result.get_csv_data(category_name, mutation_name)
                
                # Agregar datos adicionales del análisis completo
                csv_row.update({
                    'beat_spectrum_similar': 'Similar' if analysis_result['beat_spectrum'].is_similar else 'Diferencias significativas',
                    'beat_spectrum_max_difference': f"{analysis_result['beat_spectrum'].max_difference:.3f}",
                    'tempo_reference_bpm': f"{analysis_result['tempo'].tempo_ref:.2f}",
                    'tempo_live_bpm': f"{analysis_result['tempo'].tempo_live:.2f}",
                    'tempo_difference_bpm': f"{analysis_result['tempo'].difference:.2f}",
                    'tempo_similar': 'Tempo similar' if analysis_result['tempo'].is_similar else 'Diferencia significativa de tempo',
                    'structure_measures_ref': analysis_result['segments'].get('measures_ref', 0),
                    'structure_measures_live': analysis_result['segments'].get('measures_live', 0),
                    'structure_compatible': 'Estructura compatible' if analysis_result['segments'].get('overall_compatible', False) else 'Estructura incompatible',
                    'dtw_regular': analysis_result.get('dtw_analysis', {}).get('overall_assessment', 'N/A'),
                    'audio_file_path': str(audio_path),
                    'reference_audio_path': str(reference_audio_path)
                })
                
                csv_data.append(csv_row)
                
                # Generar visualizaciones detalladas para algunas mutaciones importantes
                # (para evitar generar demasiados archivos, solo para las primeras de cada categoría)
                if len([row for row in csv_data if row['mutation_category'] == category_name]) <= 2:
                    try:
                        result_visualizer.plot_onset_errors_detailed(
                            dtw_onset_result=dtw_onset_result,
                            save_name=f"{midi_name}_{category_name}_{mutation_name}",
                            show_plot=False
                        )
                    except Exception as viz_error:
                        print(f"⚠️ Error generando visualización para {mutation_name}: {viz_error}")
            else:
                print(f"⚠️ No se obtuvieron resultados DTW para {category_name}.{mutation_name}")
                
        except Exception as e:
            print(f"❌ Error analizando {category_name}.{mutation_name}: {e}")
            # Agregar una fila con datos mínimos para no perder la información
            csv_data.append({
                'mutation_category': category_name,
                'mutation_name': mutation_name,                'error': str(e),
                'audio_file_path': str(audio_path),
                'reference_audio_path': str(reference_audio_path)
            })
    
    print(f"\n📈 Análisis completado: {len(csv_data)} mutaciones analizadas")
    
    # Guardar resultados en CSV
    if csv_data:
        csv_file = results_dir / f"analysis_results_{midi_name}.csv"
        save_analysis_results_to_csv(csv_data, csv_file)
        
        # Generar un reporte resumen usando ResultVisualizer
        try:
            # Crear un análisis sintético para el reporte general
            successful_analyses = [row for row in csv_data if 'error' not in row]
            if successful_analyses:
                print(f"\n📊 Generando reporte de resumen...")
                
                # Calcular estadísticas generales
                summary_stats = {
                    'total_mutations': len(csv_data),
                    'successful_analyses': len(successful_analyses),
                    'failed_analyses': len(csv_data) - len(successful_analyses),
                    'categories_analyzed': len(set(row.get('mutation_category', 'unknown') for row in csv_data)),
                    'avg_consistency_rate': sum(float(row.get('dtw_onsets_consistency_rate', '0%').replace('%', '')) 
                                              for row in successful_analyses) / len(successful_analyses) if successful_analyses else 0
                }
                
                summary_report_path = results_dir / f"summary_report_{midi_name}.txt"
                with open(summary_report_path, 'w', encoding='utf-8') as f:
                    f.write(f"""
REPORTE RESUMEN DEL PIPELINE METRONIA
{'='*50}
Archivo MIDI analizado: {midi_file_path}
Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

ESTADÍSTICAS GENERALES:
- Total mutaciones procesadas: {summary_stats['total_mutations']}
- Análisis exitosos: {summary_stats['successful_analyses']}
- Análisis fallidos: {summary_stats['failed_analyses']}
- Categorías analizadas: {summary_stats['categories_analyzed']}
- Tasa de consistencia promedio: {summary_stats['avg_consistency_rate']:.1f}%

ARCHIVOS GENERADOS:
- Audio de referencia: {reference_audio_path}
- Archivos de mutaciones: {len(successful_mutations)} archivos WAV
- Resultados CSV: {csv_file}
- Visualizaciones: {results_dir / 'visualizations'}
- Este reporte: {summary_report_path}

ANÁLISIS POR CATEGORÍA:
""")
                    
                    # Estadísticas por categoría
                    for category in set(row.get('mutation_category', 'unknown') for row in successful_analyses):
                        category_data = [row for row in successful_analyses if row.get('mutation_category') == category]
                        if category_data:
                            avg_consistency = sum(float(row.get('dtw_onsets_consistency_rate', '0%').replace('%', '')) 
                                                for row in category_data) / len(category_data)
                            f.write(f"- {category}: {len(category_data)} mutaciones, consistencia promedio: {avg_consistency:.1f}%\n")
                
                print(f"  📋 Reporte de resumen guardado en: {summary_report_path}")
        
        except Exception as e:
            print(f"⚠️ Error generando reporte de resumen: {e}")
        
        return csv_file
    else:
        print("⚠️ No se generaron datos para CSV")
        return None


def main():
    """Función principal del pipeline."""
    
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description="Pipeline MetronIA: Genera mutaciones y analiza interpretaciones musicales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

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
    )
    
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
    
    args = parser.parse_args()
    
    # Si se pide listar categorías, mostrarlas y salir
    if args.list_categories:
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
        
        if csv_file:
            print(f"\n🎵 Pipeline completado para: {MIDI_FILE_PATH}")
            print(f"✅ Resultados guardados en: {csv_file}")
        else:
            print(f"\n⚠️ Pipeline completado pero sin resultados CSV")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Pipeline interrumpido por el usuario")
        
    except Exception as e:
        print(f"\n❌ Error en el pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
