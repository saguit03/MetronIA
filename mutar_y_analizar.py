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

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, Any, List

import matplotlib
import pandas as pd
from tqdm import tqdm

from analyzers import MusicAnalyzer
from analyzers.config import VERBOSE_LOGGING
from mutations.manager import MutationManager
from mutations.midi_utils import save_mutation_complete
from mutations.validator import run_validation_analysis, generate_average_validation_report, generate_category_validation_reports
from utils.audio_utils import obtener_audio_de_midi

DEBUG = False

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
    parser = argparse.ArgumentParser(
        description="Pipeline MetronIA: Genera mutaciones y analiza interpretaciones musicales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG)

    midi_group = parser.add_mutually_exclusive_group()

    midi_group.add_argument(
        '--midi',
        type=str,
        default="midi/Acordai-110.mid",
        help='Ruta al archivo MIDI de referencia (default: midi/Acordai-110.mid)'
    )

    midi_group.add_argument(
        '--all_midi',
        type=str,
        help='Ruta a un directorio para procesar todos los archivos .mid que contiene.'
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
    if VERBOSE_LOGGING: print(f"✅ Resultados guardados en CSV: {output_file}")

    return df


def filtrar_mutaciones(categories_filter):
    mutation_manager = MutationManager()

    if categories_filter:
        if VERBOSE_LOGGING: print(f"🎯 Filtrando categorías: {', '.join(categories_filter)}")

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
    # Crear directorio para el conjunto de mutaciones con sufijo "_Mutaciones"
    mutations_base_dir = results_dir / f"{midi_name}_Mutaciones"
    mutations_base_dir.mkdir(exist_ok=True)

    for category_name, category in mutation_manager.categories.items():
        for mutation_name, mutation in category.mutations.items():
            try:
                success = mutation.apply(
                    original_excerpt,
                    tempo=base_tempo,
                    output_dir=str(mutations_base_dir)
                )

                if success and mutation.excerpt is not None:
                    file_name = f"{midi_name}_{mutation_name}"
                    audio_path, midi_path, calculated_tempo = save_mutation_complete(
                        mutation_result=mutation,
                        mutation_name=midi_name,
                        save_name=file_name,
                        base_tempo=base_tempo
                    )

                    mutation.set_audio_path(audio_path)
                    mutation.set_midi_path(midi_path)

                    successful_mutations.append((category_name, mutation_name, mutation, audio_path, original_excerpt))
                else:
                    failed_mutations.append((category_name, mutation_name, mutation.error or "Unknown error"))

            except Exception as e:
                failed_mutations.append((category_name, mutation_name, str(e)))

    if failed_mutations:
        print(f"\n⚠️ Mutaciones fallidas:")
        for category, name, error in failed_mutations:
            print(f"  - {category}.{name}: {error}")  # 5. ANÁLISIS DE CADA MUTACIÓN VS ORIGINAL
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
    mutations_base_dir = results_dir / f"{midi_name}_Mutaciones"

    progress_bar = tqdm(successful_mutations, desc=f"Iniciando análisis de {midi_name}", unit="mutación",
                        dynamic_ncols=True, ascii=True)

    for category_name, mutation_name, mutation, audio_path, original_excerpt in progress_bar:
        try:
            # Actualizar descripción de la barra de progreso
            progress_bar.set_description(f"{midi_name}: Analizando {mutation_name}")

            # Crear nombre único para cada análisis individual según especificación: MIDI_NAME_mutation_name
            analysis_name = f"{midi_name}_{mutation_name}"

            # Crear directorio específico para este análisis dentro del directorio de mutaciones
            analysis_dir = mutations_base_dir / analysis_name
            analysis_dir.mkdir(parents=True, exist_ok=True)            # El análisis completo genera CSV y visualizaciones en el directorio específico
            analysis_result = analyzer.comprehensive_analysis(
                reference_path=reference_audio_path,
                live_path=audio_path,
                save_name=analysis_name,  # Nombre único para cada análisis
                save_dir=str(analysis_dir),  # Directorio específico donde guardar
                reference_tempo=base_tempo,
                mutation_name=mutation_name,  # Nombre de la mutación para el archivo CSV
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
                    'correct_matches': len(
                        [m for m in dtw_onset_result.matches if m.classification.value == 'correct']),
                    'late_matches': len([m for m in dtw_onset_result.matches if m.classification.value == 'late']),
                    'early_matches': len([m for m in dtw_onset_result.matches if m.classification.value == 'early']),
                    'missing_onsets': len(dtw_onset_result.missing_onsets),
                    'extra_onsets': len(dtw_onset_result.extra_onsets),
                    'beat_spectrum_similar': analysis_result['beat_spectrum'].is_similar,
                    'beat_spectrum_max_difference': f"{analysis_result['beat_spectrum'].max_difference:.3f}",
                    'tempo_reference_bpm': f"{analysis_result['tempo'].tempo_ref:.2f}",
                    'tempo_live_bpm': f"{analysis_result['tempo'].tempo_live:.2f}",
                    'tempo_difference_bpm': f"{analysis_result['tempo'].difference:.2f}",
                }

                csv_data.append(csv_row)
                # Actualizar postfix con información del análisis actual
                progress_bar.set_postfix_str(f"✅ {category_name}")
            else:
                progress_bar.set_postfix_str(f"⚠️ Sin resultados DTW")


        except Exception as e:
            progress_bar.set_postfix_str(f"❌ Error: {str(e)}")
            csv_data.append({
                'mutation_category': category_name,
                'mutation_name': mutation_name,
                'error': str(e), 'audio_file_path': str(audio_path),
                'reference_audio_path': str(reference_audio_path)
            })  # Guardar resultados consolidados en directorio del MIDI de referencia

    # Cerrar la barra de progreso con mensaje final
    progress_bar.set_description("Análisis completado")
    progress_bar.close()

    if csv_data:
        # Crear directorio para el conjunto de mutaciones usando NOMBRE_REFERENCIA_MUTACION
        mutations_summary_dir = results_dir / f"{midi_name}_Mutaciones"
        mutations_summary_dir.mkdir(parents=True, exist_ok=True)

        # Guardar CSV consolidado con resumen de todas las mutaciones
        csv_file = mutations_summary_dir / "mutations_summary.csv"
        save_analysis_results_to_csv(csv_data, csv_file)


def create_mutation_pipeline(mutation_manager, midi_file_path: str, output_base_dir: str = "results") -> Dict[
    str, float]:
    """
    Pipeline completo para generar mutaciones y analizar interpretaciones.
    
    Args:
        midi_file_path: Ruta al archivo MIDI de referencia
        output_base_dir: Directorio base para guardar resultados
        categories_filter: Lista de categorías específicas a aplicar. Si es None, aplica todas.
        
    Returns:
        Dict con las métricas de validación
    """

    # Configuración
    midi_path = Path(midi_file_path)
    midi_name = midi_path.stem
    results_dir = Path(output_base_dir)
    results_dir.mkdir(exist_ok=True)

    reference_audio_path = midi_name + "_reference"
    original_excerpt, base_tempo, reference_audio_path = obtener_audio_de_midi(midi_file_path, midi_name)

    if original_excerpt is None:
        print(f"❌ Error: obtener_audio_de_midi no pudo procesar el archivo {midi_file_path}.")
        return {}

    successful_mutations = aplicar_mutaciones(mutation_manager, original_excerpt, base_tempo, midi_name, results_dir)

    analizar_mutaciones(successful_mutations, reference_audio_path, base_tempo, midi_name, results_dir)

    # Ejecutar validación de los resultados y retornar métricas
    validation_metrics = run_validation_analysis(midi_name, results_dir)
    return validation_metrics


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


def get_midi_files_from_directory(directory_path: str) -> List[str]:
    """
    Encuentra todos los archivos .mid en un directorio y sus subdirectorios.
    
    Args:
        directory_path: Ruta al directorio a explorar.
        
    Returns:
        Lista de rutas a los archivos .mid encontrados.
    """
    midi_files = []
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' no es un directorio válido.")
        return []

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path) and item.lower().endswith(('.mid', '.midi')):
            midi_files.append(item_path)
    return midi_files


def get_midi_files_to_process(args) -> List[str]:
    midi_files_to_process = []
    if args.all_midi:
        if not os.path.isdir(args.all_midi):
            print(f"❌ Error: El directorio especificado en --all_midi no existe: {args.all_midi}")
            return
        midi_files_to_process = get_midi_files_from_directory(args.all_midi)
        if not midi_files_to_process:
            print(f"⚠️ No se encontraron archivos .mid en el directorio: {args.all_midi}")
            return
        print(f"🎶 Encontrados {len(midi_files_to_process)} archivos MIDI para procesar.")
    else:
        # Usar el archivo MIDI individual especificado (o el por defecto)
        if not os.path.exists(args.midi):
            print(f"❌ Error: El archivo {args.midi} no existe.")
            print("Por favor, verifica la ruta del archivo MIDI.")
            return
        midi_files_to_process.append(args.midi)
    return midi_files_to_process


def main():
    """Función principal del pipeline."""
    # Configurar matplotlib para evitar warnings de figuras abiertas
    matplotlib.rcParams['figure.max_open_warning'] = 0
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='More than 20 figures have been opened')
    
    args = metronia_arg_parser()

    if args.list_categories:
        listar_categorias()
        return

    if DEBUG:
        args.categories = ['timing_errors', 'note_errors']

    midi_files_to_process = get_midi_files_to_process(args)
    if not midi_files_to_process:
        return

    print("=" * 90)
    print("🎵 Mutaciones de MetronIA ― Sistema de Análisis de Sincronía de Ritmos Musicales en Audios")
    print("=" * 90)

    print("\n🎵 CONFIGURACIÓN GENERAL DEL PIPELINE:")
    print(f"  📁 Directorio salida: {args.output}")
    if args.categories:
        print(f"  🎯 Categorías filtradas: {', '.join(args.categories)}")
    else:
        print(f"  🎯 Categorías: Todas (sin filtro)")

    print("=" * 90)

    all_validation_metrics = [] 
    processed_files = [] 
    
    mutation_manager = filtrar_mutaciones(args.categories)

    midi_progress = tqdm(midi_files_to_process, desc="Procesando archivos MIDI", unit="archivo", dynamic_ncols=True)

    for midi_file_path in midi_progress:
        try:
            # # Obtener solo el nombre del archivo para mostrar en la barra
            midi_filename = Path(midi_file_path).name
            midi_progress.set_description(f"Procesando {midi_filename}")

            validation_metrics = create_mutation_pipeline(
                mutation_manager=mutation_manager,
                midi_file_path=midi_file_path,
                output_base_dir=args.output
            )

            if validation_metrics:
                all_validation_metrics.append(validation_metrics)
                processed_files.append(midi_file_path)
            else:
                tqdm.write(f"⚠️ {midi_filename} sin métricas")

        except KeyboardInterrupt:
            tqdm.write(f"[X] Pipeline interrumpido por el usuario.")
            break

        except Exception as e:
            tqdm.write(f"\n❌ Error procesando {midi_filename}: {e}")
            midi_progress.set_postfix_str(f"❌ Error")
            import traceback
            traceback.print_exc()

    midi_progress.close()
    
    if processed_files:
        tqdm.write(f"✅ Pipeline de mutaciones completado exitosamente para {len(processed_files)} archivos")
    
    if len(midi_files_to_process) > 1:
        generate_average_validation_report(
            all_validation_metrics,
            processed_files,
            args.categories,
            args.output
        )
        
        # Generar reportes por categoría
        generate_category_validation_reports(
            all_validation_metrics,
            processed_files,
            args.categories,
            args.output
        )


if __name__ == "__main__":
    main()
