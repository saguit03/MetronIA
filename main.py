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
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from tqdm import tqdm

# Imports del proyecto
from mutations.manager import MutationManager
from mutations.midi_utils import load_midi_with_pretty_midi, load_midi_with_mido, save_excerpt_in_audio
from analyzers import analyze_performance
from analyzers import MusicAnalyzer


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


def create_mutation_pipeline(midi_file_path: str, output_base_dir: str = "results"):
    """
    Pipeline completo para generar mutaciones y analizar interpretaciones.
    
    Args:
        midi_file_path: Ruta al archivo MIDI de referencia
        output_base_dir: Directorio base para guardar resultados
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
    
    total_mutations = len(mutation_manager.get_all_mutations())
    print(f"📊 Total de mutaciones disponibles: {total_mutations}")
    for category_name, category in mutation_manager.categories.items():
        print(f"  - {category.description}: {len(category.mutations)} mutaciones")
    
    # 4. APLICAR MUTACIONES Y GENERAR AUDIO
    print(f"\n🎯 Aplicando mutaciones y generando archivos de audio...")
    
    successful_mutations = []
    failed_mutations = []
    
    for category_name, category in mutation_manager.categories.items():
        print(f"\n📂 Categoría: {category.description}")
        
        for mutation_name, mutation in tqdm(category.mutations.items(), 
                                          desc=f"  Procesando {category_name}"):
            try:
                # Aplicar mutación
                success = mutation.apply(original_excerpt)
                
                if success and mutation.excerpt is not None:
                    # Generar archivo de audio
                    file_name = f"{midi_name}_{mutation_name}"
                    audio_path = save_excerpt_in_audio(
                        excerpt=mutation.excerpt,
                        save_name=file_name
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
    
    for category_name, mutation_name, mutation, audio_path in tqdm(successful_mutations, 
                                                                   desc="Analizando mutaciones"):
        try:
            # Realizar análisis completo (sin verbose para evitar spam)
            analysis_result = analyze_performance(
                reference_path=reference_audio_path,
                live_path=audio_path,
                save_name=f"{midi_name}_{mutation_name}",
                config=None,  # Usar configuración por defecto
                verbose=False  # No mostrar resultados por pantalla
            )
            
            # Extraer datos para CSV
            analyzer = MusicAnalyzer()
            csv_row = analyzer.extract_analysis_for_csv(
                beat_result=analysis_result['beat_spectrum'],
                onset_result=analysis_result['onsets'],
                tempo_result=analysis_result['tempo'],
                segment_result=analysis_result['segments'],
                dtw_regular=analysis_result['dtw_regular'],
                rhythm_errors=analysis_result['rhythm_errors']
            )
            
            # Añadir información de la mutación
            csv_row['mutation_category'] = category_name
            csv_row['mutation_name'] = mutation_name
            csv_row['mutation_description'] = mutation.description
            csv_data.append(csv_row)
            
        except Exception as e:
            print(f"❌ Error analizando {category_name}.{mutation_name}: {e}")
    
    print(f"\n📈 Análisis completado: {len(csv_data)} mutaciones analizadas")
    
    # Guardar resultados en CSV
    if csv_data:
        csv_file = results_dir / f"analysis_results_{midi_name}.csv"
        save_analysis_results_to_csv(csv_data, csv_file)
        
        # Estadísticas finales
        print(f"\n🎉 PIPELINE COMPLETADO EXITOSAMENTE")
        print(f"📊 Archivos generados:")
        print(f"  - Audio de referencia: {reference_audio_path}")
        print(f"  - Mutaciones exitosas: {len(successful_mutations)} archivos WAV")
        print(f"  - Resultados CSV: {csv_file}")
        
        return csv_file
    else:
        print("⚠️ No se generaron datos para CSV")
        return None


def main():
    """Función principal del pipeline."""
    
    # CONFIGURACIÓN: Cambiar esta ruta por el MIDI deseado
    MIDI_FILE_PATH = "midi/Acordai-100.mid"
    
    # Verificar que el archivo existe
    if not os.path.exists(MIDI_FILE_PATH):
        print(f"❌ Error: El archivo {MIDI_FILE_PATH} no existe.")
        print("Por favor, verifica la ruta del archivo MIDI.")
        return
    
    # Ejecutar pipeline
    try:
        csv_file = create_mutation_pipeline(MIDI_FILE_PATH)
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
