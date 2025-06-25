#!/usr/bin/env python3
"""
# Pipeline de mutaciones
0. Obtener un MIDI de referencia y las categorías a aplicar
1. Crear las mutaciones sobre el MIDI de referencia
2. Analizar con MetronIA las mutaciones generadas exitosamente
3. Validar las mutaciones
"""

from pathlib import Path
from typing import Dict

import traceback
from tqdm import tqdm

from analyzers import MetronIA
from mutations.validator import run_validation_analysis
from utils.audio_utils import obtener_audio_de_midi
from utils.mutation_utils import aplicar_mutaciones, analizar_mutaciones
from utils.parser_utils import mutts_pipeline_arg_parser, get_output_directory, listar_categorias, get_midi_files_to_process, filtrar_mutaciones_por_categoria, get_files_limit

def create_mutation_pipeline(mutation_manager, midi_file_path: str, output_base_dir: str) -> Dict[str, float]:
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

    analyzer = MetronIA()

    analizar_mutaciones(analyzer, successful_mutations, reference_audio_path, base_tempo, midi_name, results_dir)

    validation_metrics = run_validation_analysis(midi_name, results_dir)
    return validation_metrics

def main():
    args = mutts_pipeline_arg_parser()

    if args.list_categories:
        listar_categorias()
        return

    print("=" * 90)
    print("🎵 Mutaciones de MetronIA ― Sistema de Análisis de Sincronía de Ritmos Musicales en Audios")
    print("=" * 90)

    midi_files_to_process = get_midi_files_to_process(args)
    if not midi_files_to_process:
        print("❌ No se encontraron archivos MIDI para procesar. Asegúrate de especificar un archivo o directorio válido.")
        return

    output_dir = get_output_directory(args)

    print("\n🎵 CONFIGURACIÓN GENERAL DEL PIPELINE:")
    print(f"  📁 Directorio salida: {output_dir}")
    if args.categories:
        print(f"  🎯 Categorías filtradas: {', '.join(args.categories)}")
    else:
        print(f"  🎯 Categorías: Todas (sin filtro)")

    print("=" * 90)

    processed_files = [] 
    mutation_manager = filtrar_mutaciones_por_categoria(args.categories)
    midi_progress = tqdm(midi_files_to_process, desc="Procesando archivos MIDI", unit="archivo", dynamic_ncols=True)

    files_limit = get_files_limit(args)

    cont = 0
    for midi_file_path in midi_progress:
        if cont >= files_limit:
            tqdm.write(f"** Límite de archivos alcanzado({files_limit}): deteniendo procesamiento. **")
            break
        
        cont += 1
        try:
            midi_filename = Path(midi_file_path).name
            midi_progress.set_description(f"Procesando {midi_filename}")

            validation_metrics = create_mutation_pipeline(
                mutation_manager=mutation_manager,
                midi_file_path=midi_file_path,
                output_base_dir=output_dir
            )

            if validation_metrics:
                processed_files.append(midi_file_path)
            else:
                tqdm.write(f"⚠️ {midi_filename} sin métricas")

        except KeyboardInterrupt:
            tqdm.write(f"[X] Pipeline interrumpido por el usuario.")
            break
        except Exception as e:
            tqdm.write(f"\n❌ Error procesando {midi_filename}: {e}")
            midi_progress.set_postfix_str(f"❌ Error")
            traceback.print_exc()

    midi_progress.close()
    if processed_files:
        tqdm.write(f"✅ Pipeline de mutaciones completado para {len(processed_files)} archivos")

if __name__ == "__main__":
    main()
