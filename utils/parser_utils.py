import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List

from mutations.catalog import MutationCatalog
from mutations.globals import DEFAULT_MIDI
from .config import VERBOSE_LOGGING

FILES_LIMIT = 10

EPILOG = """Ejemplos de uso:
# Aplicar todas las mutaciones a un MIDI por defecto
python mutar_y_analizar.py

# Aplicar múltiples categorías específicas
python mutar_y_analizar.py --categories timing tempo

# Usar un archivo MIDI específico
python mutar_y_analizar.py --midi path/to/your/file.mid

# Analizar una carpeta con varios archivos MIDI
python mutar_y_analizar.py --all_midi path/to/your/midi/directory

# Establecer un límite de ficheros a procesar
python mutar_y_analizar.py --all_midi path/to/your/midi/directory --files_limit 10
  
Categorías disponibles:
  - pitch: Errores de altura de las notas
  - tempo: Errores relacionados con el tempo
  - timing: Errores de timing de las notas
  - duration: Errores de duración de las notas
  - note: Errores de presencia de notas
  - articulation: Errores de articulación
"""


def get_output_directory(args) -> str:
    base_dir = args.output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / Path(timestamp)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def mutts_pipeline_arg_parser():
    parser = argparse.ArgumentParser(
        description="Pipeline MetronIA: Genera mutaciones y analiza interpretaciones musicales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG)

    midi_group = parser.add_mutually_exclusive_group()

    midi_group.add_argument(
        '--midi',
        type=str,
        default=DEFAULT_MIDI,
        nargs='*',
        metavar='MIDI_FILE',
        help=f"Ruta al archivo MIDI de referencia (default: {DEFAULT_MIDI})"
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

    parser.add_argument(
        '--files_limit',
        type=int,
        default=FILES_LIMIT,
        help=f"Límite de archivos MIDI a procesar (default: {FILES_LIMIT})"
    )

    return parser.parse_args()


def listar_categorias():
    print("📋 CATEGORÍAS DE MUTACIONES DISPONIBLES:")
    print("=" * 50)
    manager = MutationCatalog()
    for category_name, category in manager.categories.items():
        print(f"\n🎯 {category_name}")
        print(f"   Descripción: {category.description}")
        print(f"   Mutaciones: {len(category.mutations)}")
        for mutation_name in category.mutations.keys():
            print(f"     - {mutation_name}")
    print("\n💡 Usa --categories seguido de los nombres para filtrar.")
    print("💡 Ejemplo: --categories timing tempo")


def filtrar_mutaciones_por_categoria(categories_filter):
    mutation_manager = MutationCatalog()
    if categories_filter:
        if VERBOSE_LOGGING: print(f"🎯 Filtrando categorías: {', '.join(categories_filter)}")

        available_categories = list(mutation_manager.categories.keys())
        invalid_categories = [cat for cat in categories_filter if cat not in available_categories]

        if invalid_categories:
            print(f"❌ Categorías no válidas: {', '.join(invalid_categories)}")
            print(f"📋 Categorías disponibles: {', '.join(available_categories)}")
            return None

        filtered_categories = {name: category for name, category in mutation_manager.categories.items()
                               if name in categories_filter}
        mutation_manager.categories = filtered_categories

    total_mutations = len(mutation_manager.get_all_mutations())
    print(f"📊 Total de mutaciones a aplicar: {total_mutations}")
    for category_name, category in mutation_manager.categories.items():
        print(f"  - {category.description}: {len(category.mutations)} mutaciones")
    return mutation_manager


def get_midi_files_from_directory(directory_path: str) -> List[str]:
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
            return []
        midi_files_to_process = get_midi_files_from_directory(args.all_midi)
        if not midi_files_to_process:
            print(f"⚠️ No se encontraron archivos .mid en el directorio: {args.all_midi}")
            return []
    else:
        if not args.midi or len(args.midi) == 0:
            midi_files_to_process = [DEFAULT_MIDI]
        else:
            for midi_file in args.midi:
                if os.path.isfile(midi_file) and midi_file.lower().endswith(('.mid', '.midi')):
                    midi_files_to_process.append(midi_file)
    if not midi_files_to_process: midi_files_to_process = [DEFAULT_MIDI]
    return midi_files_to_process


def get_files_limit(args) -> int:
    if args.files_limit <= 0:
        return FILES_LIMIT
    return args.files_limit
