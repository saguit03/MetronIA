import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Audio, display
from tqdm import tqdm
from mdtk.plots import save_plot_against_orig
from mdtk.utils import synthesize_from_note_df
from midi_utils import load_midi_with_pretty_midi, load_midi_with_mido, save_excerpt_in_audio
from pathlib import Path
from mutations_manager import MutationManager

midi_file_path = "midi/Acordai-100.mid"
midi_name = Path(midi_file_path).stem

try:
    original_excerpt = load_midi_with_pretty_midi(midi_file_path)
    print("Archivo MIDI cargado exitosamente con pretty_midi")
except:
    # Fallback a mido si pretty_midi falla
    original_excerpt = load_midi_with_mido(midi_file_path)
    print("Archivo MIDI cargado exitosamente con mido")

# Crear el gestor de mutaciones
mutation_manager = MutationManager()

# Aplicar todas las mutaciones
print("\n=== APLICANDO MUTACIONES ===")
for category_name, category in mutation_manager.categories.items():
    print(f"\nCategoría: {category.description}")
    
    for mutation_name, mutation in category.mutations.items():
        try:
            file_name = f"{midi_name}_{mutation_name}"
            print(f"  Aplicando: {mutation.description}")
            success = mutation.apply(original_excerpt)
            if success:
                save_plot_against_orig(
                    orig_excerpt=original_excerpt, 
                    list_of_diff_excerpts=[mutation.excerpt], 
                    save_name=file_name
                )
                audio_path = save_excerpt_in_audio(
                    excerpt=mutation.excerpt, 
                    save_name=file_name
                )
                mutation.set_path(audio_path)
                
            else:
                print(f"  Error al aplicar la mutación: {mutation.error}")
                
        except Exception as e:
            mutation.success = False
            mutation.error = str(e)

mutation_manager.print_summary()

# Exportar rutas de archivos generados
paths = mutation_manager.export_paths()
print(f"\n=== ARCHIVOS GENERADOS ({len(paths)}) ===")
for key, path in paths.items():
    print(f"  {key}: {path}")