import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Audio, display
from tqdm import tqdm
from mdtk.plots import save_plot_against_orig
from mdtk.utils import synthesize_from_note_df
from midi_utils import load_midi_with_pretty_midi, load_midi_with_mido, save_excerpt_in_audio
from pathlib import Path

from mutaciones import (
    pitch_shift_mutation,
    faster_tempo_mutation,
    a_lot_faster_tempo_mutation,
    slower_tempo_mutation,
    a_lot_slower_tempo_mutation,
    accelerando_tempo_mutation,
    ritardando_tempo_mutation,
    note_played_too_soon_time_mutation,
    note_played_too_late_time_mutation,
    note_played_too_soon_onset_mutation,
    note_played_too_late_onset_mutation,
    note_held_too_long_mutation,
    note_cut_too_soon_mutation,
    note_missing_mutation,
    note_not_expected_mutation,
    articulated_legato_mutation,
    articulated_staccato_mutation,
    articulated_accentuated_mutation,
    tempo_fluctuation_mutation,
)

midi_file_path = "midi/Acordai-100.mid"
midi_name = Path(midi_file_path).stem

try:
    original_excerpt = load_midi_with_pretty_midi(midi_file_path)
    print("Archivo MIDI cargado exitosamente con pretty_midi")
except:
    # Fallback a mido si pretty_midi falla
    original_excerpt = load_midi_with_mido(midi_file_path)
    print("Archivo MIDI cargado exitosamente con mido")


# Dictionary to organize mutation excerpts by category
mutations = {
    "pitch_errors": {
        "pitch_shift": {
            "function": pitch_shift_mutation,
            "description": "Cambio de altura de una nota",
            "excerpt": None
        }
    },
    "tempo_errors": {
        "faster_tempo": {
            "function": faster_tempo_mutation,
            "description": "Tempo más rápido",
            "excerpt": None
        },
        "a_lot_faster_tempo": {
            "function": a_lot_faster_tempo_mutation,
            "description": "Tempo mucho más rápido",
            "excerpt": None
        },
        "slower_tempo": {
            "function": slower_tempo_mutation,
            "description": "Tempo más lento",
            "excerpt": None
        },
        "a_lot_slower_tempo": {
            "function": a_lot_slower_tempo_mutation,
            "description": "Tempo mucho más lento",
            "excerpt": None
        },
        "accelerando": {s
            "function": accelerando_tempo_mutation,
            "description": "Accelerando - incremento gradual del tempo",
            "excerpt": None
        },
        "ritardando": {
            "function": ritardando_tempo_mutation,
            "description": "Ritardando - disminución gradual del tempo",
            "excerpt": None
        },
        "tempo_fluctuation": {
            "function": tempo_fluctuation_mutation,
            "description": "Fluctuaciones aleatorias del tempo",
            "excerpt": None
        }
    },
    "timing_errors": {
        "note_too_soon_time": {
            "function": note_played_too_soon_time_mutation,
            "description": "Nota tocada demasiado pronto (tiempo)",
            "excerpt": None
        },
        "note_too_late_time": {
            "function": note_played_too_late_time_mutation,
            "description": "Nota tocada demasiado tarde (tiempo)",
            "excerpt": None
        },
        "note_too_soon_onset": {
            "function": note_played_too_soon_onset_mutation,
            "description": "Nota tocada demasiado pronto (onset)",
            "excerpt": None
        },
        "note_too_late_onset": {
            "function": note_played_too_late_onset_mutation,
            "description": "Nota tocada demasiado tarde (onset)",
            "excerpt": None
        }
    },
    "duration_errors": {
        "note_held_too_long": {
            "function": note_held_too_long_mutation,
            "description": "Nota mantenida demasiado tiempo",
            "excerpt": None
        },
        "note_cut_too_soon": {
            "function": note_cut_too_soon_mutation,
            "description": "Nota cortada demasiado pronto",
            "excerpt": None
        }
    },
    "note_errors": {
        "note_missing": {
            "function": note_missing_mutation,
            "description": "Nota faltante",
            "excerpt": None
        },
        "note_not_expected": {
            "function": note_not_expected_mutation,
            "description": "Nota inesperada/extra",
            "excerpt": None
        }
    },
    "articulation_errors": {
        "articulated_legato": {
            "function": articulated_legato_mutation,
            "description": "Articulación legato",
            "excerpt": None
        },
        "articulated_staccato": {
            "function": articulated_staccato_mutation,
            "description": "Articulación staccato",
            "excerpt": None
        },
        "articulated_accentuated": {
            "function": articulated_accentuated_mutation,
            "description": "Articulación acentuada",
            "excerpt": None
        }
    }
}

for category_name, category in mutations.items():
    for mutation_name, mutation_info in category.items():
        try:
            file_name = f"{midi_name}_{mutation_name}"
            print(f"Applying mutation: {mutation_info['description']}")
            mutation_info["excerpt"] = mutation_info["function"](original_excerpt)
            if mutation_info["excerpt"] is not None:
                save_plot_against_orig(orig_excerpt=original_excerpt, list_of_diff_excerpts=[mutation_info["excerpt"]], save_name=file_name)
                save_excerpt_in_audio(excerpt=mutation_info["excerpt"], save_name=file_name)
            else:
                print(f"⚠️ Warning: {mutation_name} returned None - mutation failed")
        except Exception as e:
            print(f"❌ Error applying {mutation_name}: {str(e)}")
            mutation_info["excerpt"] = None