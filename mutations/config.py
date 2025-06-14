from pathlib import Path

FASTER = 1.2
A_LOT_FASTER = 1.5
SLOWER = 0.8
A_LOT_SLOWER = 0.5
ACCELERANDO = 1.3
RITARDANDO = 0.7
NOTE_PLAYED_TOO_SOON = {
    "min_shift": 500,
    "max_shift": 1000
}
NOTE_PLAYED_TOO_LATE = {
    "min_shift": 500,
    "max_shift": 1000
}
MIN_DUR = 100
MAX_DUR = 1000
MIN_VELOCITY = 50
MAX_VELOCITY = 127
ARTICULATED_LEGATO = {
    "max_gap": 2000, # 2 seconds
    "max_notes": 4
}
ARTICULATED_STACCATO = 0.1
ARTICULATED_ACCENTUATED = 0.1
TEMPO_FLUCTUATION = 0.1

NOTE_HELD_TOO_LONG = { 
    "min_shift": 500,
    "max_shift": 10_000
}
NOTE_CUT_TOO_SOON = {
    "min_shift": 50,
    "max_shift": 150
}
SEED = None

# Configuración de rutas usando pathlib.Path
MUTATIONS_PATH = Path("mutts")
MUTATIONS_AUDIO_PATH = MUTATIONS_PATH / "audios"
MUTATIONS_MIDI_PATH = MUTATIONS_PATH / "midis"
MUTATIONS_PLOTS_PATH = MUTATIONS_PATH / "plots"


