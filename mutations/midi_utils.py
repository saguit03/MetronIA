
import pretty_midi
import mido
import pandas as pd
from midi2audio import FluidSynth
from pathlib import Path
import scipy.io.wavfile as wavfile
import numpy as np
from mdtk.utils import synthesize_from_note_df
from mutations.config import MUTATIONS_MIDI_PATH

def save_excerpt_in_audio(excerpt, save_name, soundfont_path=None, sample_rate=16000):
    audio_data = synthesize_from_note_df(excerpt)
    audio_normalized = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    Path(MUTATIONS_MIDI_PATH).mkdir(exist_ok=True)
    output_filename = Path(MUTATIONS_MIDI_PATH) / f"{save_name}.wav"
    wavfile.write(output_filename, sample_rate, audio_normalized)
    return output_filename

def load_midi_with_mido(midi_file_path, bpm=120):
    mid = mido.MidiFile(midi_file_path)
    
    notes = []
    
    for i, track in enumerate(mid.tracks):
        track_time = 0
        for msg in track:
            track_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                # Buscar el note_off correspondiente
                temp_time = track_time
                duration = 0
                for future_msg in track[track.index(msg)+1:]:
                    temp_time += future_msg.time
                    if (future_msg.type == 'note_off' and future_msg.note == msg.note) or \
                       (future_msg.type == 'note_on' and future_msg.note == msg.note and future_msg.velocity == 0):
                        duration = temp_time - track_time
                        break
                
                # Convertir ticks a milisegundos
                ticks_per_beat = mid.ticks_per_beat
                # Asumimos 120 BPM por defecto si no se especifica
                ms_per_tick = (60000 / bpm) / ticks_per_beat
                
                notes.append({
                    'onset': track_time * ms_per_tick,
                    'pitch': msg.note,
                    'dur': duration * ms_per_tick,
                    'velocity': msg.velocity,
                    'track': i
                })
    
    return pd.DataFrame(notes)

def load_midi_with_pretty_midi(midi_file_path):
    """
    Carga un archivo MIDI usando pretty_midi y lo convierte a DataFrame
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    
    notes = []
    
    for track_idx, instrument in enumerate(midi_data.instruments):
        for note in instrument.notes:
            notes.append({
                'onset': note.start * 1000,  # Convertir segundos a milisegundos
                'pitch': note.pitch,
                'dur': (note.end - note.start) * 1000,  # Duración en milisegundos
                'velocity': note.velocity,
                'track': track_idx
            })
    
    df = pd.DataFrame(notes)
    # Ordenar por tiempo de inicio
    df = df.sort_values('onset').reset_index(drop=True)
    
    return df