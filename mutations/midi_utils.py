from pathlib import Path

import mido
import numpy as np
import pandas as pd
import pretty_midi

from mdtk.utils import synthesize_from_note_df
from mutations.config import MUTATIONS_PATH, MUTATIONS_AUDIO_PATH, MUTATIONS_MIDI_PATH
from mutations.results import MutationResult
from utils.audio_utils import save_audio


def save_excerpt_in_audio(excerpt, dir_name, save_name, sample_rate=16000):
    audio_data = synthesize_from_note_df(excerpt)
    audio_normalized = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    save_dir = MUTATIONS_PATH / Path(dir_name) / MUTATIONS_AUDIO_PATH
    output_filename = save_audio(audio_normalized, save_name, save_dir, sample_rate)
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
                for future_msg in track[track.index(msg) + 1:]:
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


def save_excerpt_in_midi(excerpt, dir_name, save_name, tempo=120):
    """
    Guarda un excerpt (DataFrame con notas) como archivo MIDI.
    
    Parameters
    ----------
    excerpt : pd.DataFrame
        DataFrame con columnas: onset, pitch, dur, velocity, track
    save_name : str
        Nombre del archivo (sin extensión)
    tempo : int
        Tempo en BPM para el archivo MIDI (default: 120)
    
    Returns
    -------
    Path
        Ruta del archivo MIDI guardado
    """

    save_dir = MUTATIONS_PATH / Path(dir_name) / MUTATIONS_MIDI_PATH
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_filename = Path(save_dir) / f"{save_name}.mid"

    # Crear objeto PrettyMIDI
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    # Agrupar notas por track
    tracks = excerpt.groupby('track') if 'track' in excerpt.columns else [('default', excerpt)]

    for track_id, track_notes in tracks:
        # Crear instrumento (Piano por defecto)
        instrument = pretty_midi.Instrument(program=0, name=f"Track_{track_id}")

        for _, note_row in track_notes.iterrows():
            # Convertir milisegundos a segundos
            start_time = note_row['onset'] / 1000.0
            duration = note_row['dur'] / 1000.0
            end_time = start_time + duration

            # Crear nota MIDI
            note = pretty_midi.Note(
                velocity=int(note_row['velocity']),
                pitch=int(note_row['pitch']),
                start=start_time,
                end=end_time
            )

            instrument.notes.append(note)

        # Agregar instrumento al MIDI
        midi_data.instruments.append(instrument)

    # Guardar archivo
    midi_data.write(str(output_filename))

    return output_filename


def save_mutation_complete(mutation_result: MutationResult, mutation_name, save_name, base_tempo=120,
                           soundfont_path=None, sample_rate=16000):
    """
    Guarda una mutación completa como archivo de audio (WAV) y MIDI con el tempo correcto.
    
    Parameters
    ----------
    mutation_result : MutationResult
        Resultado de la mutación aplicada
    save_name : str
        Nombre base del archivo (sin extensión)
    base_tempo : int
        Tempo base del MIDI original en BPM (default: 120)
    soundfont_path : str, optional
        Ruta al soundfont para síntesis de audio
    sample_rate : int
        Sample rate para el archivo de audio (default: 16000)
    
    Returns
    -------
    tuple
        Tupla con (ruta_audio, ruta_midi, tempo_calculado)
    """
    if not mutation_result.success or mutation_result.excerpt is None:
        raise ValueError("La mutación no fue exitosa o no tiene excerpt válido")

    # Calcular el tempo apropiado para la mutación
    calculated_tempo = mutation_result.get_mutation_tempo(base_tempo)

    # Guardar archivos con el tempo calculado
    audio_path = save_excerpt_in_audio(excerpt=mutation_result.excerpt, dir_name=mutation_name, save_name=save_name,
                                       sample_rate=sample_rate)
    midi_path = save_excerpt_in_midi(excerpt=mutation_result.excerpt, dir_name=mutation_name, save_name=save_name,
                                     tempo=calculated_tempo)
    # print(f"Guardando mutación en audio: {audio_path}, MIDI: {midi_path} con tempo: {calculated_tempo}")

    return audio_path, midi_path, calculated_tempo


def extract_tempo_from_midi(midi_file_path: str) -> int:
    """
    Extrae el tempo del archivo MIDI.
    
    Parameters
    ----------
    midi_file_path : str
        Ruta al archivo MIDI
        
    Returns
    -------
    int
        Tempo en BPM, 120 por defecto si no se puede extraer
    """
    try:
        # Intentar con pretty_midi primero
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)

        # Obtener cambios de tempo
        tempo_changes = midi_data.get_tempo_changes()
        if len(tempo_changes) > 1 and len(tempo_changes[1]) > 0:
            # Usar el primer tempo encontrado
            initial_tempo = tempo_changes[1][0]  # tempo_changes[1] contiene los valores de tempo
            return int(initial_tempo)

        # Si no hay cambios de tempo explícitos, intentar calcular desde el BPM estimado
        if hasattr(midi_data, 'estimate_tempo'):
            estimated_tempo = midi_data.estimate_tempo()
            return int(estimated_tempo)

    except Exception:
        pass

    try:
        # Fallback con mido
        mid = mido.MidiFile(midi_file_path)

        # Buscar eventos de tempo en todas las pistas
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    # Convertir de microsegundos por beat a BPM
                    bpm = 60000000 / msg.tempo
                    return int(bpm)

        # Si no se encuentra tempo explícito, usar ticks_per_beat para estimación
        if mid.ticks_per_beat:
            # Estimar basándose en la densidad de notas (heurística simple)
            total_notes = 0
            total_time_ticks = 0

            for track in mid.tracks:
                track_time = 0
                for msg in track:
                    track_time += msg.time
                    if msg.type == 'note_on' and msg.velocity > 0:
                        total_notes += 1

                total_time_ticks = max(total_time_ticks, track_time)

            if total_notes > 0 and total_time_ticks > 0:
                # Heurística simple: estimar BPM basado en densidad de notas
                beats_estimated = total_time_ticks / mid.ticks_per_beat
                if beats_estimated > 0:
                    # Asumir duración razonable y calcular BPM
                    estimated_duration_seconds = beats_estimated * 0.5  # Asumir ~120 BPM como base
                    estimated_bpm = (beats_estimated * 60) / estimated_duration_seconds
                    return max(60, min(200, int(estimated_bpm)))

    except Exception:
        pass

    # Fallback: tempo estándar
    return 120
