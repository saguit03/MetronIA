import mido
import numpy as np
import pandas as pd
import pretty_midi
import warnings
from pathlib import Path

from mdtk.utils import synthesize_from_note_df
from mutations.globals import MUTATIONS_PATH, MUTATIONS_AUDIO_PATH, MUTATIONS_MIDI_PATH
from mutations.results import MutationResult

warnings.filterwarnings("ignore", module="pretty_midi")


def save_excerpt_in_audio(excerpt, dir_name, save_name, sample_rate=16000):
    from utils.audio_utils import save_audio
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
                temp_time = track_time
                duration = 0
                for future_msg in track[track.index(msg) + 1:]:
                    temp_time += future_msg.time
                    if (future_msg.type == 'note_off' and future_msg.note == msg.note) or \
                            (future_msg.type == 'note_on' and future_msg.note == msg.note and future_msg.velocity == 0):
                        duration = temp_time - track_time
                        break

                ticks_per_beat = mid.ticks_per_beat
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
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    notes = []

    for track_idx, instrument in enumerate(midi_data.instruments):
        for note in instrument.notes:
            notes.append({
                'onset': note.start * 1000,
                'pitch': note.pitch,
                'dur': (note.end - note.start) * 1000,
                'velocity': note.velocity,
                'track': track_idx
            })

    df = pd.DataFrame(notes)
    df = df.sort_values('onset').reset_index(drop=True)
    return df


def save_excerpt_in_midi(excerpt, dir_name, save_name, tempo=120):
    save_dir = MUTATIONS_PATH / Path(dir_name) / MUTATIONS_MIDI_PATH
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_filename = Path(save_dir) / f"{save_name}.mid"

    midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    tracks = excerpt.groupby('track') if 'track' in excerpt.columns else [('default', excerpt)]

    for track_id, track_notes in tracks:
        instrument = pretty_midi.Instrument(program=0, name=f"Track_{track_id}")

        for _, note_row in track_notes.iterrows():
            start_time = note_row['onset'] / 1000.0
            duration = note_row['dur'] / 1000.0
            end_time = start_time + duration

            note = pretty_midi.Note(
                velocity=int(note_row['velocity']),
                pitch=int(note_row['pitch']),
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)

        midi_data.instruments.append(instrument)

    midi_data.write(str(output_filename))

    return output_filename


def save_mutation_complete(mutation_result: MutationResult, mutation_name, save_name, base_tempo=120,
                           soundfont_path=None, sample_rate=16000):
    if not mutation_result.success or mutation_result.excerpt is None:
        raise ValueError("La mutación no fue exitosa o no tiene excerpt válido")

    calculated_tempo = mutation_result.get_mutation_tempo(base_tempo)
    audio_path = save_excerpt_in_audio(excerpt=mutation_result.excerpt, dir_name=mutation_name, save_name=save_name,
                                       sample_rate=sample_rate)
    midi_path = save_excerpt_in_midi(excerpt=mutation_result.excerpt, dir_name=mutation_name, save_name=save_name,
                                     tempo=calculated_tempo)
    return audio_path, midi_path, calculated_tempo


def extract_tempo_from_midi(midi_file_path: str) -> int:
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        tempo_changes = midi_data.get_tempo_changes()
        if len(tempo_changes) > 1 and len(tempo_changes[1]) > 0:
            initial_tempo = tempo_changes[1][0]  # tempo_changes[1] contiene los valores de tempo
            return int(initial_tempo)

        if hasattr(midi_data, 'estimate_tempo'):
            estimated_tempo = midi_data.estimate_tempo()
            return int(estimated_tempo)

    except Exception:
        pass

    try:
        mid = mido.MidiFile(midi_file_path)

        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    # Convertir de microsegundos por beat a BPM
                    bpm = 60000000 / msg.tempo
                    return int(bpm)

        if mid.ticks_per_beat:
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
                beats_estimated = total_time_ticks / mid.ticks_per_beat
                if beats_estimated > 0:
                    estimated_duration_seconds = beats_estimated * 0.5
                    estimated_bpm = (beats_estimated * 60) / estimated_duration_seconds
                    return max(60, min(200, int(estimated_bpm)))

    except Exception:
        pass

    return 120  # Por defecto
