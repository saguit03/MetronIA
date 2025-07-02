import math

NOTE_NAMES = ['C', 'C#/Db', 'D', 'D#/Eb', 'E/Fb', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B/Cb']
INTERVALS = {
    0: 'Unison',
    1: '2m Minor Second',
    2: '2M Major Second',
    3: '3m Minor Third',
    4: '3M Major Third',
    5: '4P Perfect Fourth',
    6: '4Aug/5dim Tritone',
    7: '5P Perfect Fifth',
    8: '6m Minor Sixth',
    9: '6M Major Sixth',
    10: '7m Minor Seventh',
    11: '7M Major Seventh',
    12: 'Octave'
}


def hz_to_note(frequency_hz: float):
    if frequency_hz <= 0:
        return None

    midi_number = 69 + 12 * math.log2(frequency_hz / 440.0)
    midi_number = round(midi_number)

    if midi_number < 0 or midi_number > 127:
        return None

    octave = (midi_number // 12) - 1
    note_index = midi_number % 12

    return f"{NOTE_NAMES[note_index]}{octave}"


def note_to_index(note: str):
    if not note or len(note) < 2:
        return None
    note_part = note[:-1]  # Parte de la nota (sin el número de octava)
    try:
        note_index = NOTE_NAMES.index(note_part)
    except:
        return None
    return note_index


def calculate_note_similarity(note_ref, note_live):
    note_ref_index = note_to_index(note_ref)
    note_live_index = note_to_index(note_live)

    if note_ref_index is not None and note_live_index is not None:
        diff = abs(note_ref_index - note_live_index)
        if diff == 0:
            similarity = 1.0
        elif diff == 1 or diff == 11:  # Semitones apart (e.g., C to C# or C to B)
            similarity = 0.8
        elif diff == 2 or diff == 10:  # Whole tones apart (e.g., C to D or C to Bb)
            similarity = 0.6
        elif diff == 3 or diff == 9:  # Minor thirds apart (e.g., C to Eb or C to A)
            similarity = 0.4
        elif diff == 4 or diff == 8:  # Major thirds apart (e.g., C to E or C to G#)
            similarity = 0.2
        elif diff == 5 or diff == 7:  # Perfect fourths or fifths apart (e.g., C to F or C to G)
            similarity = 0.1
        else:
            similarity = 0.0
        return similarity, INTERVALS.get(diff, None)
    else:
        return 0.0, "Unknown"
