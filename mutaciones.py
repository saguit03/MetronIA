import numpy as np

from mdtk.degradations import (
    pitch_shift,
    offset_shift,
    onset_shift,
    remove_note,
    add_note,
    join_notes,
    MAX_PITCH_DEFAULT,
    MIN_PITCH_DEFAULT
)
from mdtk.mutations import (
    faster_tempo,
    articulated_staccato,
    articulated_accentuated,
    accelerando_tempo,
    slower_tempo,
    time_shift_mutation,
    tempo_fluctuation,
    offset_cut
)

from default_factors import (
    FASTER,
    A_LOT_FASTER,
    SLOWER,
    A_LOT_SLOWER,
    ACCELERANDO,
    RITARDANDO,
    NOTE_PLAYED_TOO_SOON,
    NOTE_PLAYED_TOO_LATE,
    MIN_DUR,
    MAX_DUR,
    MIN_VELOCITY,
    MAX_VELOCITY,
    NOTE_HELD_TOO_LONG,
    NOTE_CUT_TOO_SOON,
    ARTICULATED_LEGATO,
    ARTICULATED_STACCATO,
    ARTICULATED_ACCENTUATED,
    TEMPO_FLUCTUATION,
    SEED
)

# pitch_shift. Shift the pitch of a note
def pitch_shift_mutation(excerpt, min_pitch=MIN_PITCH_DEFAULT, max_pitch=MAX_PITCH_DEFAULT, distribution=None):
    return pitch_shift(excerpt, min_pitch=min_pitch, max_pitch=max_pitch, distribution=distribution, seed=SEED)
    
# faster_tempo. This tempo is faster than the original one
def faster_tempo_mutation(excerpt, tempo_factor=FASTER):
    return faster_tempo(excerpt, tempo_factor=tempo_factor)

# a_lot_faster_tempo. This tempo is a lot faster than the original one
def a_lot_faster_tempo_mutation(excerpt, tempo_factor=A_LOT_FASTER):
    return faster_tempo(excerpt, tempo_factor=tempo_factor)
    
# slower_tempo. This tempo is slower than the original one
def slower_tempo_mutation(excerpt, tempo_factor=SLOWER):
    return slower_tempo(excerpt, tempo_factor=tempo_factor)
    
# a_lot_slower_tempo. This tempo is a lot slower than the original one
def a_lot_slower_tempo_mutation(excerpt, tempo_factor=A_LOT_SLOWER):
    return slower_tempo(excerpt, tempo_factor=tempo_factor)
    
# accelerando_tempo. Tempo is increasing over time
def accelerando_tempo_mutation(excerpt, factor=ACCELERANDO):
    return accelerando_tempo(excerpt, end_factor=factor)
    
# ritardando_tempo. Tempo is decreasing over time
def ritardando_tempo_mutation(excerpt, factor=RITARDANDO):
    return accelerando_tempo(excerpt, end_factor=factor) 
    
# note_played_too_soon. Played BEFORE it should
def note_played_too_soon_time_mutation(excerpt, min_shift=NOTE_PLAYED_TOO_SOON["min_shift"], max_shift=NOTE_PLAYED_TOO_SOON["max_shift"], align_onset=False):
    return time_shift_mutation(excerpt, min_shift=min_shift, max_shift=max_shift, align_onset=align_onset, seed=SEED)
    
# note_played_too_late. Played AFTER it should
def note_played_too_late_time_mutation(excerpt, min_shift=NOTE_PLAYED_TOO_LATE["min_shift"], max_shift=NOTE_PLAYED_TOO_LATE["max_shift"], align_onset=False):
    return time_shift_mutation(excerpt, min_shift=min_shift, max_shift=max_shift, align_onset=align_onset, seed=SEED)

# note_played_too_soon. Played BEFORE it should
def note_played_too_soon_onset_mutation(excerpt, min_shift=NOTE_PLAYED_TOO_SOON["min_shift"], max_shift=NOTE_PLAYED_TOO_SOON["max_shift"], align_onset=False):
    return onset_shift(excerpt, min_shift=min_shift, max_shift=max_shift, align_onset=align_onset, seed=SEED)
    
# note_played_too_late. Played AFTER it should
def note_played_too_late_onset_mutation(excerpt, min_shift=NOTE_PLAYED_TOO_LATE["min_shift"], max_shift=NOTE_PLAYED_TOO_LATE["max_shift"], align_onset=False):
    return time_shift_mutation(excerpt, min_shift=min_shift, max_shift=max_shift, align_onset=align_onset, seed=SEED)
    
# note_held_too_long. A note is played longer than expected
def note_held_too_long_mutation(excerpt, min_shift=NOTE_HELD_TOO_LONG["min_shift"], max_shift=NOTE_HELD_TOO_LONG["max_shift"], align_dur=True):
    return offset_shift(excerpt, min_shift=min_shift, max_shift=max_shift, align_dur=align_dur, seed=SEED)

# note_cut_too_soon. A note is cut before it is expected
def note_cut_too_soon_mutation(excerpt, min_shift=NOTE_CUT_TOO_SOON["min_shift"], max_shift=NOTE_CUT_TOO_SOON["max_shift"]):
    return offset_cut(excerpt, min_cut=min_shift, max_cut=max_shift, seed=SEED)
    
# note_missing. Not played at all
def note_missing_mutation(excerpt):
    return remove_note(excerpt, seed=SEED)
    
# note_not_expected. There are more notes than expected
def note_not_expected_mutation(excerpt, min_pitch=MIN_PITCH_DEFAULT, max_pitch=MAX_PITCH_DEFAULT, min_dur=MIN_DUR, max_dur=MAX_DUR, min_velocity=MIN_VELOCITY, max_velocity=MAX_VELOCITY):
    return add_note(
        excerpt,
        min_pitch=min_pitch,
        max_pitch=max_pitch,
        min_duration=min_dur,
        max_duration=max_dur,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
        align_pitch=False,
        align_time=False,
        align_velocity=False,
        seed=SEED
    )

# articulated_legato. All the notes are smoothly connected
def articulated_legato_mutation(excerpt, max_gap=ARTICULATED_LEGATO["max_gap"], max_notes=ARTICULATED_LEGATO["max_notes"]):
    return join_notes(excerpt, max_gap=max_gap, max_notes=max_notes, only_first=True, seed=SEED)

# articulated_staccato. All the notes are silenced before playing the next one
def articulated_staccato_mutation(excerpt, gap=ARTICULATED_STACCATO):
    return articulated_staccato(excerpt, gap)

# articulated_accentuated. A note is played more intensely than others
def articulated_accentuated_mutation(excerpt, boost=ARTICULATED_ACCENTUATED):
    return articulated_accentuated(excerpt, boost=boost)

# tempo_fluctuation. Tempo changes for no reason without a pattern
def tempo_fluctuation_mutation(excerpt, fluctuation=TEMPO_FLUCTUATION):
    return tempo_fluctuation(excerpt, fluctuation=fluctuation)