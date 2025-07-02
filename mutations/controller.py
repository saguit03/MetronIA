from mdtk.degradations import (
    pitch_shift,
    add_note,
    join_notes,
    MAX_PITCH_DEFAULT,
    MIN_PITCH_DEFAULT
)
from mdtk.mutations import *
from mutations.globals import *
from .logs import NoteMutationDetail, TempoMutationDetail, ArticulationMutationDetail, get_mutation_log


# pitch_shift. Shift the pitch of a note
def pitch_shift_mutation(excerpt, min_pitch=MIN_PITCH_DEFAULT, max_pitch=MAX_PITCH_DEFAULT, distribution=None):
    mutation, index = pitch_shift(excerpt, min_pitch=min_pitch, max_pitch=max_pitch, distribution=distribution, seed=SEED)
    log = NoteMutationDetail(change_type="pitch", onset_timestamp=mutation.loc[index, "onset"],
                             pitch=mutation.loc[index, "pitch"])
    return mutation, get_mutation_log(mutation, log, index)


def tempo_mutation_log(excerpt, tempo_factor):
    mutation = tempo_change(excerpt, tempo_factor=tempo_factor)
    log = TempoMutationDetail(change_type="tempo", factor=tempo_factor)
    return mutation, get_mutation_log(mutation, log)


# faster_tempo. This tempo is faster than the original one
def faster_tempo_mutation(excerpt, tempo_factor=FASTER):
    return tempo_mutation_log(excerpt, tempo_factor=tempo_factor)


# a_lot_faster_tempo. This tempo is a lot faster than the original one
def a_lot_faster_tempo_mutation(excerpt, tempo_factor=A_LOT_FASTER):
    return tempo_mutation_log(excerpt, tempo_factor=tempo_factor)


# slower_tempo. This tempo is slower than the original one
def slower_tempo_mutation(excerpt, tempo_factor=SLOWER):
    return tempo_mutation_log(excerpt, tempo_factor=tempo_factor)


# a_lot_slower_tempo. This tempo is a lot slower than the original one
def a_lot_slower_tempo_mutation(excerpt, tempo_factor=A_LOT_SLOWER):
    return tempo_mutation_log(excerpt, tempo_factor=tempo_factor)


def progressive_tempo_change_log(excerpt, factor):
    mutation = progressive_tempo_change(excerpt, end_factor=factor)
    log = TempoMutationDetail(change_type="progressive_tempo", factor=factor)
    return mutation, get_mutation_log(mutation, log)


# accelerando_tempo. Tempo is increasing over time
def accelerando_tempo_mutation(excerpt, factor=ACCELERANDO):
    return progressive_tempo_change_log(excerpt, factor=factor)


# ritardando_tempo. Tempo is decreasing over time
def ritardando_tempo_mutation(excerpt, factor=RITARDANDO):
    return progressive_tempo_change_log(excerpt, factor=factor)


# note_played_too_soon. Played BEFORE it should
def note_played_too_soon_mutation_controller(excerpt):
    mutation, index = note_played_too_soon_mutation(excerpt)
    log = NoteMutationDetail(change_type="early", onset_timestamp=mutation.loc[index, "onset"],
                             pitch=mutation.loc[index, "pitch"])
    return mutation, get_mutation_log(mutation, log, index)


# note_played_too_late. Played AFTER it should
def note_played_too_late_mutation_controller(excerpt):
    mutation, index = note_played_too_late_mutation(excerpt)
    log = NoteMutationDetail(change_type="late", onset_timestamp=mutation.loc[index, "onset"], pitch=mutation.loc[index, "pitch"])
    return mutation, get_mutation_log(mutation, log, index)


# note_held_too_long. A note is played longer than expected
def note_held_too_long_mutation(excerpt):
    mutation, index = offset_hold(excerpt)
    log = NoteMutationDetail(change_type="articulation", onset_timestamp=mutation.loc[index, "onset"],  pitch=mutation.loc[index, "pitch"])
    return mutation, get_mutation_log(mutation, log, index)


# note_cut_too_soon. A note is cut before it is expected
def note_cut_too_soon_mutation(excerpt):
    mutation, index = offset_cut(excerpt)
    log = NoteMutationDetail(change_type="articulation", onset_timestamp=mutation.loc[index, "onset"],  pitch=mutation.loc[index, "pitch"])
    return mutation, get_mutation_log(mutation, log, index)


# note_missing. Not played at all
def note_missing_mutation(excerpt):
    mutation, index = remove_intermediate_note(excerpt)
    log = NoteMutationDetail(change_type="missing", onset_timestamp=mutation.loc[index, "onset"], pitch=mutation.loc[index, "pitch"])
    return mutation, get_mutation_log(mutation, log, index)


# note_not_expected. There are more notes than expected
def note_not_expected_mutation(excerpt, min_pitch=MIN_PITCH_DEFAULT, max_pitch=MAX_PITCH_DEFAULT, min_dur=MIN_DUR,
                               max_dur=MAX_DUR, min_velocity=MIN_VELOCITY, max_velocity=MAX_VELOCITY):
    mutation, note = add_note(
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
    log = NoteMutationDetail(change_type="extra", onset_timestamp=note["onset"], pitch=note["pitch"])
    return mutation, get_mutation_log(mutation, log)


# articulated_legato. All the notes are smoothly connected
def articulated_legato_mutation(excerpt, max_gap=ARTICULATED_LEGATO["max_gap"], max_notes=ARTICULATED_LEGATO["max_notes"]):
    mutation = join_notes(excerpt, max_gap=max_gap, max_notes=max_notes, only_first=True)
    log = ArticulationMutationDetail(change_type="articulation", articulation="legato")
    return mutation, get_mutation_log(mutation, log)


# articulated_staccato. All the notes are silenced before playing the next one
def articulated_staccato_mutation(excerpt, gap=ARTICULATED_STACCATO):
    mutation = articulated_staccato(excerpt, gap)
    log = ArticulationMutationDetail(change_type="articulation", articulation="staccato")
    return mutation, get_mutation_log(mutation, log)


# articulated_accentuated. A note is played more intensely than others
def articulated_accentuated_mutation(excerpt, boost=ARTICULATED_ACCENTUATED):
    mutation = articulated_accentuated(excerpt, boost=boost)
    log = ArticulationMutationDetail(change_type="articulation", articulation="accentuated")
    return mutation, get_mutation_log(mutation, log)


# tempo_fluctuation. Tempo changes for no reason without a pattern
def tempo_fluctuation_mutation(excerpt, fluctuation=TEMPO_FLUCTUATION):
    mutation = tempo_fluctuation(excerpt, fluctuation=fluctuation)
    log = TempoMutationDetail(change_type="tempo", factor=TEMPO_FLUCTUATION)
    return mutation, get_mutation_log(mutation, log)
