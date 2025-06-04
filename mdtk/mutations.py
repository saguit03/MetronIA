"""More functions to extend the degradations i.e. edits to the midi data"""
import logging
import sys
from functools import wraps

import numpy as np
import pandas as pd
from numpy.random import choice, randint

from mdtk.df_utils import NOTE_DF_SORT_ORDER

MIN_PITCH_DEFAULT = 21
MAX_PITCH_DEFAULT = 108

MIN_SHIFT_DEFAULT = 100
MAX_SHIFT_DEFAULT = np.inf

MIN_DURATION_DEFAULT = 50
MAX_DURATION_DEFAULT = np.inf

MAX_GAP_DEFAULT = 50

MIN_VELOCITY_DEFAULT = 100
MAX_VELOCITY_DEFAULT = 100

TRIES_DEFAULT = 10

TRIES_WARN_MSG = (
    "Generated invalid (overlapping) degraded excerpt "
    "too many times. Try raising tries parameter (default 10). "
    "Returning None."
)

from mdtk.degradations import *

@set_random_seed
def time_shift_mutation(
    excerpt,
    min_shift=MIN_SHIFT_DEFAULT,
    max_shift=MAX_SHIFT_DEFAULT,
    align_onset=False,
    tries=TRIES_DEFAULT,
):
    """
    Shift the onset and offset times of one note and the following ones from the given excerpt,
    leaving its duration unchanged.

    Parameters
    ----------
    excerpt : pd.DataFrame
        An excerpt from a piece of music.

    min_shift : int
        The minimum amount by which the note will be shifted.

    max_shift : int
        The maximum amount by which the note will be shifted.

    align_onset : boolean
        Align the shifted note to the onset time of an existing note
        (within the given shift range).

    seed : int
        A seed to be supplied to np.random.seed(). None leaves numpy's
        random state unchanged.

    tries : int
        The number of times to try the degradation before giving up, in the case
        that the degraded excerpt overlaps.


    Returns
    -------
    degraded : pd.DataFrame
        A degradation of the excerpt, with the timing of one note and the ones after it changed,
        or None if there are no notes that can be changed.
    """
    excerpt = pre_process(excerpt)

    min_shift = max(min_shift, 1)

    onset = excerpt["onset"]
    offset = onset + excerpt["dur"]
    end_time = offset.max()

    # Shift earlier
    earliest_earlier_onset = (onset - (max_shift - 1)).clip(lower=0)
    latest_earlier_onset = onset - (min_shift - 1)

    # Shift later
    latest_later_onset = onset + (((end_time + 1) - offset).clip(upper=max_shift + 1))
    earliest_later_onset = onset + min_shift

    if align_onset:
        onset = pd.Series(onset.unique())
        for i, (eeo, leo, elo, llo) in enumerate(
            zip(
                earliest_earlier_onset,
                latest_earlier_onset,
                earliest_later_onset,
                latest_later_onset,
            )
        ):
            # Go through each range to check there is a valid onset
            earlier_valid = onset.between(eeo, leo - 1).any()
            later_valid = onset.between(elo, llo - 1).any()

            # Close invalid ranges
            if not earlier_valid:
                earliest_earlier_onset.iloc[i] = leo
            if not later_valid:
                earliest_later_onset.iloc[i] = llo

    # Find valid notes
    valid = (earliest_earlier_onset < latest_earlier_onset) | (
        earliest_later_onset < latest_later_onset
    )
    valid_notes = list(valid.index[valid])

    if not valid_notes:
        logging.warning("No valid notes to time shift. Returning None.")
        return None

    # Sample a random note
    index = choice(valid_notes)

    eeo = earliest_earlier_onset[index]
    leo = max(latest_earlier_onset[index], eeo)
    elo = earliest_later_onset[index]
    llo = max(latest_later_onset[index], elo)

    if align_onset:
        valid_onsets = onset.between(eeo, leo - 1) | onset.between(elo, llo - 1)
        valid_onsets = list(onset[valid_onsets])
        onset = choice(valid_onsets)
    else:
        onset = split_range_sample([(eeo, leo), (elo, llo)])

    degraded = excerpt.copy()

    old_onset = excerpt.loc[index, "onset"]
    delta_time = onset - old_onset

    # Aplica el desplazamiento a la nota elegida
    degraded.loc[index, "onset"] = onset

    # Aplica el mismo desplazamiento a las notas posteriores
    mask_later_notes = degraded.index != index
    mask_later_notes &= degraded["onset"] > old_onset
    degraded.loc[mask_later_notes, "onset"] += delta_time

    # Chequeo de solapamientos
    if overlaps(degraded, index):
        if tries == 1:
            logging.warning(TRIES_WARN_MSG)
            return None
        return time_shift(
            excerpt,
            min_shift=min_shift,
            max_shift=max_shift,
            align_onset=align_onset,
            tries=tries - 1,
        )

    degraded = post_process(degraded)
    return degraded


def faster_tempo(excerpt, tempo_factor=1.2):
    """
    Accelerates the tempo by multiplying all times by a factor < 1
    
    Parameters
    ----------
    excerpt : pd.DataFrame
        An excerpt from a piece of music.
    tempo_factor : float
        Acceleration factor (>1 for faster) and (<1 for slower)

    Returns
    ----------  
    degraded : df.DataFrame
        A degradation of the excerpt, with the factor applied to all notes
    """
    if len(excerpt) == 0:
        return None
    
    degraded = pre_process(excerpt)
    scale_factor = 1.0 / tempo_factor
    degraded["onset"] = (degraded["onset"] * scale_factor).round().astype(int)
    degraded["dur"] = (degraded["dur"] * scale_factor).round().astype(int)
    degraded.loc[degraded["dur"] < 1, "dur"] = degraded.loc[degraded["dur"] < 1, "dur"].astype(int).clip(lower=1)  # Duración mínima
    
    return post_process(degraded)

def slower_tempo(excerpt, tempo_factor=0.8):
    """
    Slows down the tempo by multiplying all times by a factor > 1
    """
    return faster_tempo(excerpt, tempo_factor=tempo_factor)

def accelerando_tempo(excerpt, end_factor):
    """
    Acceleration of the tempo over time.
    
    Parameters
    ----------
    excerpt : pd.DataFrame
        An excerpt from a piece of music.
    end_factor : float
        Acceleration factor (>1 for faster) and (<1 for slower)
    tries : int
        Número de intentos para aplicar la mutación
    
    Returns
    ----------  
    degraded : df.DataFrame
        A degradation of the excerpt, with the factor applied to all notes, or None if
        the degradations cannot be performed.
    """
    if len(excerpt) == 0:
        return None
    
    start_factor=1.0
    degraded = pre_process(excerpt, sort=True)
    
    # Calcular tiempo total
    total_time = (degraded["onset"] + degraded["dur"]).max()
    
    for idx in degraded.index:
        # Progreso temporal (0 a 1)
        progress = degraded.loc[idx, "onset"] / total_time if total_time > 0 else 0
        
        # Interpolación del factor de tempo
        current_factor = start_factor + (end_factor - start_factor) * progress
        scale_factor = 1.0 / current_factor
        
        degraded.loc[idx, "onset"] = int(degraded.loc[idx, "onset"] * scale_factor)
        degraded.loc[idx, "dur"] = max(1, int(degraded.loc[idx, "dur"] * scale_factor))
    
    return post_process(degraded)

def ritardando_tempo(excerpt, end_factor=0.7):
    """
    Ritardando tempo mutation, where the tempo decreases over time.
    """
    return accelerando_tempo(excerpt, end_factor=end_factor)

def articulated_staccato(excerpt, gap=30):
    """
    Simulates staccato articulation by shortening the duration of notes
    to create a gap between them.

    Parameters
    ----------
    excerpt : pd.DataFrame
        An excerpt from a piece of music.
        
    gap : int
        The gap between notes in milliseconds.

    Returns
    -------
    degraded : pd.DataFrame
        A staccato version of the excerpt, with shortened note durations.

    """
    if excerpt.empty or "onset" not in excerpt.columns or "dur" not in excerpt.columns:
        logging.warning("Excerpt vacío o mal formado. No se puede aplicar staccato.")
        return None

    degraded = pre_process(excerpt.copy(), sort=True)

    for i in range(len(degraded) - 1):
        this_offset = degraded.loc[i, "onset"] + degraded.loc[i, "dur"]
        next_onset = degraded.loc[i + 1, "onset"]        # Solo si hay superposición o continuidad
        if next_onset > degraded.loc[i, "onset"] and this_offset > next_onset - gap:
            new_dur = max(1, next_onset - degraded.loc[i, "onset"] - gap)
            degraded.loc[i, "dur"] = int(new_dur)

    degraded = post_process(degraded)
    return degraded

def articulated_accentuated(excerpt, boost=30):
    """
    Simulates accentuated articulation by increasing the velocity
    of all notes in the excerpt.

    Parameters
    ----------
    excerpt : pd.DataFrame
        An excerpt from a piece of music.

    boost : int
        Increase in velocity (max. 127).

    Returns
    -------
    degraded : pd.DataFrame
        An accentuated version of the excerpt, with increased note velocities.
    """
    if excerpt.empty or "velocity" not in excerpt.columns:
        logging.warning("Excerpt vacío o sin columna 'velocity'.")
        return None

    degraded = pre_process(excerpt.copy())
    degraded["velocity"] = (degraded["velocity"] + boost).clip(upper=127)

    degraded = post_process(degraded)
    return degraded

def tempo_fluctuation(excerpt, fluctuation=0.2):
    excerpt = excerpt.copy()
    factors = np.random.uniform(1 - fluctuation, 1 + fluctuation, len(excerpt))
    excerpt["onset"] = np.cumsum([excerpt.loc[0, "onset"]] + list((excerpt["dur"] * factors)[1:].astype(int))).astype(int)
    return excerpt
@set_random_seed
def offset_cut(
    excerpt,
    min_cut=MIN_SHIFT_DEFAULT,
    max_cut=MAX_SHIFT_DEFAULT,
    min_duration=MIN_DURATION_DEFAULT,
    tries=TRIES_DEFAULT,
):
    """
    Cut (shorten) the offset time of one note in the given excerpt.

    This function reduces the duration of a single randomly selected note
    by an amount between `min_cut` and `max_cut`, ensuring the final duration
    remains above `min_duration`. It does not extend any note.

    Parameters
    ----------
    excerpt : pd.DataFrame
        A MIDI excerpt as a DataFrame with 'onset', 'dur', and other columns.

    min_cut : int
        The minimum amount by which the note duration will be reduced.

    max_cut : int
        The maximum amount by which the note duration will be reduced.

    min_duration : int
        The minimum allowed duration for any resulting note.

    tries : int
        Number of retries in case the shortened note overlaps improperly.

    Returns
    -------
    degraded : pd.DataFrame or None
        A new DataFrame with one note shortened in time, or None if no valid
        modification could be performed.
    """
    excerpt = pre_process(excerpt)

    min_cut = max(min_cut, 1)

    duration = excerpt["dur"]
    shortest_new_dur = (duration - max_cut).clip(lower=min_duration)
    longest_new_dur = (duration - (min_cut - 1)).clip(upper=duration)

    valid = shortest_new_dur < longest_new_dur
    valid_notes = list(valid.index[valid])

    if not valid_notes:
        logging.warning("No valid notes to cut. Returning None.")
        return None

    index = choice(valid_notes)

    ssd = shortest_new_dur[index]
    lsd = max(longest_new_dur[index], ssd)

    new_dur = split_range_sample([(ssd, lsd)])

    degraded = excerpt.copy()
    degraded.loc[index, "dur"] = new_dur

    if overlaps(degraded, index):
        if tries == 1:
            logging.warning(TRIES_WARN_MSG)
            return None
        return offset_cut(
            excerpt,
            min_cut=min_cut,
            max_cut=max_cut,
            min_duration=min_duration,
            tries=tries - 1,
        )

    degraded = post_process(degraded)
    return degraded

@set_random_seed
def remove_intermediate_note(excerpt, tries=TRIES_DEFAULT):
    """
    Remove one note from the given excerpt, except the first and last notes.
    This function randomly selects a note from the excerpt (excluding the first and last) and removes it

    Parameters
    ----------
    excerpt : df.DataFrame
        An excerpt from a piece of music.

    seed : int
        A seed to be supplied to np.random.seed(). None leaves numpy's
        random state unchanged.

    tries : int
        The number of times to try the degradation before giving up, in the case
        that the degraded excerpt overlaps. This is not used, but we keep it for
        consistency.

    Returns
    -------
    degraded : df.DataFrame
        A degradation of the excerpt, with one note removed, or None if
        the degradations cannot be performed.
    """
    if excerpt.shape[0] == 0:
        logging.warning("No notes to remove. Returning None.")
        return None

    degraded = pre_process(excerpt)
    possible_notes = list(degraded.index)
    # Exclude the first and last notes
    possible_notes = possible_notes[1:-1]  # Exclude first and last notes
    if not possible_notes:
        logging.warning("No intermediate notes to remove. Returning None.")
        return None
    
    # Sample a random note
    note_index = choice(possible_notes)

    # Remove that note
    degraded = degraded.drop(note_index)

    # No need to check for overlap
    degraded = post_process(degraded, sort=False)
    return degraded