"""Code to perform the degradations i.e. edits to the midi data"""
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
        # Find ranges which contain a note to align to
        # I couldn't think of a better solution than iterating here.
        # This code checks, for every range, whether at least 1 onset
        # lies within that range.
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