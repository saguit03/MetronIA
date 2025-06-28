import logging
import numpy as np
import pandas as pd
from numpy.random import choice

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


def calculate_musical_duration(tempo, note_type='eighth'):
    quarter_note_duration = (60.0 / tempo) * 1000

    note_values = {
        'whole': 4.0,  # redonda
        'half': 2.0,  # blanca
        'quarter': 1.0,  # negra
        'eighth': 0.5,  # corchea
        'sixteenth': 0.25  # semicorchea
    }

    if note_type not in note_values:
        note_type = 'eighth'  # default

    duration = quarter_note_duration * note_values[note_type]
    return int(duration)


@set_random_seed
def time_shift_mutation(
        excerpt,
        tempo=120,
        note_types=['eighth', 'sixteenth'],
):
    excerpt = pre_process(excerpt)

    if len(excerpt) == 0:
        logging.warning("Empty excerpt. Returning None.")
        return None

    valid_notes = list(excerpt.index)
    if not valid_notes:
        logging.warning("No valid notes to time shift. Returning None.")
        return None

    index = choice(valid_notes)

    note_type = choice(note_types)

    silence_duration = calculate_musical_duration(tempo, note_type)

    degraded = excerpt.copy()
    old_onset = excerpt.loc[index, "onset"]

    mask_later_notes = degraded["onset"] > old_onset
    degraded.loc[mask_later_notes, "onset"] += silence_duration

    degraded = post_process(degraded)
    return degraded, index


def tempo_change(excerpt, tempo_factor):
    if len(excerpt) == 0:
        return None

    degraded = pre_process(excerpt)
    scale_factor = 1.0 / tempo_factor
    degraded["onset"] = (degraded["onset"] * scale_factor).round().astype(int)
    degraded["dur"] = (degraded["dur"] * scale_factor).round().astype(int)
    degraded.loc[degraded["dur"] < 1, "dur"] = degraded.loc[degraded["dur"] < 1, "dur"].astype(int).clip(
        lower=1)  # Duración mínima

    return post_process(degraded)


def progressive_tempo_change(excerpt, end_factor):
    if len(excerpt) == 0:
        return None

    start_factor = 1.0
    degraded = pre_process(excerpt, sort=True)

    total_time = (degraded["onset"] + degraded["dur"]).max()

    for idx in degraded.index:
        progress = degraded.loc[idx, "onset"] / total_time if total_time > 0 else 0

        current_factor = start_factor + (end_factor - start_factor) * progress
        scale_factor = 1.0 / current_factor

        degraded.loc[idx, "onset"] = int(degraded.loc[idx, "onset"] * scale_factor)
        degraded.loc[idx, "dur"] = max(1, int(degraded.loc[idx, "dur"] * scale_factor))

    return post_process(degraded)


def articulated_staccato(excerpt, gap=30):
    if excerpt.empty or "onset" not in excerpt.columns or "dur" not in excerpt.columns:
        logging.warning("Excerpt vacío o mal formado. No se puede aplicar staccato.")
        return None

    degraded = pre_process(excerpt.copy(), sort=True)

    for i in range(len(degraded) - 1):
        this_offset = degraded.loc[i, "onset"] + degraded.loc[i, "dur"]
        next_onset = degraded.loc[i + 1, "onset"]  # Solo si hay superposición o continuidad
        if next_onset > degraded.loc[i, "onset"] and this_offset > next_onset - gap:
            new_dur = max(1, next_onset - degraded.loc[i, "onset"] - gap)
            degraded.loc[i, "dur"] = int(new_dur)

    degraded = post_process(degraded)
    return degraded


def articulated_accentuated(excerpt, boost=30):
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
    excerpt["onset"] = np.cumsum([excerpt.loc[0, "onset"]] + list((excerpt["dur"] * factors)[1:].astype(int))).astype(
        int)
    return excerpt


@set_random_seed
def offset_cut(
        excerpt,
        min_cut=MIN_SHIFT_DEFAULT,
        max_cut=MAX_SHIFT_DEFAULT,
        min_duration=MIN_DURATION_DEFAULT,
):
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

    degraded = post_process(degraded)
    return degraded, index


@set_random_seed
def remove_intermediate_note(excerpt):
    if excerpt.shape[0] == 0:
        logging.warning("No notes to remove. Returning None.")
        return None

    degraded = pre_process(excerpt)
    possible_notes = list(degraded.index)
    possible_notes = possible_notes[1:-1]
    if not possible_notes:
        logging.warning("No intermediate notes to remove. Returning None.")
        return None

    index = choice(possible_notes)

    degraded = degraded.drop(index)

    degraded = post_process(degraded, sort=False)
    return degraded, index


@set_random_seed
def note_played_too_soon_mutation(
        excerpt,
        tempo=120,
        note_types=['eighth', 'sixteenth'],
        tries=TRIES_DEFAULT,
):
    excerpt = pre_process(excerpt)

    if len(excerpt) == 0:
        logging.warning("Empty excerpt. Returning None.")
        return None

    valid_notes = list(excerpt.index[:-1]) if len(excerpt) > 1 else []
    if not valid_notes:
        logging.warning("No valid notes to shorten (need at least 2 notes). Returning None.")
        return None

    index = choice(valid_notes)

    target_note_type = choice(note_types)
    new_duration = calculate_musical_duration(tempo, target_note_type)

    degraded = excerpt.copy()
    original_duration = degraded.loc[index, "dur"]

    if new_duration >= original_duration:
        new_duration = max(MIN_DURATION_DEFAULT, original_duration // 2)

    duration_reduction = original_duration - new_duration

    degraded.loc[index, "dur"] = new_duration

    current_onset = degraded.loc[index, "onset"]
    mask_later_notes = degraded["onset"] > current_onset
    degraded.loc[mask_later_notes, "onset"] -= duration_reduction

    if (degraded["onset"] < 0).any():
        logging.warning("Generated negative onsets. Trying again.")
        if tries == 1:
            logging.warning(TRIES_WARN_MSG)
            return None
        return note_played_too_soon_mutation(
            excerpt,
            tempo=tempo,
            note_types=note_types,
            tries=tries - 1,
        )

    degraded = post_process(degraded)
    return degraded, index


@set_random_seed
def offset_shift(
        excerpt,
        min_shift=MIN_SHIFT_DEFAULT,
        max_shift=MAX_SHIFT_DEFAULT,
        min_duration=MIN_DURATION_DEFAULT,
        max_duration=MAX_DURATION_DEFAULT,
):
    excerpt = pre_process(excerpt)

    min_shift = max(min_shift, 1)
    max_duration += 1

    onset = excerpt["onset"]
    duration = excerpt["dur"]
    end_time = (onset + duration).max()

    shortest_lengthened_dur = (duration + min_shift).clip(lower=min_duration)
    longest_lengthened_dur = (
        (duration + (max_shift + 1))
        .clip(upper=(end_time + 1) - onset)
        .clip(upper=max_duration)
    )

    shortest_shortened_dur = (duration - max_shift).clip(lower=min_duration)
    longest_shortened_dur = (duration - (min_shift - 1)).clip(upper=max_duration)

    valid = (shortest_lengthened_dur < longest_lengthened_dur) | (
            shortest_shortened_dur < longest_shortened_dur
    )
    valid_notes = list(valid.index[valid])

    if not valid_notes:
        logging.warning("No valid notes to offset shift. Returning None.")
        return None

    index = choice(valid_notes)

    ssd = shortest_shortened_dur[index]
    lsd = max(longest_shortened_dur[index], ssd)
    sld = shortest_lengthened_dur[index]
    lld = max(longest_lengthened_dur[index], sld)

    duration = split_range_sample([(ssd, lsd), (sld, lld)])
    degraded = excerpt.copy()
    degraded.loc[index, "dur"] = duration
    degraded = post_process(degraded)
    return degraded, index
