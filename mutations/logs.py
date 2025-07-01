import numpy as np
import pandas as pd
from enum import Enum
from pathlib import Path
from typing import List, Any, NamedTuple, Optional


class ChangeType(Enum):
    NOTE_ADDED = "extra"
    NOTE_REMOVED = "missing"
    TIMING_SHIFT_EARLY = "early"
    TIMING_SHIFT_LATE = "late"
    PITCH_SHIFT = "pitch"
    TEMPO_CHANGE = "tempo"
    PROGRESSIVE_TEMPO_CHANGE = "progressive_tempo"
    ARTICULATION = "articulation"
    NO_CHANGE = "no_change"


class NoteMutationDetail(NamedTuple):
    change_type: ChangeType
    onset_timestamp: float = None
    pitch: int = None


class TempoMutationDetail(NamedTuple):
    change_type: ChangeType
    factor: float = None


class ArticulationMutationDetail(NamedTuple):
    change_type: ChangeType
    articulation: str = None


VERBOSE = False


def get_mutation_log(mutation, mutation_log, index: Optional[int] = None):
    logs = []
    if isinstance(mutation_log, TempoMutationDetail) or isinstance(mutation_log, ArticulationMutationDetail):
        logs.append(mutation_log)
        return logs

    for onset, pitch in zip(mutation["onset"], mutation["pitch"]):
        if VERBOSE: print(f"Onset: {onset}, Pitch: {pitch}")
        logs.append(NoteMutationDetail(change_type="no_change", onset_timestamp=onset, pitch=pitch))

    if index is not None:
        logs.pop(index)

    logs.append(mutation_log)
    if VERBOSE:
        for i, log in enumerate(logs):
            if isinstance(log, NoteMutationDetail):
                print(f"Log {i}: {log.change_type} at {log.onset_timestamp}, pitch: {log.pitch}")
            elif isinstance(log, TempoMutationDetail):
                print(f"Log {i}: {log.change_type} with factor {log.factor}")
    return logs


def save_mutation_logs_to_csv(logs: List[Any], save_dir: str, save_name):
    if not logs:
        print("⚠️ No hay logs de mutación para guardar.")
        return

    timestamps = False
    onset_times_seen = set()

    df = pd.DataFrame()
    for i, log in enumerate(logs):
        if isinstance(log, NoteMutationDetail):
            onset = np.round(log.onset_timestamp / 1000, 3)
            if onset in onset_times_seen:
                continue

            onset_times_seen.add(onset)
            aux = pd.DataFrame({
                'onset_type': [log.change_type],
                'onset_time': [onset],
                'pitch': [log.pitch],
            })
            df = pd.concat([df, aux], ignore_index=True)
            timestamps = True
        elif isinstance(log, TempoMutationDetail):
            aux = pd.DataFrame({
                'onset_type': [log.change_type],
                'factor': [log.factor],
            })
            df = pd.concat([df, aux], ignore_index=True)
        elif isinstance(log, ArticulationMutationDetail):
            aux = pd.DataFrame({
                'onset_type': [log.change_type],
                'articulation': [log.articulation],
            })
            df = pd.concat([df, aux], ignore_index=True)

    if timestamps:
        df.sort_values(by='onset_time', inplace=True)

    logs_dir = Path(save_dir) / "logs"
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(logs_dir) / f"{save_name}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
