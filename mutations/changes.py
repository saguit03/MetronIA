from enum import Enum
from pathlib import Path
from typing import List, Any, NamedTuple

import numpy as np
import pandas as pd


class ChangeType(Enum):
    """
    Enum para los tipos de cambios que se pueden aplicar a un fragmento musical.
    """
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
    """
    Representa un único cambio atómico aplicado durante una mutación.
    - change_type: El tipo de cambio (del Enum ChangeType).
    """
    change_type: ChangeType
    onset_timestamp: float = None
    pitch: int = None


class TempoMutationDetail(NamedTuple):
    """
    Representa un único cambio atómico aplicado durante una mutación.
    - change_type: El tipo de cambio (del Enum ChangeType).
    """
    change_type: ChangeType
    factor: float = None


class ArticulationMutationDetail(NamedTuple):
    """
    Representa un cambio de articulación aplicado durante una mutación.
    - change_type: El tipo de cambio (del Enum ChangeType).
    """
    change_type: ChangeType
    articulation: str = None


def save_mutation_logs_to_csv(logs: List[Any], save_dir: str, save_name):
    """
    Guarda una lista de logs de mutación en un archivo CSV.

    Detecta automáticamente todos los atributos de los logs (pitch, onset_timestamp, factor, etc.)
    y convierte onset_timestamp a segundos si se proporciona sample_rate.

    Args:
        logs: Lista de objetos de log de mutación.
        save_dir: Carpeta donde guardar el CSV.
        save_name: Nombre base del archivo (sin extensión).
        sample_rate: Sample rate para convertir onset_timestamp (opcional).
    """
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
