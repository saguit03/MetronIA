from enum import Enum
from typing import List, Any, NamedTuple
import pandas as pd
from pathlib import Path
import numpy as np

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
    
    df = pd.DataFrame()
    for i, log in enumerate(logs):
        if isinstance(log, NoteMutationDetail):
            onset = np.round(log.onset_timestamp/1000, 3)
            aux = pd.DataFrame({
                'onset_timestamp': [onset],
                'pitch': [log.pitch],
                'change_type': [log.change_type]
            })
            df = pd.concat([df, aux], ignore_index=True)
            # print(f"Log {i}: {log.change_type} at {log.onset_timestamp}, pitch: {log.pitch}")
        elif isinstance(log, TempoMutationDetail):
            df.insert(i, log.change_type, log.factor)
            # print(f"Log {i}: {log.change_type} with factor {log.factor}")

    logs_dir = Path(save_dir) / "logs"
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(logs_dir) / f"{save_name}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    # print(f"✅ Logs guardados en {csv_path}") 
