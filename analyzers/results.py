"""
Estructuras de datos para resultados de análisis de audio musical.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class BeatSpectrumResult:
    """Resultado del análisis de beat spectrum."""
    beat_ref: np.ndarray
    beat_aligned: np.ndarray
    similarity_diff: np.ndarray
    is_similar: bool
    max_difference: float


@dataclass
class TempoAnalysisResult:
    """Resultado del análisis de tempo."""
    tempo_ref: float
    tempo_live: float
    difference: float
    is_similar: bool
    tempo_proportion: float = 1.0
    original_ref_tempo: float = None
    original_live_tempo: float = None
    resampling_applied: bool = False
