"""
Estructuras de datos para resultados de análisis de audio musical.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class OnsetAnalysisResult:
    """Resultado del análisis de onsets."""
    onsets_ref: np.ndarray
    onsets_live: np.ndarray
    matched_correct: List[Tuple[float, float]]
    matched_early: List[Tuple[float, float]]
    matched_late: List[Tuple[float, float]]
    unmatched_ref: List[float]
    unmatched_live: List[float]
    
    @property
    def stats(self) -> Dict[str, int]:
        """Estadísticas del análisis de onsets."""
        return {
            'correct': len(self.matched_correct),
            'early': len(self.matched_early),
            'late': len(self.matched_late),
            'missing': len(self.unmatched_ref),
            'extra': len(self.unmatched_live),
            'total_ref': len(self.onsets_ref),
            'total_live': len(self.onsets_live)
        }


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
