"""
Analizador de tempo musical y estructura de compases.
"""

import numpy as np
import librosa
from typing import Dict, Any
from .config import AudioAnalysisConfig
from .results import TempoAnalysisResult


class TempoAnalyzer:
    """Analizador de tempo musical."""
    
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
    
    def analyze_tempo(self, audio_ref: np.ndarray, audio_live: np.ndarray, sr: int) -> TempoAnalysisResult:
        """Analiza las diferencias de tempo entre grabaciones."""
        tempo_ref, _ = librosa.beat.beat_track(y=audio_ref, sr=sr)
        tempo_live, _ = librosa.beat.beat_track(y=audio_live, sr=sr)
        
        tempo_ref = float(tempo_ref.item())
        tempo_live = float(tempo_live.item())
        difference = abs(tempo_ref - tempo_live)
        is_similar = difference <= self.config.tempo_threshold
        
        return TempoAnalysisResult(
            tempo_ref=tempo_ref,
            tempo_live=tempo_live,
            difference=difference,
            is_similar=is_similar
        )
    
    def validate_segments(self, audio_ref: np.ndarray, audio_live: np.ndarray, sr: int, 
                         tolerance: float = 0.2) -> Dict[str, Any]:
        """Valida la estructura de compases."""
        duration_ref = librosa.get_duration(y=audio_ref, sr=sr)
        duration_live = librosa.get_duration(y=audio_live, sr=sr)
        
        n_compases_ref = int(duration_ref // self.config.compas_duration)
        n_compases_live = int(duration_live // self.config.compas_duration)
        
        measures_compatible = abs(n_compases_ref - n_compases_live) <= 1
        duration_compatible = abs(duration_ref - duration_live) <= tolerance * duration_ref
        
        return {
            'measures_ref': n_compases_ref,
            'measures_live': n_compases_live,
            'duration_ref': duration_ref,
            'duration_live': duration_live,
            'measures_compatible': measures_compatible,
            'duration_compatible': duration_compatible,
            'overall_compatible': measures_compatible and duration_compatible
        }
