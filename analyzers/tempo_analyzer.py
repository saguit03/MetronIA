"""
Analizador de tempo musical y estructura de compases.
"""

from typing import Dict, Any, List, Optional

import librosa
import numpy as np
import traceback
from dataclasses  import dataclass

from .config import AudioAnalysisConfig

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


class TempoAnalyzer:
    """Analizador de tempo musical."""
    
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
        
    def analyze_tempo_with_reference(self, audio_ref: np.ndarray, audio_live: np.ndarray, 
                                   sr: int, reference_tempo: Optional[float] = None) -> TempoAnalysisResult:
        """
        Análisis de tempo usando un tempo de referencia conocido para mayor precisión.
        
        Args:
            audio_ref: Audio de referencia
            audio_live: Audio en vivo  
            sr: Sample rate
            reference_tempo: Tempo conocido del MIDI original (opcional)
            
        Returns:
            Resultado del análisis de tempo
        """
        if reference_tempo:
            tempo_ref = reference_tempo
        else:
            candidates_ref = self.extract_multiple_tempo_candidates(audio_ref, sr)
            tempo_ref = candidates_ref[0]
            
        candidates_live = self.extract_multiple_tempo_candidates(audio_live, sr, tempo_ref)
        best_tempo_live = self.get_best_candidate_tempo(candidates_live, tempo_ref)
        
        difference = abs(tempo_ref - best_tempo_live)
        is_similar = difference <= self.config.tempo_threshold
        
        return TempoAnalysisResult(
            tempo_ref=int(tempo_ref),
            tempo_live=int(best_tempo_live),
            difference=difference,
            is_similar=is_similar
        )
    
    def get_best_candidate_tempo(self, candidates_live, tempo_ref):
        best_diff = None
        for candidate in candidates_live:
                tempo_live = self.correct_tempo_octave_errors(tempo_ref, candidate)
                diff = abs(tempo_ref - tempo_live)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_tempo_live = tempo_live
        return best_tempo_live
        
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
    
    def extract_multiple_tempo_candidates(self, audio: np.ndarray, sr: int, start_tempo: Optional[int] = None) -> List[float]:
        """
        Extrae múltiples candidatos de tempo usando diferentes métodos para mayor robustez.
        
        Args:
            audio: Audio a analizar
            sr: Sample rate
            
        Returns:
            Lista de candidatos de tempo ordenados por confianza
        """
        tempos = []
        
        try:
            if start_tempo:
                static_tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, start_bpm=start_tempo)
            else:
                static_tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            tempos.append(float(static_tempo.item()))
        except:
            traceback.print_exc()
        
        try:
            if start_tempo:
                dynamic_tempo = librosa.feature.tempo(y=audio, sr=sr, aggregate=np.median, start_bpm=start_tempo)[0]
            else:
                dynamic_tempo = librosa.feature.tempo(y=audio, sr=sr, aggregate=np.median)[0]
            tempos.append(float(dynamic_tempo))
        except:
            traceback.print_exc()
        
        # Eliminar duplicados y valores extremos
        tempos = [t for t in tempos if 40 <= t <= 300]
        tempos = list(set([round(t, 0) for t in tempos]))
        
        return sorted(tempos)
    
    def correct_tempo_octave_errors(self, tempo_ref: float, tempo_live: float) -> float:
        """
        Corrige errores de octava en la detección de tempo (doble/mitad).
        
        Args:
            tempo_ref: Tempo de referencia
            tempo_live: Tempo detectado en vivo (posiblemente erróneo)
            
        Returns:
            Tempo corregido
        """
        ratio_double = tempo_live / tempo_ref if tempo_ref > 0 else 0
        ratio_half = tempo_ref / tempo_live if tempo_live > 0 else 0
        octave_tolerance = 0.15
        if abs(ratio_double - 2.0) < octave_tolerance:
            return tempo_live / 2.0
        elif abs(ratio_half - 2.0) < octave_tolerance:
            return tempo_live * 2.0
        return tempo_live
    