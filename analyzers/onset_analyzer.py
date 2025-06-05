"""
Analizador de onsets musicales para detección de errores de timing.
"""

import numpy as np
import librosa
from typing import Tuple
from .config import AudioAnalysisConfig
from .results import OnsetAnalysisResult


class OnsetAnalyzer:
    """Analizador de onsets musicales."""
    
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
    
    def detect_onsets(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Detecta onsets en el audio."""
        return librosa.onset.onset_detect(y=audio, sr=sr, units='time')
    
    def compare_onsets_basic(self, audio_ref: np.ndarray, audio_live: np.ndarray, sr: int) -> Tuple:
        """Comparación básica de onsets."""
        onsets_ref = self.detect_onsets(audio_ref, sr)
        onsets_live = self.detect_onsets(audio_live, sr)
        
        matched = []
        unmatched_ref = []
        unmatched_live = list(onsets_live)
        
        for onset in onsets_ref:
            if not unmatched_live:
                unmatched_ref.append(onset)
                continue
                
            diffs = np.abs(np.array(unmatched_live) - onset)
            if np.min(diffs) < self.config.onset_margin:
                idx = np.argmin(diffs)
                matched.append((onset, unmatched_live[idx]))
                unmatched_live.pop(idx)
            else:
                unmatched_ref.append(onset)
        
        return onsets_ref, onsets_live, matched, unmatched_ref, unmatched_live
    
    def compare_onsets_detailed(self, audio_ref: np.ndarray, audio_live: np.ndarray, sr: int) -> OnsetAnalysisResult:
        """Análisis detallado de onsets con clasificación de errores."""
        onsets_ref = self.detect_onsets(audio_ref, sr)
        onsets_live = self.detect_onsets(audio_live, sr)
        
        matched_correct = []
        matched_early = []
        matched_late = []
        unmatched_ref = []
        unmatched_live = list(onsets_live)
        
        for onset in onsets_ref:
            if not unmatched_live:
                unmatched_ref.append(onset)
                continue
            
            diffs = np.array(unmatched_live) - onset
            abs_diffs = np.abs(diffs)
            min_idx = np.argmin(abs_diffs)
            min_diff = diffs[min_idx]
            
            if abs_diffs[min_idx] <= self.config.onset_margin:
                matched_correct.append((onset, unmatched_live[min_idx]))
                unmatched_live.pop(min_idx)
            elif abs_diffs[min_idx] <= 2 * self.config.onset_margin:
                if min_diff < 0:  # onset anticipado
                    matched_early.append((onset, unmatched_live[min_idx]))
                else:  # onset retrasado
                    matched_late.append((onset, unmatched_live[min_idx]))
                unmatched_live.pop(min_idx)
            else:
                unmatched_ref.append(onset)
        
        return OnsetAnalysisResult(
            onsets_ref=onsets_ref,
            onsets_live=onsets_live,
            matched_correct=matched_correct,
            matched_early=matched_early,
            matched_late=matched_late,
            unmatched_ref=unmatched_ref,
            unmatched_live=unmatched_live
        )
    
    def detect_rhythm_pattern_errors(self, onsets_ref: np.ndarray, onsets_live: np.ndarray, 
                                   threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Detecta errores de patrón rítmico."""
        intervals_ref = np.diff(onsets_ref)
        intervals_live = np.diff(onsets_live)
        
        # Detectar repeticiones (onsets muy cercanos)
        repeats_live = np.where(np.diff(onsets_live) < 0.1)[0]
        
        # Detectar huecos grandes
        avg_interval_ref = np.mean(intervals_ref)
        large_gaps_live = np.where(intervals_live > avg_interval_ref + threshold)[0]
        
        return repeats_live, large_gaps_live
