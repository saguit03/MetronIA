"""
Analizador de onsets musicales para detección de errores de timing.
"""

import numpy as np
import librosa
from typing import Tuple, Optional
from .config import AudioAnalysisConfig
from .results import OnsetAnalysisResult


class OnsetAnalyzer:
    """Analizador de onsets musicales."""
    
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
    
    def detect_onsets(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Detecta onsets en el audio."""
        return librosa.onset.onset_detect(y=audio, sr=sr, units='time')
    
    def align_onsets_with_dtw(self, onsets_live: np.ndarray, wp: np.ndarray, 
                             hop_length: int, sr: int) -> np.ndarray:
        """
        Alinea los onsets del audio en vivo usando el camino DTW.
        
        Args:
            onsets_live: Onsets detectados en el audio en vivo (en segundos)
            wp: Camino DTW (warping path) como array de pares [ref_frame, live_frame]
            hop_length: Hop length usado para extraer features
            sr: Sample rate
            
        Returns:
            Onsets del audio en vivo alineados a la escala temporal de referencia
        """
        # Convertir onsets de tiempo a frames
        onsets_live_frames = librosa.time_to_frames(onsets_live, sr=sr, hop_length=hop_length)
        
        # Crear mapeo de frames usando DTW
        live_to_ref_mapping = {}
        for ref_frame, live_frame in wp:
            live_to_ref_mapping[live_frame] = ref_frame
        
        # Alinear cada onset
        aligned_onsets = []
        for onset_frame in onsets_live_frames:
            # Encontrar el frame más cercano en el mapeo DTW
            if onset_frame in live_to_ref_mapping:
                aligned_frame = live_to_ref_mapping[onset_frame]
            else:
                # Buscar el frame más cercano disponible
                available_frames = np.array(list(live_to_ref_mapping.keys()))
                if len(available_frames) > 0:
                    closest_idx = np.argmin(np.abs(available_frames - onset_frame))
                    closest_live_frame = available_frames[closest_idx]
                    aligned_frame = live_to_ref_mapping[closest_live_frame]
                else:
                    # Fallback: mantener el frame original (no debería ocurrir)
                    aligned_frame = onset_frame
            
            # Convertir de vuelta a tiempo
            aligned_time = librosa.frames_to_time(aligned_frame, sr=sr, hop_length=hop_length)
            aligned_onsets.append(aligned_time)
        
        return np.array(aligned_onsets)
    def compare_onsets_basic(self, audio_ref: np.ndarray, audio_live: np.ndarray, sr: int, 
                            wp: Optional[np.ndarray] = None, hop_length: int = 512) -> Tuple:
        """
        Comparación básica de onsets con alineamiento DTW opcional.
        
        Args:
            audio_ref: Audio de referencia
            audio_live: Audio en vivo
            sr: Sample rate
            wp: Camino DTW opcional para alineamiento
            hop_length: Hop length para conversión tiempo-frame
        """
        onsets_ref = self.detect_onsets(audio_ref, sr)
        onsets_live = self.detect_onsets(audio_live, sr)
        
        # Si se proporciona DTW, alinear los onsets del audio en vivo
        if wp is not None:
            onsets_live_aligned = self.align_onsets_with_dtw(onsets_live, wp, hop_length, sr)
        else:
            onsets_live_aligned = onsets_live
        
        matched = []
        unmatched_ref = []
        unmatched_live = list(onsets_live_aligned)
        
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
    def compare_onsets_detailed(self, audio_ref: np.ndarray, audio_live: np.ndarray, sr: int,
                               wp: Optional[np.ndarray] = None, hop_length: int = 512) -> OnsetAnalysisResult:
        """
        Análisis detallado de onsets con clasificación de errores y alineamiento DTW opcional.
        
        Args:
            audio_ref: Audio de referencia
            audio_live: Audio en vivo
            sr: Sample rate
            wp: Camino DTW opcional para alineamiento
            hop_length: Hop length para conversión tiempo-frame
        """
        onsets_ref = self.detect_onsets(audio_ref, sr)
        onsets_live = self.detect_onsets(audio_live, sr)
        
        # Si se proporciona DTW, alinear los onsets del audio en vivo
        if wp is not None:
            onsets_live_aligned = self.align_onsets_with_dtw(onsets_live, wp, hop_length, sr)
        else:
            onsets_live_aligned = onsets_live
        
        matched_correct = []
        matched_early = []
        matched_late = []
        unmatched_ref = []
        unmatched_live = list(onsets_live_aligned)
        
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
            onsets_live=onsets_live_aligned,  # Usar los onsets alineados
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
