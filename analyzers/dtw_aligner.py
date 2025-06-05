"""
Alineador usando Dynamic Time Warping para análisis musical.
"""

import numpy as np
import librosa
from typing import Tuple
from .config import AudioAnalysisConfig
from .feature_extractor import AudioFeatureExtractor


class DTWAligner:
    """Alineador usando Dynamic Time Warping."""
    
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
        self.feature_extractor = AudioFeatureExtractor(config)
    
    def align_features(self, audio_ref: np.ndarray, audio_live: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Alinea características usando DTW."""
        # Extraer características
        ref_feat = self.feature_extractor.extract_mfcc_features(audio_ref, sr)
        live_feat = self.feature_extractor.extract_mfcc_features(audio_live, sr)
        
        # DTW para alinear
        D, wp = librosa.sequence.dtw(X=ref_feat.T, Y=live_feat.T, metric='cosine')
        wp = np.array(wp[::-1])  # Asegurar orden de principio a fin
        
        # Aplicar alineamiento
        aligned_live_feat = np.zeros_like(ref_feat)
        for ref_idx, live_idx in wp:
            if ref_idx < len(aligned_live_feat) and live_idx < len(live_feat):
                aligned_live_feat[ref_idx] = live_feat[live_idx]
        
        return ref_feat, aligned_live_feat, wp
    
    def evaluate_dtw_path(self, wp: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Evalúa la calidad del camino DTW."""
        wp = np.array(wp)
        ref_idxs, live_idxs = wp[:, 0], wp[:, 1]
        deltas = live_idxs - ref_idxs
        deviations = np.abs(deltas - np.mean(deltas))
        
        is_regular = np.max(deviations) <= self.config.dtw_tolerance * len(ref_idxs)
        return deviations, is_regular
