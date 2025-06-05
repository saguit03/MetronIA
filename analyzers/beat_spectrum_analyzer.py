"""
Analizador de beat spectrum para comparación de patrones rítmicos.
"""

import numpy as np
from .config import AudioAnalysisConfig
from .results import BeatSpectrumResult
from .feature_extractor import AudioFeatureExtractor


class BeatSpectrumAnalyzer:
    """Analizador de beat spectrum."""
    
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
        self.feature_extractor = AudioFeatureExtractor(config)
    
    def analyze_beat_spectrum(self, ref_feat: np.ndarray, aligned_live_feat: np.ndarray) -> BeatSpectrumResult:
        """Analiza el beat spectrum de las características alineadas."""
        # Calcular matrices de auto-semejanza
        S_ref = self.feature_extractor.compute_self_similarity_matrix(ref_feat)
        S_aligned = self.feature_extractor.compute_self_similarity_matrix(aligned_live_feat)
        
        # Calcular beat spectrums
        beat_ref = self.feature_extractor.compute_beat_spectrum(S_ref)
        beat_aligned = self.feature_extractor.compute_beat_spectrum(S_aligned)
        
        # Comparar
        similarity_diff = np.abs(beat_ref - beat_aligned)
        max_difference = np.max(similarity_diff)
        is_similar = max_difference <= self.config.beat_spectrum_threshold
        
        return BeatSpectrumResult(
            beat_ref=beat_ref,
            beat_aligned=beat_aligned,
            similarity_diff=similarity_diff,
            is_similar=is_similar,
            max_difference=max_difference
        )
