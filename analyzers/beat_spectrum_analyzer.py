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

    def analyze_beat_spectrum(self,  audio_ref: np.ndarray, audio_live: np.ndarray, sr: int) -> BeatSpectrumResult:
        """Analiza el beat spectrum de las características alineadas."""
        reference_features = self.feature_extractor.extract_chroma_features(audio_ref, sr)
        aligned_live_features = self.feature_extractor.extract_chroma_features(audio_live, sr)
        
        # Calcular matrices de auto-semejanza
        S_ref = self.feature_extractor.compute_self_similarity_matrix(reference_features)
        S_aligned = self.feature_extractor.compute_self_similarity_matrix(aligned_live_features)
        
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