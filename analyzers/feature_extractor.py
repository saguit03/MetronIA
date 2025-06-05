"""
Extractor de características de audio para análisis musical.
"""

import numpy as np
import librosa
from scipy.spatial.distance import cdist
from .config import AudioAnalysisConfig


class AudioFeatureExtractor:
    """Extractor de características de audio."""
    
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
    
    def extract_mfcc_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extrae características MFCC del audio."""
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            hop_length=self.config.hop_length, 
            n_mfcc=self.config.n_mfcc
        )
        return librosa.util.normalize(mfcc, axis=1).T  # (frames, features)
    
    def compute_self_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """Calcula la matriz de auto-semejanza."""
        D = cdist(features, features, metric='cosine')
        return 1 - D
    
    def compute_beat_spectrum(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Calcula el beat spectrum de una matriz de semejanza."""
        n = similarity_matrix.shape[0]
        return np.array([
            np.mean(np.diag(similarity_matrix, k=lag)) 
            for lag in range(1, n)
        ])
