"""
Extractor de características de audio para análisis musical.
"""

import librosa
import numpy as np
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

    def compute_self_similarity_matrix(self, features):
        D = cdist(features, features, metric='cosine')
        S = 1 - D
        return S

    def compute_beat_spectrum(self, similarity_matrix):
        n = similarity_matrix.shape[0]
        return np.array([np.mean(np.diag(similarity_matrix, k=lag)) for lag in range(1, n)])

    def extract_chroma_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extrae características cromáticas del audio."""
        chroma = librosa.feature.chroma_cqt(
            y=audio, 
            sr=sr, 
            hop_length=self.config.hop_length, 
            n_chroma=self.config.n_chroma
        )
        return librosa.util.normalize(chroma, axis=1).T
    