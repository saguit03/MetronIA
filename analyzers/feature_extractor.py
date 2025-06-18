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
    
    def extract_combined_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extrae características combinadas MFCC + chroma del audio.
        
        Devuelve una matriz (frames, n_mfcc + n_chroma)
        """
        mfcc = self.extract_mfcc_features(audio, sr)  # (frames, n_mfcc)
        chroma = self.extract_chroma_features(audio, sr)  # (frames, n_chroma)
        
        # Alinear por cantidad de frames
        min_frames = min(len(mfcc), len(chroma))
        mfcc = mfcc[:min_frames]
        chroma = chroma[:min_frames]
        
        combined = np.concatenate([mfcc, chroma], axis=1)  # (frames, features)
        return combined
