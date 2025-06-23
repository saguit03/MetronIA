"""
Analizador de beat spectrum para comparación de patrones rítmicos.
"""

import librosa
import numpy as np

from .config import AudioAnalysisConfig
from .feature_extractor import AudioFeatureExtractor
from .results import BeatSpectrumResult


class BeatSpectrumAnalyzer:
    """Analizador de beat spectrum."""
    
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
        self.feature_extractor = AudioFeatureExtractor(config)


    def beat_spectrum(self, reference_audio, live_audio, sampling_rate):
        hop_length = self.config.hop_length
        
        # === Cargar características ===
        ref_feat = self.feature_extractor.extract_mfcc_features(reference_audio, sampling_rate)
        live_feat = self.feature_extractor.extract_mfcc_features(live_audio, sampling_rate)

        # === DTW para alinear ===
        D, wp = librosa.sequence.dtw(X=ref_feat.T, Y=live_feat.T, metric='cosine')
        wp = np.array(wp[::-1])  

        # === Aplicar alineamiento a live_feat ===
        aligned_live_feat = np.zeros_like(ref_feat)
        for i, (ref_idx, live_idx) in enumerate(wp):
            if ref_idx < len(aligned_live_feat) and live_idx < len(live_feat):
                aligned_live_feat[ref_idx] = live_feat[live_idx]

        # === Calcular matrices de autosemejanza ===
        S_ref = self.feature_extractor.compute_self_similarity_matrix(ref_feat)
        S_aligned = self.feature_extractor.compute_self_similarity_matrix(aligned_live_feat)

        # === Beat spectrums ===
        beat_ref = self.feature_extractor.compute_beat_spectrum(S_ref)
        beat_aligned = self.feature_extractor.compute_beat_spectrum(S_aligned)

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
        # Alinear arrays para que tengan el mismo tamaño antes de la comparación
        print(f"🔍 Beat spectrum sizes: ref={len(beat_ref)}, aligned={len(beat_aligned)}")
        
        if len(beat_ref) != len(beat_aligned):
            max_length = max(len(beat_ref), len(beat_aligned))
            min_length = min(len(beat_ref), len(beat_aligned))
            
            print(f"⚠️ Diferentes tamaños de beat spectrum. Alineando con padding a: {max_length}")
            
            # Usar padding en lugar de recortar para no perder información
            if len(beat_ref) < max_length:
                # Pad beat_ref con ceros al final
                beat_ref_aligned = np.pad(beat_ref, (0, max_length - len(beat_ref)), mode='constant', constant_values=0)
                beat_aligned_aligned = beat_aligned
            else:
                # Pad beat_aligned con ceros al final
                beat_ref_aligned = beat_ref
                beat_aligned_aligned = np.pad(beat_aligned, (0, max_length - len(beat_aligned)), mode='constant', constant_values=0)
                
            print(f"✅ Arrays alineados con padding. Nuevos tamaños: {len(beat_ref_aligned)}, {len(beat_aligned_aligned)}")
        else:
            beat_ref_aligned = beat_ref
            beat_aligned_aligned = beat_aligned
          # Comparar con arrays del mismo tamaño
        similarity_diff = np.abs(beat_ref_aligned - beat_aligned_aligned)
        max_difference = np.max(similarity_diff)
        is_similar = max_difference <= self.config.beat_spectrum_threshold
        
        return BeatSpectrumResult(
            beat_ref=beat_ref_aligned,  # Usar arrays alineados
            beat_aligned=beat_aligned_aligned,  # Usar arrays alineados
            similarity_diff=similarity_diff,
            is_similar=is_similar,
            max_difference=max_difference
        )