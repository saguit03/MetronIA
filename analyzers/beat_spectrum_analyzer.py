import librosa
import numpy as np
from dataclasses import dataclass

from utils.config import AudioAnalysisConfig
from utils.feature_extractor import AudioFeatureExtractor


@dataclass
class BeatSpectrumResult:
    beat_ref: np.ndarray
    beat_aligned: np.ndarray
    similarity_diff: np.ndarray
    max_difference: float


class BeatSpectrumAnalyzer:
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
        self.feature_extractor = AudioFeatureExtractor(config)

    def chroma_features(self, reference_audio, live_audio, sampling_rate):
        ref_feat = self.feature_extractor.extract_chroma_features(reference_audio, sampling_rate)
        live_feat = self.feature_extractor.extract_chroma_features(live_audio, sampling_rate)
        return ref_feat, live_feat

    def beat_spectrum(self, reference_audio, live_audio, sampling_rate):
        ref_feat, live_feat = self.chroma_features(reference_audio, live_audio, sampling_rate)
        
        D, wp = librosa.sequence.dtw(X=ref_feat.T, Y=live_feat.T, metric='cosine')
        wp = np.array(wp[::-1])

        aligned_live_feat = np.zeros_like(ref_feat)
        for i, (idx_ref, idx_live) in enumerate(wp):
            if idx_ref < len(aligned_live_feat) and idx_live < len(live_feat):
                aligned_live_feat[idx_ref] = live_feat[idx_live]

        S_ref = self.feature_extractor.compute_self_similarity_matrix(ref_feat)
        S_aligned = self.feature_extractor.compute_self_similarity_matrix(aligned_live_feat)

        beat_ref = self.feature_extractor.compute_beat_spectrum(S_ref)
        beat_aligned = self.feature_extractor.compute_beat_spectrum(S_aligned)

        similarity_diff = np.abs(beat_ref - beat_aligned)
        max_difference = np.max(similarity_diff)

        return BeatSpectrumResult(
            beat_ref=beat_ref,
            beat_aligned=beat_aligned,
            similarity_diff=similarity_diff,
            max_difference=max_difference
        )
