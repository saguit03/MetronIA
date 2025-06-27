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

    def mfcc_features(self, reference_audio, live_audio, sampling_rate):
        ref_feat = self.feature_extractor.extract_mfcc_features(reference_audio, sampling_rate)
        live_feat = self.feature_extractor.extract_mfcc_features(live_audio, sampling_rate)
        return ref_feat, live_feat

    def chroma_features(self, reference_audio, live_audio, sampling_rate):
        ref_feat = self.feature_extractor.extract_chroma_features(reference_audio, sampling_rate)
        live_feat = self.feature_extractor.extract_chroma_features(live_audio, sampling_rate)
        return ref_feat, live_feat

    def both_beat_spectrums(self, reference_audio, live_audio, sampling_rate):
        mfcc_ref, mfcc_live = self.mfcc_features(reference_audio, live_audio, sampling_rate)
        chroma_ref, chroma_live = self.chroma_features(reference_audio, live_audio, sampling_rate)
        mfcc_beat_spectrum = self.beat_spectrum(mfcc_ref, mfcc_live)
        chroma_beat_spectrum = self.beat_spectrum(chroma_ref, chroma_live)
        return mfcc_beat_spectrum, chroma_beat_spectrum

    def beat_spectrum(self, ref_feat, live_feat):
        D, wp = librosa.sequence.dtw(X=ref_feat.T, Y=live_feat.T, metric='cosine')
        wp = np.array(wp[::-1])

        aligned_live_feat = np.zeros_like(ref_feat)
        for i, (ref_idx, live_idx) in enumerate(wp):
            if ref_idx < len(aligned_live_feat) and live_idx < len(live_feat):
                aligned_live_feat[ref_idx] = live_feat[live_idx]

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
