from typing import List, Optional

import librosa
import numpy as np
import traceback
from dataclasses  import dataclass

from utils.config import AudioAnalysisConfig

@dataclass
class TempoAnalysisResult:
    tempo_ref: float
    tempo_live: float
    difference: float
    tempo_proportion: float = 1.0
    original_ref_tempo: float = None
    original_live_tempo: float = None
    resampling_applied: bool = False


class TempoAnalyzer:
    
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
        
    def analyze_tempo_with_reference(self, ref_audio: np.ndarray, live_audio: np.ndarray, 
                                   sr: int, reference_tempo: Optional[float] = None) -> TempoAnalysisResult:
        if reference_tempo:
            tempo_ref = reference_tempo
        else:
            candidates_ref = self.extract_multiple_tempo_candidates(ref_audio, sr)
            tempo_ref = candidates_ref[0]
            
        candidates_live = self.extract_multiple_tempo_candidates(live_audio, sr, tempo_ref)
        best_tempo_live = self.get_best_candidate_tempo(candidates_live, tempo_ref)
        
        difference = abs(tempo_ref - best_tempo_live)
        
        return TempoAnalysisResult(
            tempo_ref=int(tempo_ref),
            tempo_live=int(best_tempo_live),
            difference=difference,
        )
    
    def get_best_candidate_tempo(self, candidates_live, tempo_ref):
        best_diff = None
        best_tempo_live = tempo_ref
        for candidate in candidates_live:
            tempo_live = self.correct_tempo_octave_errors(tempo_ref, candidate)
            diff = abs(tempo_ref - tempo_live)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_tempo_live = tempo_live
        return best_tempo_live
        
    def extract_multiple_tempo_candidates(self, audio: np.ndarray, sr: int, start_tempo: Optional[int] = None) -> List[float]:
        tempos = []
        
        try:
            if start_tempo:
                static_tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, start_bpm=start_tempo)
            else:
                static_tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            tempos.append(float(static_tempo.item()))
        except:
            traceback.print_exc()
        
        try:
            if start_tempo:
                dynamic_tempo = librosa.feature.tempo(y=audio, sr=sr, aggregate=np.median, start_bpm=start_tempo)[0]
            else:
                dynamic_tempo = librosa.feature.tempo(y=audio, sr=sr, aggregate=np.median)[0]
            tempos.append(float(dynamic_tempo))
        except:
            traceback.print_exc()
        
        # Eliminar duplicados y valores extremos
        tempos = [t for t in tempos if 40 <= t <= 300]
        tempos = list(set([round(t, 0) for t in tempos]))
        
        return sorted(tempos)
    
    def correct_tempo_octave_errors(self, tempo_ref: float, tempo_live: float) -> float:
        ratio_double = tempo_live / tempo_ref if tempo_ref > 0 else 0
        ratio_half = tempo_ref / tempo_live if tempo_live > 0 else 0
        octave_tolerance = 0.15
        if abs(ratio_double - 2.0) < octave_tolerance:
            return tempo_live / 2.0
        elif abs(ratio_half - 2.0) < octave_tolerance:
            return tempo_live * 2.0
        return tempo_live
    