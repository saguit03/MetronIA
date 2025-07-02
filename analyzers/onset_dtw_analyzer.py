from typing import Tuple, Optional, List
import numpy as np
import pandas as pd

from utils.config import AudioAnalysisConfig
from .onset_results import OnsetDTWAnalysisResult, OnsetMatch, OnsetType
from utils.onset_utils import OnsetUtils
from utils.pitch_utils import calculate_note_similarity

class OnsetDTWAnalyzer:

    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
        self.tolerance_ms = config.tolerance_ms

    def dtw_valid_path(self, dtw_path: np.ndarray, onsets_ref: np.ndarray, onsets_live: np.ndarray,
                       verbose: Optional[bool] = False) -> List[Tuple[int, int]]:
        max_idx_ref = len(onsets_ref) - 1
        max_idx_live = len(onsets_live) - 1

        valid_dtw_path = []
        invalid_indices_count = 0

        for idx_ref, idx_live in dtw_path:
            if idx_ref <= max_idx_ref and idx_live <= max_idx_live:
                valid_dtw_path.append((idx_ref, idx_live))
            else:
                invalid_indices_count += 1
        if verbose and invalid_indices_count > 0:
            print(f"⚠️ DTW path contenía {invalid_indices_count} índices inválidos")
            print(f"   Rango válido ref: 0-{max_idx_ref}, live: 0-{max_idx_live}")
            print(f"   Usando {len(valid_dtw_path)} emparejamientos válidos")

        return valid_dtw_path[::-1]

    def get_matches(self, dtw_path: List[Tuple[int, int]], onsets_ref: np.ndarray, pitches_ref: np.ndarray,  onsets_live: np.ndarray, pitches_live: np.ndarray) -> List[OnsetMatch]:
        valid_dtw_path = self.dtw_valid_path(dtw_path, onsets_ref, onsets_live)
        matches = []
        used_live_indices = set()
        prev_adj = None

        for ref_idx, live_idx in valid_dtw_path:
            onset_ref = onsets_ref[ref_idx]
            onset_live = onsets_live[live_idx]
            note_ref = pitches_ref[ref_idx]
            note_live = pitches_live[live_idx]
            time_adjustment = np.round((onset_ref - onset_live), self.config.round_decimals)
            diff_adj = time_adjustment - prev_adj if prev_adj is not None else 0.0
            if abs(time_adjustment) < self.tolerance_ms or abs(diff_adj) < self.tolerance_ms:
                classification = OnsetType.CORRECT
            elif diff_adj < 0:
                classification = OnsetType.LATE
            else:
                classification = OnsetType.EARLY
            note_similarity_value, note_interval = calculate_note_similarity(note_ref, note_live)

            match = OnsetMatch(
                onset_ref=onset_ref,
                onset_live=onset_live,
                note_ref=note_ref,
                note_live=note_live,
                time_adjustment=time_adjustment,
                classification=classification,
                note_similarity=note_similarity_value,
                note_interval=note_interval
            )
            matches.append(match)
            used_live_indices.add(live_idx)
            prev_adj = time_adjustment

        matched_ref_indices = {ref_idx for ref_idx, _ in valid_dtw_path}
        matched_live_indices = used_live_indices

        unmatched_ref = [(onsets_ref[i], pitches_ref[i])
                         for i in range(len(onsets_ref))
                         if i not in matched_ref_indices]

        unmatched_live = [(onsets_live[i], pitches_live[i])
                          for i in range(len(onsets_live))
                          if i not in matched_live_indices]
        return matches, unmatched_ref, unmatched_live

    def match_onsets_with_dtw(self, reference_audio: np.ndarray, live_audio: np.ndarray, sr: int, dtw_path) -> OnsetDTWAnalysisResult:
        onsets_ref, pitches_ref = OnsetUtils.detect_onsets_with_pitch(reference_audio, sr)
        onsets_live, pitches_live = OnsetUtils.detect_onsets_with_pitch(live_audio, sr)

        matches, unmatched_ref, unmatched_live = self.get_matches(dtw_path=dtw_path, onsets_ref=onsets_ref,  pitches_ref=pitches_ref, onsets_live=onsets_live, pitches_live=pitches_live)

        correct_notes = len([m for m in matches if m.note_similarity == 1.0])

        return OnsetDTWAnalysisResult(
            matches=matches,
            missing_onsets=unmatched_ref,
            extra_onsets=unmatched_live,
            correct_notes=correct_notes,
        )
