import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

from analyzers.onset_results import OnsetDTWAnalysisResult
from utils.config import VERBOSE_LOGGING
from utils.pitch_utils import hz_to_note

ROUND_DECIMALS = 2


class OnsetUtils:

    @staticmethod
    def detect_onsets(audio: np.ndarray, sr: int) -> np.ndarray:
        onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time', backtrack=True)

        if onsets is None or len(onsets) == 0:
            return np.array([])
        onsets = np.atleast_1d(onsets)

        rounded_onsets = np.round(onsets, ROUND_DECIMALS)
        unique_onsets = np.array(sorted(set(rounded_onsets)))

        return unique_onsets

    @staticmethod
    def detect_onsets_with_pitch(audio: np.ndarray, sr: int):
        onsets = OnsetUtils.detect_onsets(audio, sr)

        if len(onsets) == 0:
            print("⚠️ No se detectaron onsets.")
            return np.array([]), np.array([])
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, threshold=0.1)

        onset_notes = []
        for onset_time in onsets:
            onset_frame = librosa.time_to_frames(onset_time, sr=sr)
            if onset_frame < pitches.shape[1]:  # Evita acceder a un frame que no existe en los datos de pitch
                frame_pitches = pitches[:, onset_frame]
                frame_magnitudes = magnitudes[:, onset_frame]
                if np.any(frame_magnitudes > 0):  # Evita extraer el pitch si no se ha detectado ninguno
                    max_mag_idx = np.argmax(frame_magnitudes)
                    pitch_hz = frame_pitches[max_mag_idx]
                    if pitch_hz > 0:  # Evitar valores inválidos
                        onset_notes.append(hz_to_note(np.round(pitch_hz, ROUND_DECIMALS)))
                    else:
                        onset_notes.append('X0')
                else:
                    onset_notes.append('X0')
            else:
                onset_notes.append('X0')

        normalized_onsets = OnsetUtils.normalize_onsets_to_zero(onsets)
        unique_onsets = np.array(sorted(set(normalized_onsets)))

        return normalized_onsets, np.array(onset_notes)

    @staticmethod
    def save_onsets_analysis_to_csv(dtw_onset_result: OnsetDTWAnalysisResult, save_name: str, dir_path,
                                    mutation_name: Optional[str] = None) -> None:
        results_dir = Path(dir_path)
        results_dir.mkdir(parents=True, exist_ok=True)

        if mutation_name:
            csv_filename = results_dir / f"{mutation_name}_analysis.csv"
        elif save_name:
            csv_filename = results_dir / f"{save_name}.csv"
        else:
            csv_filename = results_dir / "analysis.csv"

        csv_data = []
        for match in dtw_onset_result.matches:
            csv_data.append({
                'onset_type': match.classification.value,
                'onset_ref_time': match.onset_ref,
                'onset_live_time': match.onset_live,
                'adjustment_ms': match.time_adjustment,
                'note_ref': match.note_ref,
                'note_live': match.note_live,
                'notes_similarity': match.note_similarity,

            })

        for ref_time, note_ref in dtw_onset_result.missing_onsets:
            csv_data.append({
                'onset_type': 'missing',
                'onset_ref_time': ref_time,
                'onset_live_time': None,
                'adjustment_ms': None,
                'note_ref': note_ref,
                'note_live': None,
                'notes_similarity': None
            })

        for live_time, note_live in dtw_onset_result.extra_onsets:
            csv_data.append({
                'onset_type': 'extra',
                'onset_ref_time': None,
                'onset_live_time': live_time,
                'adjustment_ms': None,
                'note_ref': None,
                'note_live': note_live,
                'notes_similarity': None
            })

        df = pd.DataFrame(csv_data)

        matched_mask = df['onset_type'].isin(['correct', 'late', 'early'])
        df_matched = df[matched_mask].drop_duplicates(subset=['onset_ref_time', 'onset_live_time'], keep='first')

        missing_mask = df['onset_type'] == 'missing'
        df_missing = df[missing_mask].drop_duplicates(subset=['onset_ref_time'], keep='first')

        extra_mask = df['onset_type'] == 'extra'
        df_extra = df[extra_mask].drop_duplicates(subset=['onset_live_time'], keep='first')

        df_filtered = pd.concat([df_matched, df_missing, df_extra], ignore_index=True)

        df_sorted = df_filtered.sort_values(
            by=['onset_ref_time', 'onset_live_time'],
            na_position='last'
        )

        df_sorted.to_csv(csv_filename, index=False, encoding='utf-8')

    @staticmethod
    def normalize_onsets_to_zero(onsets: np.ndarray) -> np.ndarray:
        if len(onsets) == 0:
            return onsets

        first_onset_time = onsets[0]
        if first_onset_time != 0.0:
            normalized_onsets = onsets - first_onset_time
            return np.round(normalized_onsets, 1)

        return onsets
