"""
Utilidades comunes para análisis de onsets musicales.
"""

from pathlib import Path
from typing import Tuple, Optional

import librosa
import numpy as np
import pandas as pd

from .config import VERBOSE_LOGGING
from .onset_results import OnsetDTWAnalysisResult


class OnsetUtils:
    """Utilidades comunes para análisis de onsets."""
    
    @staticmethod
    def detect_onsets(audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Detecta onsets en el audio.
        
        Args:
            audio: Señal de audio
            sr: Sample rate
        Returns:
            Array de onsets únicos ordenados
        """
        # Detectar onsets con librosa
        onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time') # TODO añadir bpm
        
        # Verificar si se detectaron onsets
        if onsets is None or len(onsets) == 0:
            return np.array([])
        onsets = np.atleast_1d(onsets)
        
        # Redondear onsets a 3 decimales para evitar problemas de precisión
        rounded_onsets = np.round(onsets, 3)
        unique_onsets = np.array(sorted(set(rounded_onsets)))
        
        return unique_onsets
    
    @staticmethod
    def detect_onsets_with_pitch(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detecta onsets y extrae la altura en cada onset.
        
        Args:
            audio: Señal de audio
            sr: Sample rate
            
        Returns:
            Tuple con (onsets_times, pitches) donde:
            - onsets_times: Tiempos de onset en segundos
            - pitches: Alturas en Hz en cada onset        """
        # Detectar onsets
        onsets = OnsetUtils.detect_onsets(audio, sr)
        
        # Verificar si se detectaron onsets
        if len(onsets) == 0:
            print("⚠️ No se detectaron onsets.")
            return np.array([]), np.array([])
        
        # Extraer pitch usando piptrack
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, threshold=0.1)
        
        # Obtener pitch en cada onset
        onset_pitches = []
        for onset_time in onsets:
            # Convertir tiempo a frame
            onset_frame = librosa.time_to_frames(onset_time, sr=sr)
            
            # Buscar el pitch más fuerte en ese frame
            if onset_frame < pitches.shape[1]:
                frame_pitches = pitches[:, onset_frame]
                frame_magnitudes = magnitudes[:, onset_frame]
                  # Encontrar el pitch con mayor magnitud
                if np.any(frame_magnitudes > 0):
                    max_mag_idx = np.argmax(frame_magnitudes)
                    pitch_hz = frame_pitches[max_mag_idx]
                    if pitch_hz > 0:
                        onset_pitches.append(pitch_hz)
                    else:
                        onset_pitches.append(0.0)  # Pitch no detectado
                else:
                    onset_pitches.append(0.0)
            else:
                onset_pitches.append(0.0)

        normalized_onsets = OnsetUtils.normalize_onsets_to_zero(onsets)
        
        return normalized_onsets, np.array(onset_pitches)
    
    @staticmethod
    def hz_to_midi(freq_hz: float) -> float:
        """Convierte frecuencia en Hz a número MIDI."""
        if freq_hz <= 0:
            return 0.0
        return 69 + 12 * np.log2(freq_hz / 440.0)
    
    @staticmethod
    def midi_to_hz(midi_note: float) -> float:
        """Convierte número MIDI a frecuencia en Hz."""
        if midi_note <= 0:
            return 0.0
        return 440.0 * (2 ** ((midi_note - 69) / 12))
    
    @staticmethod
    def calculate_pitch_similarity(pitch1_hz: float, pitch2_hz: float, max_diff_semitones: float = 2.0) -> float:
        """
        Calcula similitud de altura entre dos pitches.
        
        Args:
            pitch1_hz: Primer pitch en Hz
            pitch2_hz: Segundo pitch en Hz
            max_diff_semitones: Diferencia máxima en semitonos para considerar similares
        
        Returns:
            Similitud entre 0 y 1 (1 = idénticos)
        """
        if pitch1_hz <= 0 or pitch2_hz <= 0:
            return 0.0
        
        # Convertir a MIDI para trabajar en semitonos
        midi1 = OnsetUtils.hz_to_midi(pitch1_hz)
        midi2 = OnsetUtils.hz_to_midi(pitch2_hz)
        
        # Calcular diferencia en semitonos
        semitone_diff = abs(midi1 - midi2)
          # Similitud exponencial decreciente
        if semitone_diff > max_diff_semitones:
            return 0.0
        
        similarity = np.exp(-semitone_diff / max_diff_semitones)
        return similarity

    @staticmethod
    def save_onsets_analysis_to_csv(dtw_onset_result: OnsetDTWAnalysisResult,                                   save_name: str, dir_path, mutation_name: Optional[str] = None) -> None:
        """
        Guarda el análisis detallado de onsets en un archivo CSV.
        
        Args:
            dtw_onset_result: Resultado del análisis DTW de onsets
            save_name: Nombre base para el archivo
            dir_path: Directorio donde guardar el archivo (opcional)
            progress_bar: Barra de progreso (opcional)
            mutation_name: Nombre de la mutación para nombrar el archivo (opcional)
        """
        # Crear directorio de resultados si no existe
        results_dir = Path(dir_path)
        results_dir.mkdir(parents=True, exist_ok=True)
          # Nombre del archivo CSV - usar mutation_name_analysis.csv si está disponible
        if mutation_name:
            csv_filename = results_dir / f"{mutation_name}_analysis.csv"
        elif save_name:
            csv_filename = results_dir / f"{save_name}.csv"
        else:
            csv_filename = results_dir / "analysis.csv"
        
        # Crear lista de datos para el CSV
        csv_data = []
          # Procesar todos los matches (correctos, tarde, adelantado)
        for match in dtw_onset_result.matches:
            csv_data.append({
                'onset_type': match.classification.value,
                'ref_onset_time': round(match.ref_onset, 4),
                'live_onset_time': round(match.live_onset, 4),
                'adjustment_ms': round(match.time_adjustment, 4),
                'ref_pitch_hz': round(match.ref_pitch, 1) if match.ref_pitch > 0 else None,
                'live_pitch_hz': round(match.live_pitch, 1) if match.live_pitch > 0 else None,
                'pitch': round(match.ref_pitch, 1) if match.ref_pitch > 0 else round(match.live_pitch, 1),
                'pitch_similarity': round(match.pitch_similarity, 3),
                'is_matched': True
            })
        
        # Procesar onsets perdidos (missing)
        for ref_time, ref_pitch in dtw_onset_result.missing_onsets:
            csv_data.append({
                'onset_type': 'missing',
                'ref_onset_time': round(ref_time, 4),
                'live_onset_time': None,
                'adjustment_ms': None,
                'ref_pitch_hz': round(ref_pitch, 1) if ref_pitch > 0 else None,
                'live_pitch_hz': None,
                'pitch': round(ref_pitch, 1) if ref_pitch > 0 else 60,
                'pitch_similarity': None,
                'is_matched': False
            })
        
        # Procesar onsets extra
        for live_time, live_pitch in dtw_onset_result.extra_onsets:
            csv_data.append({
                'onset_type': 'extra',
                'ref_onset_time': None,
                'live_onset_time': round(live_time, 4),
                'adjustment_ms': None,
                'ref_pitch_hz': None,
                'live_pitch_hz': round(live_pitch, 1) if live_pitch > 0 else None,
                'pitch': round(live_pitch, 1) if live_pitch > 0 else 60,
                'pitch_similarity': None,
                'is_matched': False
            })
        df = pd.DataFrame(csv_data)
        
        matched_mask = df['onset_type'].isin(['correct', 'late', 'early'])
        df_matched = df[matched_mask].drop_duplicates(subset=['ref_onset_time', 'live_onset_time'], keep='first')
        
        missing_mask = df['onset_type'] == 'missing'
        df_missing = df[missing_mask].drop_duplicates(subset=['ref_onset_time'], keep='first')
        
        extra_mask = df['onset_type'] == 'extra'
        df_extra = df[extra_mask].drop_duplicates(subset=['live_onset_time'], keep='first')
        
        df_filtered = pd.concat([df_matched, df_missing, df_extra], ignore_index=True)
        
        df_sorted = df_filtered.sort_values(
            by=['ref_onset_time', 'live_onset_time'], 
            na_position='last'
        )
        
        df_sorted.to_csv(csv_filename, index=False, encoding='utf-8')
        if VERBOSE_LOGGING:
            print(f"💾 Análisis de onsets guardado en: {csv_filename}")
            print(f"   📊 Total de onsets analizados: {len(df_sorted)}")
            print(f"   ✅ Correctos: {len(df_sorted[df_sorted['onset_type'] == 'correct'])}")
            print(f"   ⏰ Tarde: {len(df_sorted[df_sorted['onset_type'] == 'late'])}")
            print(f"   ⚡ Adelantado: {len(df_sorted[df_sorted['onset_type'] == 'early'])}")
            print(f"   ❌ Perdidos: {len(df_sorted[df_sorted['onset_type'] == 'missing'])}")
            print(f"   ➕ Extra: {len(df_sorted[df_sorted['onset_type'] == 'extra'])}")
    
    @staticmethod
    def normalize_onsets_to_zero(onsets: np.ndarray) -> np.ndarray:
        """
        Normaliza los timestamps de onsets para que el primero empiece en 0.
        
        Args:
            onsets: Array de timestamps de onsets
            label: Etiqueta para logging (opcional)
            
        Returns:
            Array de onsets normalizados
        """
        if len(onsets) == 0:
            return onsets
        
        first_onset_time = onsets[0]
        if first_onset_time != 0.0:
            normalized_onsets = onsets - first_onset_time
            return normalized_onsets
        
        return onsets

    # def align_onsets_with_dtw(onsets_live: np.ndarray, wp: np.ndarray, 
    #                          hop_length: int, sr: int) -> np.ndarray:
    #     """
    #     Alinea los onsets del audio en vivo usando el camino DTW.
        
    #     Args:
    #         onsets_live: Onsets detectados en el audio en vivo (en segundos)
    #         wp: Camino DTW (warping path) como array de pares [ref_frame, live_frame]
    #         hop_length: Hop length usado para extraer features
    #         sr: Sample rate
            
    #     Returns:
    #         Array de onsets alineados temporalmente
    #     """
    #     if len(onsets_live) == 0:
    #         return np.array([])
        
    #     # Convertir onsets de segundos a frames
    #     onsets_frames = librosa.time_to_frames(onsets_live, sr=sr, hop_length=hop_length)
        
    #     # Crear array para almacenar onsets alineados
    #     aligned_onsets = []
        
    #     for onset_frame in onsets_frames:
    #         # Buscar el frame más cercano en el camino DTW
    #         live_frames = wp[:, 1]  # Columna de frames del audio en vivo
    #         ref_frames = wp[:, 0]   # Columna de frames del audio de referencia
            
    #         # Encontrar el índice del frame más cercano
    #         closest_idx = np.argmin(np.abs(live_frames - onset_frame))
            
    #         # Obtener el frame correspondiente en la referencia
    #         aligned_ref_frame = ref_frames[closest_idx]
            
    #         # Convertir de vuelta a tiempo
    #         aligned_time = librosa.frames_to_time(aligned_ref_frame, sr=sr, hop_length=hop_length)
    #         aligned_onsets.append(aligned_time)
        
    #     return np.array(aligned_onsets)
    
# class MusicalTimeUtils:
#     """Utilidades para cálculos de tiempo musical."""
    
#     @staticmethod
#     def calculate_musical_duration(tempo: float, note_type: str = 'sixteenth') -> float:
#         """
#         Calcula la duración de una nota musical en segundos basada en el tempo.
        
#         Args:
#             tempo: Tempo en BPM (beats per minute)
#             note_type: Tipo de nota ('whole', 'half', 'quarter', 'eighth', 'sixteenth')
            
#         Returns:
#             Duración en segundos
#         """
#         # Duración de una negra en segundos
#         quarter_note_duration = 60.0 / tempo
        
#         note_values = {
#             'whole': 4.0,      # redonda
#             'half': 2.0,       # blanca
#             'quarter': 1.0,    # negra
#             'eighth': 0.5,     # corchea
#             'sixteenth': 0.25  # semicorchea
#         }
        
#         if note_type not in note_values:
#             note_type = 'sixteenth'  # default: usar la subdivisión más pequeña
            
#         duration = quarter_note_duration * note_values[note_type]
#         return duration
    
#     @staticmethod
#     def detect_smallest_subdivision(onsets: np.ndarray, tempo: float) -> str:
#         """
#         Detecta la subdivisión musical más pequeña presente en una secuencia de onsets.
        
#         Args:
#             onsets: Array de tiempos de onsets en segundos
#             tempo: Tempo en BPM
            
#         Returns:
#             Tipo de nota que representa la subdivisión más pequeña
#         """
#         if len(onsets) < 2:
#             return 'sixteenth'  # default
        
#         # Calcular intervalos entre onsets consecutivos
#         intervals = np.diff(onsets)
        
#         # Calcular duraciones de diferentes tipos de notas
#         quarter_duration = 60.0 / tempo
#         note_durations = {
#             'quarter': quarter_duration,
#             'eighth': quarter_duration * 0.5,
#             'sixteenth': quarter_duration * 0.25,
#         }
        
#         # Encontrar el intervalo más pequeño significativo (ignorar valores muy pequeños)
#         min_significant_interval = np.min(intervals[intervals > 0.05])  # Ignorar intervalos < 50ms
#           # Determinar qué tipo de nota se aproxima más al intervalo mínimo
#         best_match = 'sixteenth'
#         best_diff = float('inf')
        
#         for note_type, duration in note_durations.items():
#             diff = abs(min_significant_interval - duration)
#             if diff < best_diff:
#                 best_diff = diff
#                 best_match = note_type
        
#         return best_match
    