"""
Utilidades comunes para análisis de onsets musicales.
"""

import numpy as np
import librosa
from typing import Tuple, Optional, List, Dict
from scipy.spatial.distance import cdist
from .config import AudioAnalysisConfig
from .onset_results import OnsetDTWAnalysisResult, OnsetMatch
from pathlib import Path
import pandas as pd

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
            Array de onsets
        """
        return librosa.onset.onset_detect(y=audio, sr=sr, units='time')
    
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
            - pitches: Alturas en Hz en cada onset
        """
        # Detectar onsets
        onsets = OnsetUtils.detect_onsets(audio, sr)
        
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
        
        return onsets, np.array(onset_pitches)
    
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

       
    def save_onsets_analysis_to_csv(self, dtw_onset_result: OnsetDTWAnalysisResult, 
                                   save_name: str, dir_path: Optional[str] = None) -> None:
        """
        Guarda el análisis detallado de onsets en un archivo CSV.
        
        Args:
            dtw_onset_result: Resultado del análisis DTW de onsets
            save_name: Nombre base para el archivo
            reference_path: Ruta del archivo de referencia (opcional)
        """
        # Crear directorio de resultados si no existe
        results_dir = Path(dir_path if dir_path else "results")
        results_dir.mkdir(parents=True, exist_ok=True)
          # Nombre del archivo CSV
        if save_name:
            csv_filename = results_dir / f"{save_name}_analysis.csv"
        else:
            csv_filename = results_dir / "analysis.csv"
        
        # Crear lista de datos para el CSV
        csv_data = []
        
        # Procesar todos los matches (correctos, tarde, adelantado)
        for match in dtw_onset_result.matches:
            csv_data.append({
                'onset_type': match.classification.value,
                'ref_timestamp': round(match.ref_onset, 4),
                'live_timestamp': round(match.live_onset, 4),
                'time_difference_ms': round(match.time_adjustment, 2),
                'ref_pitch_hz': round(match.ref_pitch, 2) if match.ref_pitch > 0 else None,
                'live_pitch_hz': round(match.live_pitch, 2) if match.live_pitch > 0 else None,
                'pitch_similarity': round(match.pitch_similarity, 3),
                'is_matched': True
            })
        
        # Procesar onsets perdidos (missing)
        for ref_time, ref_pitch in dtw_onset_result.missing_onsets:
            csv_data.append({
                'onset_type': 'missing',
                'ref_timestamp': round(ref_time, 4),
                'live_timestamp': None,
                'time_difference_ms': None,
                'ref_pitch_hz': round(ref_pitch, 2) if ref_pitch > 0 else None,
                'live_pitch_hz': None,
                'pitch_similarity': None,
                'is_matched': False
            })
        
        # Procesar onsets extra
        for live_time, live_pitch in dtw_onset_result.extra_onsets:
            csv_data.append({
                'onset_type': 'extra',
                'ref_timestamp': None,
                'live_timestamp': round(live_time, 4),
                'time_difference_ms': None,
                'ref_pitch_hz': None,
                'live_pitch_hz': round(live_pitch, 2) if live_pitch > 0 else None,
                'pitch_similarity': None,
                'is_matched': False
            })
        
        # Crear DataFrame y ordenar por timestamp de referencia
        df = pd.DataFrame(csv_data)
        
        # Ordenar por timestamp de referencia (poner None al final)
        df_sorted = df.sort_values(
            by=['ref_timestamp', 'live_timestamp'], 
            na_position='last'
        )
        
        # Guardar CSV
        df_sorted.to_csv(csv_filename, index=False, encoding='utf-8')
          # Logging mejorado
        print(f"💾 Análisis de onsets guardado en: {csv_filename}")
        print(f"   📊 Total de onsets analizados: {len(csv_data)}")
        print(f"   ✅ Correctos: {len([d for d in csv_data if d['onset_type'] == 'correct'])}")
        print(f"   ⏰ Tarde: {len([d for d in csv_data if d['onset_type'] == 'late'])}")
        print(f"   ⚡ Adelantado: {len([d for d in csv_data if d['onset_type'] == 'early'])}")
        print(f"   ❌ Perdidos: {len([d for d in csv_data if d['onset_type'] == 'missing'])}")
        print(f"   ➕ Extra: {len([d for d in csv_data if d['onset_type'] == 'extra'])}")
    

class MusicalTimeUtils:
    """Utilidades para cálculos de tiempo musical."""
    
    @staticmethod
    def calculate_musical_duration(tempo: float, note_type: str = 'sixteenth') -> float:
        """
        Calcula la duración de una nota musical en segundos basada en el tempo.
        
        Args:
            tempo: Tempo en BPM (beats per minute)
            note_type: Tipo de nota ('whole', 'half', 'quarter', 'eighth', 'sixteenth')
            
        Returns:
            Duración en segundos
        """
        # Duración de una negra en segundos
        quarter_note_duration = 60.0 / tempo
        
        note_values = {
            'whole': 4.0,      # redonda
            'half': 2.0,       # blanca
            'quarter': 1.0,    # negra
            'eighth': 0.5,     # corchea
            'sixteenth': 0.25  # semicorchea
        }
        
        if note_type not in note_values:
            note_type = 'sixteenth'  # default: usar la subdivisión más pequeña
            
        duration = quarter_note_duration * note_values[note_type]
        return duration
    
    @staticmethod
    def detect_smallest_subdivision(onsets: np.ndarray, tempo: float) -> str:
        """
        Detecta la subdivisión musical más pequeña presente en una secuencia de onsets.
        
        Args:
            onsets: Array de tiempos de onsets en segundos
            tempo: Tempo en BPM
            
        Returns:
            Tipo de nota que representa la subdivisión más pequeña
        """
        if len(onsets) < 2:
            return 'sixteenth'  # default
        
        # Calcular intervalos entre onsets consecutivos
        intervals = np.diff(onsets)
        
        # Calcular duraciones de diferentes tipos de notas
        quarter_duration = 60.0 / tempo
        note_durations = {
            'quarter': quarter_duration,
            'eighth': quarter_duration * 0.5,
            'sixteenth': quarter_duration * 0.25,
        }
        
        # Encontrar el intervalo más pequeño significativo (ignorar valores muy pequeños)
        min_significant_interval = np.min(intervals[intervals > 0.05])  # Ignorar intervalos < 50ms
        
        # Determinar qué tipo de nota se aproxima más al intervalo mínimo
        best_match = 'sixteenth'
        best_diff = float('inf')
        
        for note_type, duration in note_durations.items():
            diff = abs(min_significant_interval - duration)
            if diff < best_diff:
                best_diff = diff
                best_match = note_type
        
        return best_match
    
    @staticmethod
    def get_tempo_based_margins(tempo: float, subdivision: str = None) -> Tuple[float, float]:
        """
        Calcula márgenes de análisis basados en el tempo y la subdivisión musical.
        
        Args:
            tempo: Tempo en BPM
            subdivision: Subdivisión a usar ('sixteenth', 'eighth', etc.)
            
        Returns:
            Tupla con (margen_estricto, margen_amplio) en segundos
        """
        if subdivision is None:
            subdivision = 'sixteenth'  # Usar la subdivisión más pequeña por defecto
            
        # Calcular duración de la subdivisión
        subdivision_duration = MusicalTimeUtils.calculate_musical_duration(tempo, subdivision)
        
        # Márgenes como porcentaje de la duración de la subdivisión
        strict_margin = subdivision_duration * 0.15   # 15% de la duración
        wide_margin = subdivision_duration * 0.40     # 40% de la duración
        
        # Limitar los márgenes para evitar valores extremos
        strict_margin = max(0.010, min(strict_margin, 0.050))  # Entre 10ms y 50ms
        wide_margin = max(0.025, min(wide_margin, 0.150))     # Entre 25ms y 150ms
        
        return strict_margin, wide_margin


class DTWUtils:
    """Utilidades para alineamiento DTW de onsets."""
    
    @staticmethod
    def align_onsets_with_dtw(onsets_live: np.ndarray, wp: np.ndarray, 
                             hop_length: int, sr: int) -> np.ndarray:
        """
        Alinea los onsets del audio en vivo usando el camino DTW.
        
        Args:
            onsets_live: Onsets detectados en el audio en vivo (en segundos)
            wp: Camino DTW (warping path) como array de pares [ref_frame, live_frame]
            hop_length: Hop length usado para extraer features
            sr: Sample rate
            
        Returns:
            Array de onsets alineados temporalmente
        """
        if len(onsets_live) == 0:
            return np.array([])
        
        # Convertir onsets de segundos a frames
        onsets_frames = librosa.time_to_frames(onsets_live, sr=sr, hop_length=hop_length)
        
        # Crear array para almacenar onsets alineados
        aligned_onsets = []
        
        for onset_frame in onsets_frames:
            # Buscar el frame más cercano en el camino DTW
            live_frames = wp[:, 1]  # Columna de frames del audio en vivo
            ref_frames = wp[:, 0]   # Columna de frames del audio de referencia
            
            # Encontrar el índice del frame más cercano
            closest_idx = np.argmin(np.abs(live_frames - onset_frame))
            
            # Obtener el frame correspondiente en la referencia
            aligned_ref_frame = ref_frames[closest_idx]
            
            # Convertir de vuelta a tiempo
            aligned_time = librosa.frames_to_time(aligned_ref_frame, sr=sr, hop_length=hop_length)
            aligned_onsets.append(aligned_time)
        
        return np.array(aligned_onsets)
    
    @staticmethod
    def create_dtw_features(onsets: np.ndarray, pitches: np.ndarray, 
                           pitch_weight: float = 0.7, time_weight: float = 0.3) -> np.ndarray:
        """
        Crea features para DTW combinando tiempo y altura.
        
        Args:
            onsets: Tiempos de onsets
            pitches: Alturas en Hz
            pitch_weight: Peso de la información de altura
            time_weight: Peso de la información temporal
            
        Returns:
            Array de features combinadas [tiempo_normalizado, pitch_normalizado]
        """
        if len(onsets) == 0:
            return np.array([]).reshape(0, 2)
        
        # Normalizar tiempos (0 a 1)
        if len(onsets) > 1:
            time_normalized = (onsets - onsets.min()) / (onsets.max() - onsets.min())
        else:
            time_normalized = np.array([0.0])
        
        # Normalizar pitches (convertir a MIDI y normalizar)
        pitch_midi = np.array([OnsetUtils.hz_to_midi(p) if p > 0 else 0 for p in pitches])
        if len(pitch_midi) > 1 and pitch_midi.max() > pitch_midi.min():
            pitch_normalized = (pitch_midi - pitch_midi.min()) / (pitch_midi.max() - pitch_midi.min())
        else:
            pitch_normalized = np.zeros_like(pitch_midi)
        
        # Combinar features con pesos
        features = np.column_stack([
            time_normalized * time_weight,
            pitch_normalized * pitch_weight
        ])
        
        return features


class OnsetMatchingUtils:
    """Utilidades para emparejamiento de onsets."""
    
    @staticmethod
    def calculate_search_window(onsets_ref: np.ndarray, ref_idx: int, 
                               base_window: float = 0.1, adaptive: bool = True) -> float:
        """
        Calcula la ventana de búsqueda adaptativa para un onset de referencia.
        
        Args:
            onsets_ref: Array de onsets de referencia
            ref_idx: Índice del onset actual
            base_window: Ventana base en segundos
            adaptive: Si usar ventana adaptativa basada en densidad local
            
        Returns:
            Tamaño de ventana en segundos
        """
        if not adaptive or len(onsets_ref) < 3:
            return base_window
        
        # Calcular densidad local de onsets
        start_idx = max(0, ref_idx - 2)
        end_idx = min(len(onsets_ref), ref_idx + 3)
        local_onsets = onsets_ref[start_idx:end_idx]
        
        if len(local_onsets) < 2:
            return base_window
        
        # Calcular intervalo promedio en la región local
        local_intervals = np.diff(local_onsets)
        avg_interval = np.mean(local_intervals)
        
        # Ventana adaptativa: 25% del intervalo promedio local, pero mínimo base_window
        adaptive_window = max(base_window, avg_interval * 0.25)
        
        # Limitar la ventana máxima
        max_window = base_window * 3
        return min(adaptive_window, max_window)
    
    @staticmethod
    def find_candidates_in_window(ref_onset: float, live_onsets: List[float], 
                                 window_size: float) -> List[int]:
        """
        Encuentra candidatos de onsets en vivo dentro de una ventana temporal.
        
        Args:
            ref_onset: Tiempo del onset de referencia
            live_onsets: Lista de onsets en vivo disponibles  
            window_size: Tamaño de la ventana en segundos
            
        Returns:
            Lista de índices de candidatos válidos
        """
        candidates = []
        for i, live_onset in enumerate(live_onsets):
            if abs(live_onset - ref_onset) <= window_size:
                candidates.append(i)
        return candidates
    
    @staticmethod
    def select_best_candidate(ref_onset: float, candidates: List[int], 
                             live_onsets: List[float], used_indices: set,
                             prefer_unused: bool = True) -> Optional[int]:
        """
        Selecciona el mejor candidato de una lista.
        
        Args:
            ref_onset: Tiempo del onset de referencia
            candidates: Lista de índices candidatos
            live_onsets: Lista completa de onsets en vivo
            used_indices: Conjunto de índices ya utilizados
            prefer_unused: Si preferir candidatos no utilizados
            
        Returns:
            Índice del mejor candidato o None si no hay candidatos válidos
        """
        if not candidates:
            return None
        
        # Filtrar candidatos no utilizados si se prefiere
        if prefer_unused:
            unused_candidates = [c for c in candidates if c not in used_indices]
            if unused_candidates:
                candidates = unused_candidates
        
        # Seleccionar el candidato más cercano temporalmente
        best_candidate = None
        best_distance = float('inf')
        
        for candidate_idx in candidates:
            distance = abs(live_onsets[candidate_idx] - ref_onset)
            if distance < best_distance:
                best_distance = distance
                best_candidate = candidate_idx
        
        return best_candidate
