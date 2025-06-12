"""
Analizador de onsets musicales para detección de errores de timing.
"""

import numpy as np
import librosa
from typing import Tuple, Optional, List
from .config import AudioAnalysisConfig
from .results import OnsetAnalysisResult


class OnsetAnalyzer:
    """Analizador de onsets musicales con análisis basado en tempo."""
    
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
    
    def calculate_musical_duration(self, tempo: float, note_type: str = 'sixteenth') -> float:
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
    
    def detect_smallest_subdivision(self, onsets: np.ndarray, tempo: float) -> str:
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
    
    def get_tempo_based_margins(self, tempo: float, subdivision: str = None) -> Tuple[float, float]:
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
        subdivision_duration = self.calculate_musical_duration(tempo, subdivision)
        
        # Margen estricto: 10% de la duración de la subdivisión
        strict_margin = subdivision_duration * 0.1
        
        # Margen amplio: 25% de la duración de la subdivisión
        wide_margin = subdivision_duration * 0.25
        
        # Aplicar límites mínimos y máximos razonables
        strict_margin = max(0.010, min(strict_margin, 0.050))  # Entre 10ms y 50ms
        wide_margin = max(0.025, min(wide_margin, 0.150))     # Entre 25ms y 150ms
        
        return strict_margin, wide_margin
    
    def detect_onsets(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Detecta onsets en el audio."""
        return librosa.onset.onset_detect(y=audio, sr=sr, units='time')
    
    def align_onsets_with_dtw(self, onsets_live: np.ndarray, wp: np.ndarray, 
                             hop_length: int, sr: int) -> np.ndarray:
        """
        Alinea los onsets del audio en vivo usando el camino DTW.
        
        Args:
            onsets_live: Onsets detectados en el audio en vivo (en segundos)
            wp: Camino DTW (warping path) como array de pares [ref_frame, live_frame]
            hop_length: Hop length usado para extraer features
            sr: Sample rate
            
        Returns:
            Onsets del audio en vivo alineados a la escala temporal de referencia
        """
        # Convertir onsets de tiempo a frames
        onsets_live_frames = librosa.time_to_frames(onsets_live, sr=sr, hop_length=hop_length)
        
        # Crear mapeo de frames usando DTW
        live_to_ref_mapping = {}
        for ref_frame, live_frame in wp:
            live_to_ref_mapping[live_frame] = ref_frame
        
        # Alinear cada onset
        aligned_onsets = []
        for onset_frame in onsets_live_frames:
            # Encontrar el frame más cercano en el mapeo DTW
            if onset_frame in live_to_ref_mapping:
                aligned_frame = live_to_ref_mapping[onset_frame]
            else:
                # Buscar el frame más cercano disponible
                available_frames = np.array(list(live_to_ref_mapping.keys()))
                if len(available_frames) > 0:
                    closest_idx = np.argmin(np.abs(available_frames - onset_frame))
                    closest_live_frame = available_frames[closest_idx]
                    aligned_frame = live_to_ref_mapping[closest_live_frame]
                else:
                    # Fallback: mantener el frame original (no debería ocurrir)
                    aligned_frame = onset_frame
            
            # Convertir de vuelta a tiempo
            aligned_time = librosa.frames_to_time(aligned_frame, sr=sr, hop_length=hop_length)
            aligned_onsets.append(aligned_time)
        
        return np.array(aligned_onsets)    
    
    def compare_onsets_basic(self, audio_ref: np.ndarray, audio_live: np.ndarray, sr: int, 
                            wp: Optional[np.ndarray] = None, hop_length: int = 512, 
                            tempo: Optional[float] = None) -> Tuple:
        """
        Comparación básica de onsets con alineamiento DTW opcional y márgenes basados en tempo.
        
        Args:
            audio_ref: Audio de referencia
            audio_live: Audio en vivo
            sr: Sample rate
            wp: Camino DTW opcional para alineamiento
            hop_length: Hop length para conversión tiempo-frame
            tempo: Tempo en BPM para calcular márgenes musicales
        """
        onsets_ref = self.detect_onsets(audio_ref, sr)
        onsets_live = self.detect_onsets(audio_live, sr)
        
        # Si se proporciona DTW, alinear los onsets del audio en vivo
        if wp is not None:
            onsets_live_aligned = self.align_onsets_with_dtw(onsets_live, wp, hop_length, sr)
        else:
            onsets_live_aligned = onsets_live
            
        matched = []
        unmatched_ref = []
        unmatched_live = list(onsets_live_aligned)
        
        # Calcular margen basado en tempo si está disponible
        if tempo is not None:
            subdivision = self.detect_smallest_subdivision(onsets_ref, tempo)
            _, matching_margin = self.get_tempo_based_margins(tempo, subdivision)
        else:
            # Fallback al margen fijo multiplicado
            matching_margin = self.config.onset_margin * 4
        
        for onset in onsets_ref:
            if not unmatched_live:
                unmatched_ref.append(onset)
                continue
                
            diffs = np.abs(np.array(unmatched_live) - onset)
            
            if np.min(diffs) < matching_margin:
                idx = np.argmin(diffs)
                matched.append((onset, unmatched_live[idx]))
                unmatched_live.pop(idx)
            else:
                unmatched_ref.append(onset)
        
        return onsets_ref, onsets_live, matched, unmatched_ref, unmatched_live
    
    def compare_onsets_detailed(self, audio_ref: np.ndarray, audio_live: np.ndarray, sr: int,
                               wp: Optional[np.ndarray] = None, hop_length: int = 512) -> OnsetAnalysisResult:
        """
        Análisis detallado de onsets con clasificación de errores y alineamiento DTW opcional.
        Usa un algoritmo contextual que considera el orden temporal de las notas.
        
        Args:
            audio_ref: Audio de referencia
            audio_live: Audio en vivo
            sr: Sample rate
            wp: Camino DTW opcional para alineamiento
            hop_length: Hop length para conversión tiempo-frame
        """
        onsets_ref = self.detect_onsets(audio_ref, sr)
        onsets_live = self.detect_onsets(audio_live, sr)
        
        # Si se proporciona DTW, alinear los onsets del audio en vivo
        if wp is not None:
            onsets_live_aligned = self.align_onsets_with_dtw(onsets_live, wp, hop_length, sr)
        else:
            onsets_live_aligned = onsets_live
        
        return self._smart_onset_matching(onsets_ref, onsets_live_aligned)
    
    def _smart_onset_matching(self, onsets_ref: np.ndarray, onsets_live: np.ndarray) -> OnsetAnalysisResult:
        """
        Algoritmo inteligente de matching de onsets que considera el contexto temporal.
        """
        matched_correct = []
        matched_early = []
        matched_late = []
        unmatched_ref = []
        unmatched_live = list(onsets_live)
        
        ref_idx = 0
        
        while ref_idx < len(onsets_ref):
            current_ref_onset = onsets_ref[ref_idx]
            
            if not unmatched_live:
                # No quedan onsets en vivo, el resto son faltantes
                unmatched_ref.extend(onsets_ref[ref_idx:])
                break
            
            # Calcular ventana de búsqueda dinámica
            search_window = self._calculate_search_window(onsets_ref, ref_idx)
            
            # Buscar candidatos dentro de la ventana
            candidates = self._find_candidates_in_window(
                current_ref_onset, unmatched_live, search_window
            )
            
            if not candidates:
                # No hay candidatos, pero esperamos a ver la siguiente nota de referencia
                if ref_idx < len(onsets_ref) - 1:
                    # Verificar si hay un onset en vivo que podría corresponder a la siguiente nota
                    next_ref_onset = onsets_ref[ref_idx + 1]
                    next_candidates = self._find_candidates_in_window(
                        next_ref_onset, unmatched_live, search_window
                    )
                    
                    if next_candidates:
                        # Hay candidato para la siguiente, la actual probablemente falta
                        unmatched_ref.append(current_ref_onset)
                        ref_idx += 1
                        continue
                
                # No hay candidatos ni para esta ni para la siguiente
                unmatched_ref.append(current_ref_onset)
                ref_idx += 1
                continue
            
            # Seleccionar el mejor candidato
            best_candidate_idx, best_live_onset = self._select_best_candidate(
                current_ref_onset, candidates, unmatched_live
            )
            
            # Clasificar el match
            time_diff = best_live_onset - current_ref_onset
            abs_diff = abs(time_diff)
            
            # Margen estricto para correcto
            strict_margin = self.config.onset_margin
            
            if abs_diff <= strict_margin:
                matched_correct.append((current_ref_onset, best_live_onset))
            elif time_diff < 0:  # Adelantado
                matched_early.append((current_ref_onset, best_live_onset))
            else:  # Atrasado
                matched_late.append((current_ref_onset, best_live_onset))
            
            # Remover el onset usado
            unmatched_live.pop(best_candidate_idx)
            ref_idx += 1
        
        return OnsetAnalysisResult(
            onsets_ref=onsets_ref,
            onsets_live=onsets_live,
            matched_correct=matched_correct,
            matched_early=matched_early,
            matched_late=matched_late,
            unmatched_ref=unmatched_ref,
            unmatched_live=unmatched_live
        )
    
    def _calculate_search_window(self, onsets_ref: np.ndarray, ref_idx: int) -> float:
        """
        Calcula la ventana de búsqueda dinámica basada en el contexto musical.
        """
        base_window = self.config.onset_margin * 8  # Ventana base amplia
        
        if ref_idx == 0:
            # Primera nota: usar ventana base
            return base_window
        
        if ref_idx >= len(onsets_ref) - 1:
            # Última nota: usar ventana base
            return base_window
        
        # Calcular intervalo con la nota anterior y siguiente
        prev_interval = onsets_ref[ref_idx] - onsets_ref[ref_idx - 1]
        next_interval = onsets_ref[ref_idx + 1] - onsets_ref[ref_idx]
        
        # La ventana es proporcional al intervalo más pequeño
        min_interval = min(prev_interval, next_interval)
        dynamic_window = min_interval * 0.5  # 50% del intervalo más pequeño
        
        # Usar el máximo entre ventana base y dinámica, con un límite superior
        return min(max(base_window, dynamic_window), 0.5)  # Máximo 500ms
    
    def _find_candidates_in_window(self, ref_onset: float, live_onsets: list, 
                                  window: float) -> list:
        """
        Encuentra candidatos de onsets en vivo dentro de la ventana de búsqueda.
        """
        candidates = []
        for i, live_onset in enumerate(live_onsets):
            if abs(live_onset - ref_onset) <= window:
                candidates.append((i, live_onset))
        return candidates
    
    def _select_best_candidate(self, ref_onset: float, candidates: list, 
                              live_onsets: list) -> tuple:
        """
        Selecciona el mejor candidato basado en proximidad temporal.
        """
        if len(candidates) == 1:
            return candidates[0]
        
        # Seleccionar el más cercano temporalmente
        best_idx = 0
        best_diff = abs(candidates[0][1] - ref_onset)
        
        for i, (idx, live_onset) in enumerate(candidates):
            diff = abs(live_onset - ref_onset)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        
        return candidates[best_idx]
    
    def detect_rhythm_pattern_errors(self, onsets_ref: np.ndarray, onsets_live: np.ndarray, 
                                   threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Detecta errores de patrón rítmico."""
        intervals_ref = np.diff(onsets_ref)
        intervals_live = np.diff(onsets_live)
        
        # Detectar repeticiones (onsets muy cercanos)
        repeats_live = np.where(np.diff(onsets_live) < 0.1)[0]
        
        # Detectar huecos grandes
        avg_interval_ref = np.mean(intervals_ref)
        large_gaps_live = np.where(intervals_live > avg_interval_ref + threshold)[0]
        
        return repeats_live, large_gaps_live
    
    def compare_onsets_without_alignment(self, audio_ref: np.ndarray, audio_live: np.ndarray, 
                                       sr: int, reference_tempo: Optional[float] = None) -> OnsetAnalysisResult:
        """
        Análisis de onsets SIN alineamiento DTW para preservar errores de timing reales.
        Usa márgenes basados en tempo para mayor precisión musical.
        
        Args:
            audio_ref: Audio de referencia
            audio_live: Audio en vivo
            sr: Sample rate
            reference_tempo: Tempo conocido en BPM (opcional)
            
        Returns:
            OnsetAnalysisResult con análisis detallado
        """
        # Detectar onsets en ambos audios
        onsets_ref = self.detect_onsets(audio_ref, sr)
        onsets_live = self.detect_onsets(audio_live, sr)
        
        # Si no se proporciona tempo, estimarlo del audio de referencia
        if reference_tempo is None:
            from . import tempo_analyzer
            tempo_est = tempo_analyzer.TempoAnalyzer(self.config)
            reference_tempo = tempo_est.extract_tempo(audio_ref, sr)
        
        # Detectar subdivisión musical más pequeña para ajustar márgenes
        subdivision = self.detect_smallest_subdivision(onsets_ref, reference_tempo)
        strict_margin, wide_margin = self.get_tempo_based_margins(reference_tempo, subdivision)
        
        # Usar algoritmo de matching inteligente con márgenes musicales
        result = self._smart_onset_matching_with_tempo_margins(
            onsets_ref, onsets_live, reference_tempo, strict_margin, wide_margin
        )
        
        return result
    
    def _smart_onset_matching_with_tempo_margins(self, onsets_ref: np.ndarray, onsets_live: np.ndarray,
                                               tempo: float, strict_margin: float, 
                                               wide_margin: float) -> OnsetAnalysisResult:
        """
        Algoritmo de matching con márgenes basados en tempo musical.
        
        Args:
            onsets_ref: Onsets de referencia
            onsets_live: Onsets en vivo (sin alinear)
            tempo: Tempo en BPM
            strict_margin: Margen estricto para onsets "correctos"
            wide_margin: Margen amplio para matching general
        """
        matched_correct = []
        matched_early = []
        matched_late = []
        unmatched_ref = []
        unmatched_live = list(onsets_live)
        
        ref_idx = 0
        
        while ref_idx < len(onsets_ref):
            current_ref_onset = onsets_ref[ref_idx]
            
            if not unmatched_live:
                # No quedan onsets en vivo, el resto son faltantes
                unmatched_ref.extend(onsets_ref[ref_idx:])
                break
            
            # Buscar candidatos dentro del margen amplio
            candidates = []
            for i, live_onset in enumerate(unmatched_live):
                if abs(live_onset - current_ref_onset) <= wide_margin:
                    candidates.append((i, live_onset))
            
            if not candidates:
                # No hay candidatos cercanos, esta nota falta
                unmatched_ref.append(current_ref_onset)
                ref_idx += 1
                continue
            
            # Seleccionar el candidato más cercano
            best_candidate_idx, best_live_onset = min(
                candidates, key=lambda x: abs(x[1] - current_ref_onset)
            )
            
            # Clasificar según timing
            time_diff = best_live_onset - current_ref_onset
            abs_diff = abs(time_diff)
            
            if abs_diff <= strict_margin:
                # Onset correcto (dentro del margen estricto)
                matched_correct.append((current_ref_onset, best_live_onset))
            elif time_diff < 0:  # Adelantado
                matched_early.append((current_ref_onset, best_live_onset))
            else:  # Atrasado
                matched_late.append((current_ref_onset, best_live_onset))
            
            # Remover el onset usado
            unmatched_live.pop(best_candidate_idx)
            ref_idx += 1
        
        return OnsetAnalysisResult(
            onsets_ref=onsets_ref,
            onsets_live=onsets_live,  # Mantenemos los originales sin alinear
            matched_correct=matched_correct,
            matched_early=matched_early,
            matched_late=matched_late,
            unmatched_ref=unmatched_ref,
            unmatched_live=unmatched_live
        )
