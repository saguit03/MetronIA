"""
Analizador de onsets basado en DTW y altura para emparejamiento preciso.
"""

import numpy as np
import librosa
from typing import Tuple, Optional, List, Dict
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from .config import AudioAnalysisConfig
from .onset_results import OnsetMatch, OnsetDTWAnalysisResult, OnsetMatchClassified, OnsetType
from .onset_utils import OnsetUtils, DTWUtils, OnsetMatchingUtils

class OnsetDTWAnalyzer:
    """
    Analizador de onsets que usa DTW para alineamiento temporal y 
    emparejamiento basado en altura musical.
    """
    
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
        # Parámetros específicos para DTW y pitch matching
        self.pitch_weight = 0.7  # Peso de la similitud de altura
        self.time_weight = 0.3   # Peso de la proximidad temporal
        self.max_pitch_diff = 2.0  # Diferencia máxima de semitonos
        
    def detect_onsets_with_pitch(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detecta onsets y extrae la altura en cada onset.
        Delegado a OnsetUtils para funcionalidad común.
        """
        return OnsetUtils.detect_onsets_with_pitch(audio, sr)
    
    def calculate_pitch_similarity(self, pitch1_hz: float, pitch2_hz: float) -> float:
        """
        Calcula similitud de altura entre dos pitches.
        Delegado a OnsetUtils para funcionalidad común.
        """
        return OnsetUtils.calculate_pitch_similarity(pitch1_hz, pitch2_hz, self.max_pitch_diff)
    
    def create_dtw_features(self, onsets: np.ndarray, pitches: np.ndarray) -> np.ndarray:
        """
        Crea características para DTW combinando tiempo y pitch.
        Delegado a DTWUtils para funcionalidad común.
        """
        return DTWUtils.create_dtw_features(onsets, pitches, self.pitch_weight, self.time_weight)
    
    def dtw_distance(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Implementación básica de DTW.
        
        Args:
            x: Primera secuencia de características
            y: Segunda secuencia de características
            
        Returns:
            Tuple con (path, cost)
        """
        n, m = len(x), len(y)
        
        # Crear matriz de distancias
        distance_matrix = cdist(x, y, metric='euclidean')
        
        # Matriz de costos acumulados
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Llenar la matriz DTW
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = distance_matrix[i-1, j-1]
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # inserción
                    dtw_matrix[i, j-1],      # eliminación
                    dtw_matrix[i-1, j-1]     # match
                )
        
        # Reconstruir el camino óptimo
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            
            # Elegir el movimiento que dio el menor costo
            moves = [
                (dtw_matrix[i-1, j-1], i-1, j-1),  # diagonal
                (dtw_matrix[i-1, j], i-1, j),      # arriba
                (dtw_matrix[i, j-1], i, j-1)       # izquierda
            ]
            _, i, j = min(moves)
        
        path.reverse()
        return np.array(path), dtw_matrix[n, m]
    
    def align_with_dtw(self, features_ref: np.ndarray, features_live: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Alinea las características usando DTW.
        
        Args:
            features_ref: Características del audio de referencia
            features_live: Características del audio en vivo
            
        Returns:
            Tuple con (path, cost) donde:
            - path: Camino DTW como array de pares [ref_idx, live_idx]
            - cost: Costo del alineamiento
        """
        return self.dtw_distance(features_ref, features_live)
    
    def match_onsets_with_dtw(self, audio_ref: np.ndarray, audio_live: np.ndarray, 
                             sr: int) -> OnsetDTWAnalysisResult:
        """
        Empareja onsets usando DTW y similitud de altura.
        
        Args:
            audio_ref: Audio de referencia
            audio_live: Audio en vivo
            sr: Sample rate
            
        Returns:
            Resultado del análisis con emparejamientos y ajustes temporales
        """        # Detectar onsets y pitches
        onsets_ref, pitches_ref = self.detect_onsets_with_pitch(audio_ref, sr)
        onsets_live, pitches_live = self.detect_onsets_with_pitch(audio_live, sr)        
        if len(onsets_ref) == 0 or len(onsets_live) == 0:
            return OnsetDTWAnalysisResult(
                matches=[],  # Lista vacía de matches clasificados
                missing_onsets=[(t, p) for t, p in zip(onsets_ref, pitches_ref)],
                extra_onsets=[(t, p) for t, p in zip(onsets_live, pitches_live)],
                dtw_path=np.array([]),
                alignment_cost=float('inf'),
                tolerance_ms=100.0  # valor por defecto
            )
        
        # Crear características para DTW
        features_ref = self.create_dtw_features(onsets_ref, pitches_ref)
        features_live = self.create_dtw_features(onsets_live, pitches_live)
        
        # Alinear con DTW
        dtw_path, alignment_cost = self.align_with_dtw(features_ref, features_live)
        
        # Crear emparejamientos basados en el camino DTW
        matches = []
        used_live_indices = set()
        
        for ref_idx, live_idx in dtw_path:
            if live_idx not in used_live_indices:
                ref_onset = onsets_ref[ref_idx]
                live_onset = onsets_live[live_idx]
                ref_pitch = pitches_ref[ref_idx]
                live_pitch = pitches_live[live_idx]
                
                # Calcular ajuste temporal necesario
                time_adjustment = (ref_onset - live_onset) * 1000  # en ms
                
                # Calcular similitud de altura
                pitch_similarity = self.calculate_pitch_similarity(ref_pitch, live_pitch)
                
                match = OnsetMatch(
                    ref_onset=ref_onset,
                    live_onset=live_onset,
                    ref_pitch=ref_pitch,
                    live_pitch=live_pitch,
                    time_adjustment=time_adjustment,
                    pitch_similarity=pitch_similarity
                )
                
                matches.append(match)
                used_live_indices.add(live_idx)
        
        # Identificar onsets no emparejados
        matched_ref_indices = {ref_idx for ref_idx, _ in dtw_path}
        matched_live_indices = used_live_indices
        
        unmatched_ref = [(onsets_ref[i], pitches_ref[i]) 
                        for i in range(len(onsets_ref)) 
                        if i not in matched_ref_indices]
        
        unmatched_live = [(onsets_live[i], pitches_live[i]) 
                         for i in range(len(onsets_live)) 
                         if i not in matched_live_indices]
          # Convertir matches a la nueva estructura clasificada
        correct_matches = []
        late_matches = []
        early_matches = []
        
        # Clasificar los matches basándose en el ajuste temporal
        for match in matches:
            time_adj = match.time_adjustment
            if abs(time_adj) <= 50.0:  # Tolerancia para considerar "correcto"
                correct_matches.append(match)
            elif time_adj > 0:  # Tarde
                late_matches.append(match)
            else:  # Temprano
                early_matches.append(match)
          # Crear lista unificada de matches clasificados
        classified_matches = []
        for match in correct_matches:
            classified_matches.append(OnsetMatchClassified(
                match.ref_onset, match.live_onset, match.ref_pitch, match.live_pitch,
                match.time_adjustment, match.pitch_similarity, OnsetType.CORRECT
            ))
        for match in late_matches:
            classified_matches.append(OnsetMatchClassified(
                match.ref_onset, match.live_onset, match.ref_pitch, match.live_pitch,
                match.time_adjustment, match.pitch_similarity, OnsetType.LATE
            ))
        for match in early_matches:
            classified_matches.append(OnsetMatchClassified(
                match.ref_onset, match.live_onset, match.ref_pitch, match.live_pitch,
                match.time_adjustment, match.pitch_similarity, OnsetType.EARLY
            ))
        
        return OnsetDTWAnalysisResult(
            matches=classified_matches,
            missing_onsets=unmatched_ref,
            extra_onsets=unmatched_live,
            dtw_path=dtw_path,
            alignment_cost=alignment_cost,
            tolerance_ms=50.0  # Tolerancia utilizada
        )
    
    def get_alignment_adjustments(self, result: OnsetDTWAnalysisResult) -> Dict[str, List[float]]:
        """
        Extrae los ajustes temporales necesarios para alinear el audio en vivo.
        
        Args:
            result: Resultado del análisis DTW
            
        Returns:
            Diccionario con estadísticas de los ajustes temporales        """
        # Obtener todos los matches de la lista unificada
        adjustments = [match.time_adjustment for match in result.matches]
        
        if not adjustments:
            return {
                'adjustments_ms': [],
                'mean_adjustment': 0.0,
                'std_adjustment': 0.0,
                'max_adjustment': 0.0,
                'min_adjustment': 0.0
            }
        
        return {
            'adjustments_ms': adjustments,
            'mean_adjustment': np.mean(adjustments),
            'std_adjustment': np.std(adjustments),
            'max_adjustment': np.max(adjustments),
            'min_adjustment': np.min(adjustments)
        }
    
    def analyze_pitch_accuracy(self, result: OnsetDTWAnalysisResult) -> Dict[str, float]:
        """
        Analiza la precisión de altura en los emparejamientos.
        
        Args:
            result: Resultado del análisis DTW
            
        Returns:
            Diccionario con métricas de precisión de altura
        """
        if not result.matches:
            return {
                'mean_pitch_similarity': 0.0,
                'pitch_accuracy_rate': 0.0,
                'perfect_pitch_matches': 0
            }
        
        similarities = [match.pitch_similarity for match in result.matches]
        perfect_matches = sum(1 for s in similarities if s > 0.95)
        
        return {
            'mean_pitch_similarity': np.mean(similarities),
            'pitch_accuracy_rate': perfect_matches / len(similarities),
            'perfect_pitch_matches': perfect_matches
        }
    
    def analyze_onsets_with_rhythm_consistency(self, audio_ref: np.ndarray, audio_live: np.ndarray, 
                                             sr: int, tolerance_ms: float = 1.0) -> OnsetDTWAnalysisResult:
        """
        Analiza onsets con DTW y clasifica errores basándose en consistencia rítmica.
        
        Args:
            audio_ref: Audio de referencia
            audio_live: Audio en vivo
            sr: Sample rate
            tolerance_ms: Tolerancia para considerar ritmo consistente
            
        Returns:
            Resultado completo del análisis DTW con clasificación de errores
        """
        # Primero hacer el análisis DTW básico
        basic_result = self.match_onsets_with_dtw(audio_ref, audio_live, sr)        
        if not basic_result.matches:
            return OnsetDTWAnalysisResult(
                matches=[],  # Lista vacía de matches clasificados
                missing_onsets=basic_result.missing_onsets,
                extra_onsets=basic_result.extra_onsets,
                dtw_path=basic_result.dtw_path,
                alignment_cost=basic_result.alignment_cost,
                tolerance_ms=tolerance_ms
            )
        
        # Clasificar matches según consistencia rítmica
        correct_matches = []
        late_matches = []
        early_matches = []
        prev_adj = None
        
        for match in basic_result.matches:
            adj = match.time_adjustment
            
            if prev_adj is None:
                # Primer match siempre se considera correcto como referencia
                correct_matches.append(match)
            else:
                # Verificar si mantiene consistencia rítmica
                if abs(adj - prev_adj) <= tolerance_ms:
                    correct_matches.append(match)
                else:
                    # Clasificar según el signo del time_adjustment
                    # time_adjustment = (ref_onset - live_onset) * 1000
                    # Si time_adjustment < 0: live_onset > ref_onset → TARDE
                    # Si time_adjustment > 0: live_onset < ref_onset → ADELANTADO
                    if adj < 0:
                        late_matches.append(match)  # Tarde (live > ref)
                    else:
                        early_matches.append(match)  # Adelantado (live < ref)
            
            prev_adj = adj
          # Crear lista unificada de matches clasificados
        classified_matches = []
        for match in correct_matches:
            classified_matches.append(OnsetMatchClassified(
                match.ref_onset, match.live_onset, match.ref_pitch, match.live_pitch,
                match.time_adjustment, match.pitch_similarity, OnsetType.CORRECT
            ))
        for match in late_matches:
            classified_matches.append(OnsetMatchClassified(
                match.ref_onset, match.live_onset, match.ref_pitch, match.live_pitch,
                match.time_adjustment, match.pitch_similarity, OnsetType.LATE
            ))
        for match in early_matches:
            classified_matches.append(OnsetMatchClassified(
                match.ref_onset, match.live_onset, match.ref_pitch, match.live_pitch,
                match.time_adjustment, match.pitch_similarity, OnsetType.EARLY
            ))
        
        return OnsetDTWAnalysisResult(
            matches=classified_matches,
            missing_onsets=basic_result.missing_onsets,
            extra_onsets=basic_result.extra_onsets,
            dtw_path=basic_result.dtw_path,
            alignment_cost=basic_result.alignment_cost,
            tolerance_ms=tolerance_ms
        )
