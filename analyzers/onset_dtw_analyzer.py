"""
Analizador de onsets basado en DTW y altura para emparejamiento preciso.
"""

from typing import Tuple, Optional, List, Dict

import numpy as np

from .config import AudioAnalysisConfig
from .onset_results import OnsetDTWAnalysisResult, OnsetMatch, OnsetType
from .onset_utils import OnsetUtils


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
        self.tolerance_ms = config.tolerance_ms  # Tolerancia en milisegundos para emparejamiento
            
    def invalid_onsets(self, onsets_ref, pitches_ref, onsets_live, pitches_live):
        return OnsetDTWAnalysisResult(
            matches=[],
            missing_onsets=[(t, p) for t, p in zip(onsets_ref, pitches_ref)],
            extra_onsets=[(t, p) for t, p in zip(onsets_live, pitches_live)],
            dtw_path=np.array([]),
            alignment_cost=float('inf'),
            tolerance_ms=self.tolerance_ms
        )
    
    def validate_onsets(self, onsets_ref: np.ndarray, pitches_ref: np.ndarray,
                        onsets_live: np.ndarray, pitches_live: np.ndarray):
        
        # Verificaciones de validez más exhaustivas
        if len(onsets_ref) == 0 or len(onsets_live) == 0:
            return self.invalid_onsets(onsets_ref, pitches_ref, onsets_live, pitches_live)
        
        # Verificar que no haya valores NaN o infinitos en los onsets y pitches
        if (np.any(np.isnan(onsets_ref)) or np.any(np.isinf(onsets_ref)) or
            np.any(np.isnan(onsets_live)) or np.any(np.isinf(onsets_live)) or
            np.any(np.isnan(pitches_ref)) or np.any(np.isinf(pitches_ref)) or
            np.any(np.isnan(pitches_live)) or np.any(np.isinf(pitches_live))):
            
            print("⚠️ Valores NaN o infinitos detectados en onsets o pitches")
            print(f"   Onsets ref: NaN={np.any(np.isnan(onsets_ref))}, Inf={np.any(np.isinf(onsets_ref))}")
            print(f"   Onsets live: NaN={np.any(np.isnan(onsets_live))}, Inf={np.any(np.isinf(onsets_live))}")
            print(f"   Pitches ref: NaN={np.any(np.isnan(pitches_ref))}, Inf={np.any(np.isinf(pitches_ref))}")
            print(f"   Pitches live: NaN={np.any(np.isnan(pitches_live))}, Inf={np.any(np.isinf(pitches_live))}")
            
            # Filtrar valores válidos
            valid_ref_mask = ~(np.isnan(onsets_ref) | np.isinf(onsets_ref) | 
                              np.isnan(pitches_ref) | np.isinf(pitches_ref))
            valid_live_mask = ~(np.isnan(onsets_live) | np.isinf(onsets_live) | 
                               np.isnan(pitches_live) | np.isinf(pitches_live))
            
            onsets_ref = onsets_ref[valid_ref_mask]
            pitches_ref = pitches_ref[valid_ref_mask]
            onsets_live = onsets_live[valid_live_mask]
            pitches_live = pitches_live[valid_live_mask]
            
            print(f"   Después del filtrado: {len(onsets_ref)} onsets ref, {len(onsets_live)} onsets live")
            
            # Verificar nuevamente si quedan onsets válidos
        if len(onsets_ref) == 0 or len(onsets_live) == 0:
            return self.invalid_onsets(onsets_ref, pitches_ref, onsets_live, pitches_live)
    
    def dtw_valid_path(self, dtw_path: np.ndarray, onsets_ref: np.ndarray, onsets_live: np.ndarray, verbose: Optional[bool] = False) -> List[Tuple[int, int]]:
        max_ref_idx = len(onsets_ref) - 1
        max_live_idx = len(onsets_live) - 1
        
        valid_dtw_path = []
        invalid_indices_count = 0
        
        for ref_idx, live_idx in dtw_path:
            if ref_idx <= max_ref_idx and live_idx <= max_live_idx:
                valid_dtw_path.append((ref_idx, live_idx))
            else:
                invalid_indices_count += 1
        if verbose and invalid_indices_count > 0:
            print(f"⚠️ DTW path contenía {invalid_indices_count} índices inválidos")
            print(f"   Rango válido ref: 0-{max_ref_idx}, live: 0-{max_live_idx}")
            print(f"   Usando {len(valid_dtw_path)} emparejamientos válidos")

        return valid_dtw_path
    
    def get_matches(self, dtw_path: List[Tuple[int, int]], onsets_ref: np.ndarray, pitches_ref: np.ndarray, onsets_live: np.ndarray, pitches_live: np.ndarray, verbose: Optional[bool] = False) -> List[OnsetMatch]:
        valid_dtw_path = self.dtw_valid_path(dtw_path, onsets_ref, onsets_live)
        matches = []
        used_live_indices = set()
        prev_adj = None
        valid_dtw_path = valid_dtw_path[::-1]
       
        for ref_idx, live_idx in valid_dtw_path:
            if live_idx not in used_live_indices:
                ref_onset = onsets_ref[ref_idx]
                live_onset = onsets_live[live_idx]
                ref_pitch = pitches_ref[ref_idx]
                live_pitch = pitches_live[live_idx]
                time_adjustment = np.round((ref_onset - live_onset), self.config.round_decimals)
                diff_adj = time_adjustment - prev_adj if prev_adj is not None else 0.0
                pitch_similarity = OnsetUtils.calculate_pitch_similarity(ref_pitch, live_pitch)
                if abs(diff_adj) <= self.tolerance_ms:
                    classification = OnsetType.CORRECT
                elif diff_adj < 0:
                    classification = OnsetType.LATE
                else:
                    classification = OnsetType.EARLY
                if verbose:
                    print(f"Referencia ({ref_onset:.2f}) - Vivo ({live_onset:.2f}) \n Ajuste: {time_adjustment:.2f} ms")
                    print(f"Diferencia ajustada: {diff_adj:.2f} ms")
                    print(f"Clasificación: {classification.value}")
                
                match = OnsetMatch(
                    ref_onset=ref_onset,
                    live_onset=live_onset,
                    ref_pitch=ref_pitch,
                    live_pitch=live_pitch,
                    time_adjustment=time_adjustment,
                    pitch_similarity=pitch_similarity,
                    classification=classification
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


    def match_onsets_with_dtw(self, audio_ref: np.ndarray, audio_live: np.ndarray, 
                             sr: int, dtw_path, alignment_cost) -> OnsetDTWAnalysisResult:
        """
        Empareja onsets usando DTW y similitud de altura.
        
        Args:
            audio_ref: Audio de referencia
            audio_live: Audio en vivo
            sr: Sample rate
            
        Returns:
            Resultado del análisis con emparejamientos y ajustes temporales        
"""        
        # Detectar onsets y pitches
        onsets_ref, pitches_ref = OnsetUtils.detect_onsets_with_pitch(audio_ref, sr)
        onsets_live, pitches_live = OnsetUtils.detect_onsets_with_pitch(audio_live, sr)
        
        matches, unmatched_ref, unmatched_live = self.get_matches(dtw_path=dtw_path, onsets_ref=onsets_ref, pitches_ref=pitches_ref, onsets_live=onsets_live, pitches_live=pitches_live)

        return OnsetDTWAnalysisResult(
            matches=matches,
            missing_onsets=unmatched_ref,
            extra_onsets=unmatched_live,
            dtw_path=dtw_path,
            alignment_cost=alignment_cost,
            tolerance_ms=self.tolerance_ms
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
    