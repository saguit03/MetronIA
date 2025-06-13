"""
Clases de resultados para el análisis DTW de onsets.
"""

import numpy as np
from typing import List, Dict, Any, NamedTuple, Optional
from dataclasses import dataclass
from enum import Enum


class OnsetType(Enum):
    """Tipos de clasificación de onsets."""
    CORRECT = "correct"
    LATE = "late" 
    EARLY = "early"


class OnsetMatch(NamedTuple):
    """Representa un emparejamiento entre onsets de referencia y en vivo."""
    ref_onset: float
    live_onset: float
    ref_pitch: float
    live_pitch: float
    time_adjustment: float  # Ajuste temporal aplicado (ms)
    pitch_similarity: float  # Similitud de altura (0-1)


class OnsetMatchClassified(NamedTuple):
    """Representa un emparejamiento clasificado entre onsets de referencia y en vivo."""
    ref_onset: float
    live_onset: float
    ref_pitch: float
    live_pitch: float
    time_adjustment: float  # Ajuste temporal aplicado (ms)
    pitch_similarity: float  # Similitud de altura (0-1)
    classification: OnsetType  # Clasificación del onset


@dataclass
class OnsetDTWAnalysisResult:
    """
    Resultado completo del análisis DTW de onsets con clasificación de errores.
    
    Esta clase maneja los resultados del análisis DTW de onsets, incluyendo:
    - Todos los onsets emparejados (correctos, tarde, adelantado)
    - Onsets faltantes y extras
    """
    
    # Matches unificados (incluye todos los emparejamientos con clasificación)
    matches: List[OnsetMatchClassified]
    
    # Onsets no emparejados
    missing_onsets: List[tuple]  # (tiempo, pitch) de referencia
    extra_onsets: List[tuple]   # (tiempo, pitch) en vivo
    
    # Datos del DTW
    dtw_path: np.ndarray
    alignment_cost: float
    
    # Parámetros del análisis
    tolerance_ms: float = 1.0    
    def __post_init__(self):
        """Calcula estadísticas después de la inicialización."""
        self._calculate_stats()
    
    @property
    def correct_matches(self) -> List[OnsetMatch]:
        """Obtiene solo los matches correctos."""
        return [OnsetMatch(m.ref_onset, m.live_onset, m.ref_pitch, m.live_pitch, 
                          m.time_adjustment, m.pitch_similarity) 
                for m in self.matches if m.classification == OnsetType.CORRECT]
    
    @property
    def late_matches(self) -> List[OnsetMatch]:
        """Obtiene solo los matches tardíos."""
        return [OnsetMatch(m.ref_onset, m.live_onset, m.ref_pitch, m.live_pitch, 
                          m.time_adjustment, m.pitch_similarity) 
                for m in self.matches if m.classification == OnsetType.LATE]
    
    @property
    def early_matches(self) -> List[OnsetMatch]:
        """Obtiene solo los matches adelantados."""
        return [OnsetMatch(m.ref_onset, m.live_onset, m.ref_pitch, m.live_pitch, 
                          m.time_adjustment, m.pitch_similarity) 
                for m in self.matches if m.classification == OnsetType.EARLY]
    
    def _calculate_stats(self):
        """Calcula estadísticas del análisis."""
        self.total_ref_onsets = len(self.matches) + len(self.missing_onsets)
        self.total_live_onsets = len(self.matches) + len(self.extra_onsets)
        self.total_matches = len(self.matches)
        
        # Contar por clasificación
        correct_count = len([m for m in self.matches if m.classification == OnsetType.CORRECT])
        late_count = len([m for m in self.matches if m.classification == OnsetType.LATE])
        early_count = len([m for m in self.matches if m.classification == OnsetType.EARLY])
        
        # Tasas de error
        self.consistency_rate = correct_count / self.total_ref_onsets if self.total_ref_onsets > 0 else 0.0
        self.late_rate = late_count / self.total_ref_onsets if self.total_ref_onsets > 0 else 0.0
        self.early_rate = early_count / self.total_ref_onsets if self.total_ref_onsets > 0 else 0.0
        self.missing_rate = len(self.missing_onsets) / self.total_ref_onsets if self.total_ref_onsets > 0 else 0.0
        self.extra_rate = len(self.extra_onsets) / self.total_live_onsets if self.total_live_onsets > 0 else 0.0          # Estadísticas de ajustes temporales
        all_adjustments = [m.time_adjustment for m in self.matches]
        
        if all_adjustments:
            self.mean_adjustment = np.mean(all_adjustments)
            self.std_adjustment = np.std(all_adjustments)
            self.max_adjustment = np.max(all_adjustments)
            self.min_adjustment = np.min(all_adjustments)
        else:
            self.mean_adjustment = 0.0
            self.std_adjustment = 0.0
            self.max_adjustment = 0.0
            self.min_adjustment = 0.0
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas resumidas del análisis.
        
        Returns:
            Diccionario con estadísticas clave
        """
        correct_count = len([m for m in self.matches if m.classification == OnsetType.CORRECT])
        late_count = len([m for m in self.matches if m.classification == OnsetType.LATE]) 
        early_count = len([m for m in self.matches if m.classification == OnsetType.EARLY])
        
        return {
            'total_ref_onsets': self.total_ref_onsets,
            'total_live_onsets': self.total_live_onsets,
            'total_matches': self.total_matches,
            'correct_matches': correct_count,
            'late_matches': late_count,
            'early_matches': early_count,
            'missing_onsets': len(self.missing_onsets),
            'extra_onsets': len(self.extra_onsets),
            'consistency_rate': self.consistency_rate,
            'late_rate': self.late_rate,
            'early_rate': self.early_rate,
            'missing_rate': self.missing_rate,
            'extra_rate': self.extra_rate,
            'mean_adjustment_ms': self.mean_adjustment,
            'std_adjustment_ms': self.std_adjustment,
            'max_adjustment_ms': self.max_adjustment,
            'min_adjustment_ms': self.min_adjustment,
            'alignment_cost': self.alignment_cost,            'tolerance_ms': self.tolerance_ms
        }
    
    def get_csv_data(self, mutation_category: str = "", mutation_name: str = "") -> Dict[str, Any]:
        """
        Extrae los datos en formato CSV.
        
        Args:
            mutation_category: Categoría de la mutación (opcional)
            mutation_name: Nombre de la mutación (opcional)
            
        Returns:
            Diccionario con los datos formateados para CSV
        """
        correct_count = len([m for m in self.matches if m.classification == OnsetType.CORRECT])
        late_count = len([m for m in self.matches if m.classification == OnsetType.LATE])
        early_count = len([m for m in self.matches if m.classification == OnsetType.EARLY])
        
        return {
            # Información de mutación
            'mutation_category': mutation_category,
            'mutation_name': mutation_name,
            
            # Conteos de onsets
            'dtw_onsets_total_ref': self.total_ref_onsets,
            'dtw_onsets_total_live': self.total_live_onsets,
            'dtw_onsets_total_matches': self.total_matches,
            
            # Clasificación de onsets
            'dtw_onsets_correct': correct_count,
            'dtw_onsets_late': late_count,
            'dtw_onsets_early': early_count,
            'dtw_onsets_missing': len(self.missing_onsets),
            'dtw_onsets_extra': len(self.extra_onsets),
            
            # Tasas de error (porcentajes)
            'dtw_onsets_consistency_rate': f"{self.consistency_rate*100:.1f}%",
            'dtw_onsets_late_rate': f"{self.late_rate*100:.1f}%",
            'dtw_onsets_early_rate': f"{self.early_rate*100:.1f}%",
            'dtw_onsets_missing_rate': f"{self.missing_rate*100:.1f}%",
            'dtw_onsets_extra_rate': f"{self.extra_rate*100:.1f}%",
            
            # Estadísticas de ajustes temporales
            'dtw_onsets_mean_adjustment_ms': f"{self.mean_adjustment:.2f}",
            'dtw_onsets_std_adjustment_ms': f"{self.std_adjustment:.2f}",
            'dtw_onsets_max_adjustment_ms': f"{self.max_adjustment:.2f}",
            'dtw_onsets_min_adjustment_ms': f"{self.min_adjustment:.2f}",
            
            # Calidad del alineamiento
            'dtw_alignment_cost': f"{self.alignment_cost:.3f}",
            'dtw_analysis_tolerance_ms': f"{self.tolerance_ms:.1f}",        }
    
    def print_summary(self):
        """Imprime un resumen del análisis."""
        correct_count = len([m for m in self.matches if m.classification == OnsetType.CORRECT])
        late_count = len([m for m in self.matches if m.classification == OnsetType.LATE])
        early_count = len([m for m in self.matches if m.classification == OnsetType.EARLY])
        
        print("🎯 ANÁLISIS DTW DE ONSETS - RESUMEN")
        print("=" * 50)
        
        print(f"📊 Total onsets referencia: {self.total_ref_onsets}")
        print(f"📊 Total onsets en vivo: {self.total_live_onsets}")
        print(f"📊 Total emparejamientos: {self.total_matches}")
        
        print(f"\n✅ Onsets correctos (ritmo consistente): {correct_count}")
        print(f"🐌 Onsets tarde (desajuste positivo): {late_count}")
        print(f"⚡ Onsets adelantados (desajuste negativo): {early_count}")
        print(f"❌ Onsets faltantes: {len(self.missing_onsets)}")
        print(f"➕ Onsets extras: {len(self.extra_onsets)}")
        
        print(f"\n📈 Tasa de consistencia: {self.consistency_rate*100:.1f}%")
        print(f"📈 Tasa de tardanza: {self.late_rate*100:.1f}%")
        print(f"📈 Tasa de adelanto: {self.early_rate*100:.1f}%")
        print(f"📈 Tasa de faltantes: {self.missing_rate*100:.1f}%")
        print(f"📈 Tasa de extras: {self.extra_rate*100:.1f}%")
        
        print(f"\n⏱️ Ajuste temporal promedio: {self.mean_adjustment:.2f}ms")
        print(f"⏱️ Desviación estándar: {self.std_adjustment:.2f}ms")
        print(f"⏱️ Ajuste máximo: {self.max_adjustment:.2f}ms")
        print(f"⏱️ Ajuste mínimo: {self.min_adjustment:.2f}ms")
        
        print(f"\n🔧 Costo de alineamiento DTW: {self.alignment_cost:.3f}")
        print(f"🔧 Tolerancia utilizada: {self.tolerance_ms:.1f}ms")
    
    def get_detailed_matches(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Obtiene información detallada de todos los matches.
        
        Returns:
            Diccionario con listas de matches detallados por categoría
        """
        def match_to_dict(match: OnsetMatchClassified) -> Dict[str, Any]:
            return {
                'ref_onset': match.ref_onset,
                'live_onset': match.live_onset,
                'ref_pitch': match.ref_pitch,
                'live_pitch': match.live_pitch,
                'time_adjustment': match.time_adjustment,
                'pitch_similarity': match.pitch_similarity,
                'classification': match.classification.value
            }
        
        correct_matches = [match_to_dict(m) for m in self.matches if m.classification == OnsetType.CORRECT]
        late_matches = [match_to_dict(m) for m in self.matches if m.classification == OnsetType.LATE]
        early_matches = [match_to_dict(m) for m in self.matches if m.classification == OnsetType.EARLY]
        
        return {
            'correct': correct_matches,
            'late': late_matches,
            'early': early_matches,
            'missing': [{'onset': t, 'pitch': p} for t, p in self.missing_onsets],
            'extra': [{'onset': t, 'pitch': p} for t, p in self.extra_onsets]
        }
