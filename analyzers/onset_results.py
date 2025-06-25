"""
Clases de resultados para el análisis DTW de onsets.
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, NamedTuple
from .config import TOLERANCE_MS

import numpy as np


class OnsetType(Enum):
    """Tipos de clasificación de onsets."""
    CORRECT = "correct"
    LATE = "late" 
    EARLY = "early"
    MISSING = "missing"
    EXTRA = "extra"


class OnsetMatch(NamedTuple):
    """Representa un emparejamiento entre onsets de referencia y en vivo."""
    ref_onset: float
    live_onset: float
    ref_pitch: float
    live_pitch: float
    time_adjustment: float
    pitch_similarity: float
    classification: OnsetType

@dataclass
class OnsetDTWAnalysisResult:
    """
    Resultado completo del análisis DTW de onsets con clasificación de errores.
    
    Esta clase maneja los resultados del análisis DTW de onsets, incluyendo:
    - Todos los onsets emparejados (correctos, tarde, adelantado)
    - Onsets faltantes y extras
    """
    
    matches: List[OnsetMatch]
    
    missing_onsets: List[tuple]  # (tiempo, pitch) de referencia
    extra_onsets: List[tuple]   # (tiempo, pitch) en vivo
    
    dtw_path: np.ndarray
    alignment_cost: float
    
    tolerance_ms: float = TOLERANCE_MS
    def __post_init__(self):
        """Calcula estadísticas después de la inicialización."""
        self._calculate_stats()
    
    @property
    def correct_matches(self) -> List[OnsetMatch]:
        """Obtiene solo los matches correctos."""
        return [OnsetMatch(m.ref_onset, m.live_onset, m.ref_pitch, m.live_pitch, m.time_adjustment, m.pitch_similarity, m.classification) 
                for m in self.matches if m.classification == OnsetType.CORRECT]
    
    @property
    def late_matches(self) -> List[OnsetMatch]:
        """Obtiene solo los matches tardíos."""
        return [OnsetMatch(m.ref_onset, m.live_onset, m.ref_pitch, m.live_pitch, m.time_adjustment, m.pitch_similarity,  m.classification) 
                for m in self.matches if m.classification == OnsetType.LATE]
    
    @property
    def early_matches(self) -> List[OnsetMatch]:
        """Obtiene solo los matches adelantados."""
        return [OnsetMatch(m.ref_onset, m.live_onset, m.ref_pitch, m.live_pitch, m.time_adjustment, m.pitch_similarity, m.classification) 
                for m in self.matches if m.classification == OnsetType.EARLY]
    
    def _calculate_stats(self):
        """Calcula estadísticas del análisis."""
        self.total_ref_onsets = len(self.matches) + len(self.missing_onsets)
        self.total_live_onsets = len(self.matches) + len(self.extra_onsets)
        self.total_matches = len(self.matches)
        
        correct_count = len([m for m in self.matches if m.classification == OnsetType.CORRECT])
        late_count = len([m for m in self.matches if m.classification == OnsetType.LATE])
        early_count = len([m for m in self.matches if m.classification == OnsetType.EARLY])
        
        self.consistency_rate = correct_count / self.total_ref_onsets if self.total_ref_onsets > 0 else 0.0
        self.late_rate = late_count / self.total_ref_onsets if self.total_ref_onsets > 0 else 0.0
        self.early_rate = early_count / self.total_ref_onsets if self.total_ref_onsets > 0 else 0.0
        self.missing_rate = len(self.missing_onsets) / self.total_ref_onsets if self.total_ref_onsets > 0 else 0.0
        self.extra_rate = len(self.extra_onsets) / self.total_live_onsets if self.total_live_onsets > 0 else 0.0
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
    
    def to_json_dict(self, mutation_category: str = "", mutation_name: str = "", 
                     reference_path: str = "", live_path: str = "", 
                     additional_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Convierte el resultado del análisis a un diccionario serializable en JSON.
        
        Args:
            mutation_category: Categoría de la mutación (opcional)
            mutation_name: Nombre de la mutación (opcional)
            reference_path: Ruta del archivo de referencia (opcional)
            live_path: Ruta del archivo en vivo (opcional)
            additional_metadata: Metadatos adicionales (opcional)
            
        Returns:
            Diccionario con todos los datos del análisis en formato JSON-serializable
        """
        matches_data = []
        for match in self.matches:
            matches_data.append({
                'ref_onset': float(match.ref_onset),
                'live_onset': float(match.live_onset),
                'ref_pitch': float(match.ref_pitch),
                'live_pitch': float(match.live_pitch),
                'time_adjustment': float(match.time_adjustment),
                'pitch_similarity': float(match.pitch_similarity),
                'classification': match.classification.value
            })
        
        dtw_path_list = []
        if self.dtw_path is not None:
            dtw_path_list = self.dtw_path.tolist()
        
        return {
            'metadata': {
                'analysis_type': 'OnsetDTWAnalysis',
                'mutation_category': mutation_category,
                'mutation_name': mutation_name,
                'reference_path': reference_path,
                'live_path': live_path,
                'tolerance_ms': float(self.tolerance_ms)
            },
            'statistics': {
                'total_ref_onsets': int(self.total_ref_onsets),
                'total_live_onsets': int(self.total_live_onsets),
                'total_matches': int(self.total_matches),
                'correct_matches': len([m for m in self.matches if m.classification == OnsetType.CORRECT]),
                'late_matches': len([m for m in self.matches if m.classification == OnsetType.LATE]),
                'early_matches': len([m for m in self.matches if m.classification == OnsetType.EARLY]),
                'missing_onsets': len(self.missing_onsets),
                'extra_onsets': len(self.extra_onsets),
                'consistency_rate': float(self.consistency_rate),
                'late_rate': float(self.late_rate),
                'early_rate': float(self.early_rate),
                'missing_rate': float(self.missing_rate),
                'extra_rate': float(self.extra_rate),
                'mean_adjustment_ms': float(self.mean_adjustment),
                'std_adjustment_ms': float(self.std_adjustment),
                'max_adjustment_ms': float(self.max_adjustment),
                'min_adjustment_ms': float(self.min_adjustment),
                'alignment_cost': float(self.alignment_cost)
            },
            'matches': matches_data,
            'missing_onsets': [{'onset': float(t), 'pitch': float(p)} for t, p in self.missing_onsets],
            'extra_onsets': [{'onset': float(t), 'pitch': float(p)} for t, p in self.extra_onsets],
            'dtw_path': dtw_path_list
        }

    def export_to_json(self, filepath: str, mutation_category: str = "", mutation_name: str = "", 
                       reference_path: str = "", live_path: str = "", indent: int = 2) -> None:
        """
        Exporta el resultado del análisis a un archivo JSON.
        
        Args:
            filepath: Ruta del archivo JSON donde guardar
            mutation_category: Categoría de la mutación (opcional)
            mutation_name: Nombre de la mutación (opcional)
            reference_path: Ruta del archivo de referencia (opcional)
            live_path: Ruta del archivo en vivo (opcional)
            indent: Indentación para el JSON (por defecto 2)
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        json_data = self.to_json_dict(mutation_category, mutation_name, reference_path, live_path)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=indent, ensure_ascii=False)
        
        print(f"📄 Análisis DTW exportado a JSON: {filepath}")

    @classmethod
    def from_json(cls, filepath: str) -> 'OnsetDTWAnalysisResult':
        """
        Carga un resultado de análisis desde un archivo JSON.
        
        Args:
            filepath: Ruta del archivo JSON
            
        Returns:
            Instancia de OnsetDTWAnalysisResult
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        matches = []
        for match_data in data['matches']:
            match = OnsetMatch(
                ref_onset=match_data['ref_onset'],
                live_onset=match_data['live_onset'],
                ref_pitch=match_data['ref_pitch'],
                live_pitch=match_data['live_pitch'],
                time_adjustment=match_data['time_adjustment'],
                pitch_similarity=match_data['pitch_similarity'],
                classification=OnsetType(match_data['classification'])
            )
            matches.append(match)
        
        missing_onsets = [(onset['onset'], onset['pitch']) for onset in data['missing_onsets']]
        extra_onsets = [(onset['onset'], onset['pitch']) for onset in data['extra_onsets']]
        dtw_path = np.array(data['dtw_path']) if data['dtw_path'] else np.array([])
        
        return cls(
            matches=matches,
            missing_onsets=missing_onsets,
            extra_onsets=extra_onsets,
            dtw_path=dtw_path,
            alignment_cost=data['statistics']['alignment_cost'],
            tolerance_ms=data['metadata']['tolerance_ms']
        )