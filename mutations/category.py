"""
Clases de categoría para mutaciones musicales.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd

from .results import MutationResult


@dataclass
class MutationCategory:
    """Representa una categoría de mutaciones."""
    name: str
    description: str
    mutations: Dict[str, MutationResult] = field(default_factory=dict)
    
    def add_mutation(self, mutation: MutationResult):
        """Añade una mutación a la categoría."""
        self.mutations[mutation.name] = mutation
    def apply_all(self, original_excerpt: pd.DataFrame, tempo: int = 120) -> Dict[str, bool]:
        """
        Aplica todas las mutaciones de la categoría.
        
        Args:
            original_excerpt: DataFrame con el excerpt musical original
            tempo: Tempo en BPM del MIDI original
        
        Returns:
            Dict[str, bool]: Diccionario con el resultado de cada mutación.
        """
        results = {}
        for name, mutation in self.mutations.items():
            results[name] = mutation.apply(original_excerpt, tempo=tempo)
        return results
    
    def get_successful_mutations(self) -> List[MutationResult]:
        """Retorna solo las mutaciones que fueron exitosas."""
        return [mut for mut in self.mutations.values() if mut.success]
    
    def get_failed_mutations(self) -> List[MutationResult]:
        """Retorna solo las mutaciones que fallaron."""
        return [mut for mut in self.mutations.values() if not mut.success]
    
    def __str__(self):
        successful = len(self.get_successful_mutations())
        total = len(self.mutations)
        return f"{self.name}: {successful}/{total} mutaciones exitosas"
