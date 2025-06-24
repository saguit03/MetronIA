"""
Clases de categoría para mutaciones musicales.
"""

from dataclasses import dataclass, field
from typing import Dict, List

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
