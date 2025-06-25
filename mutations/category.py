from dataclasses import dataclass, field
from typing import Dict, List

from .results import MutationResult


@dataclass
class MutationCategory:
    name: str
    description: str
    mutations: Dict[str, MutationResult] = field(default_factory=dict)

    def add_mutation(self, mutation: MutationResult):
        self.mutations[mutation.name] = mutation

    def get_successful_mutations(self) -> List[MutationResult]:
        return [mut for mut in self.mutations.values() if mut.success]

    def __str__(self):
        successful = len(self.get_successful_mutations())
        total = len(self.mutations)
        return f"{self.name}: {successful}/{total} mutaciones exitosas"
