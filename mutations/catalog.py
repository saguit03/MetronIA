from typing import Dict, List, Optional

from mutations.controller import (
    pitch_shift_mutation,
    faster_tempo_mutation,
    a_lot_faster_tempo_mutation,
    slower_tempo_mutation,
    a_lot_slower_tempo_mutation,
    accelerando_tempo_mutation,
    ritardando_tempo_mutation,
    note_played_too_soon_mutation_controller,
    note_played_too_late_mutation,
    note_held_too_long_mutation,
    note_cut_too_soon_mutation,
    note_missing_mutation,
    note_not_expected_mutation,
    articulated_legato_mutation,
    articulated_staccato_mutation,
    articulated_accentuated_mutation,
    tempo_fluctuation_mutation,
)
from .category import MutationCategory
from .results import MutationResult


class MutationCatalog:

    def __init__(self):
        self.categories: Dict[str, MutationCategory] = {}
        self._initialize_mutations()

    def _initialize_mutations(self):
        pitch_category = MutationCategory(
            name="pitch_errors",
            description="Errores de altura de las notas"
        )
        pitch_category.add_mutation(MutationResult(
            name="pitch_shift",
            description="Cambio de altura de una nota",
            function=pitch_shift_mutation
        ))
        self.categories["pitch_errors"] = pitch_category

        tempo_category = MutationCategory(
            name="tempo_errors",
            description="Errores relacionados con el tempo"
        )

        tempo_mutations = [
            ("faster_tempo", "Tempo más rápido", faster_tempo_mutation),
            ("a_lot_faster_tempo", "Tempo mucho más rápido", a_lot_faster_tempo_mutation),
            ("slower_tempo", "Tempo más lento", slower_tempo_mutation),
            ("a_lot_slower_tempo", "Tempo mucho más lento", a_lot_slower_tempo_mutation)
        ]
        for name, desc, func in tempo_mutations:
            tempo_category.add_mutation(MutationResult(name=name, description=desc, function=func))
        self.categories["tempo_errors"] = tempo_category

        progressive_tempo_category = MutationCategory(
            name="progressive_tempo_errors",
            description="Errores de tempo progresivo"
        )

        progressive_tempo_mutations = [
            ("accelerando", "Accelerando - incremento gradual del tempo", accelerando_tempo_mutation),
            ("ritardando", "Ritardando - disminución gradual del tempo", ritardando_tempo_mutation),
            ("tempo_fluctuation", "Fluctuaciones aleatorias del tempo", tempo_fluctuation_mutation)
        ]
        for name, desc, func in progressive_tempo_mutations:
            progressive_tempo_category.add_mutation(MutationResult(name=name, description=desc, function=func))
        self.categories["progressive_tempo_errors"] = progressive_tempo_category

        timing_category = MutationCategory(
            name="timing_errors",
            description="Errores de timing de las notas"
        )
        timing_mutations = [
            ("note_too_soon", "Nota tocada demasiado pronto", note_played_too_soon_mutation_controller),
            ("note_too_late", "Nota tocada demasiado tarde", note_played_too_late_mutation),
        ]
        for name, desc, func in timing_mutations:
            timing_category.add_mutation(MutationResult(name=name, description=desc, function=func))
        self.categories["timing_errors"] = timing_category

        duration_category = MutationCategory(
            name="duration_errors",
            description="Errores de duración de las notas"
        )
        duration_mutations = [
            ("note_held_too_long", "Nota mantenida demasiado tiempo", note_held_too_long_mutation),
            ("note_cut_too_soon", "Nota cortada demasiado pronto", note_cut_too_soon_mutation),
        ]
        for name, desc, func in duration_mutations:
            duration_category.add_mutation(MutationResult(name=name, description=desc, function=func))
        self.categories["duration_errors"] = duration_category

        note_category = MutationCategory(
            name="note_errors",
            description="Errores de presencia de notas"
        )
        note_mutations = [
            ("note_missing", "Nota faltante", note_missing_mutation),
            ("note_not_expected", "Nota inesperada/extra", note_not_expected_mutation),
        ]
        for name, desc, func in note_mutations:
            note_category.add_mutation(MutationResult(name=name, description=desc, function=func))
        self.categories["note_errors"] = note_category

        articulation_category = MutationCategory(
            name="articulation_errors",
            description="Errores de articulación"
        )
        articulation_mutations = [
            ("articulated_legato", "Articulación legato", articulated_legato_mutation),
            ("articulated_staccato", "Articulación staccato", articulated_staccato_mutation),
            ("articulated_accentuated", "Articulación acentuada", articulated_accentuated_mutation),
        ]
        for name, desc, func in articulation_mutations:
            articulation_category.add_mutation(MutationResult(name=name, description=desc, function=func))
        self.categories["articulation_errors"] = articulation_category

    def get_mutation(self, category_name: str, mutation_name: str) -> Optional[MutationResult]:
        if category_name in self.categories:
            return self.categories[category_name].mutations.get(mutation_name)
        return None

    def get_all_mutations(self) -> List[MutationResult]:
        all_mutations = []
        for category in self.categories.values():
            all_mutations.extend(category.mutations.values())
        return all_mutations

