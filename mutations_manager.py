"""
Gestión de mutaciones musicales usando clases para mejor organización y legibilidad.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
from pathlib import Path

from mutaciones import (
    pitch_shift_mutation,
    faster_tempo_mutation,
    a_lot_faster_tempo_mutation,
    slower_tempo_mutation,
    a_lot_slower_tempo_mutation,
    accelerando_tempo_mutation,
    ritardando_tempo_mutation,
    note_played_too_soon_time_mutation,
    note_played_too_late_time_mutation,
    note_played_too_soon_onset_mutation,
    note_played_too_late_onset_mutation,
    note_held_too_long_mutation,
    note_cut_too_soon_mutation,
    note_missing_mutation,
    note_not_expected_mutation,
    articulated_legato_mutation,
    articulated_staccato_mutation,
    articulated_accentuated_mutation,
    tempo_fluctuation_mutation,
)


@dataclass
class MutationResult:
    """Representa el resultado de aplicar una mutación."""
    name: str
    description: str
    function: Callable
    excerpt: Optional[pd.DataFrame] = None
    path: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    
    def apply(self, original_excerpt: pd.DataFrame) -> bool:
        """
        Aplica la mutación al excerpt original.
        
        Returns:
            bool: True si la mutación fue exitosa, False en caso contrario.
        """
        try:
            self.excerpt = self.function(original_excerpt)
            if self.excerpt is not None:
                self.success = True
                return True
            else:
                self.success = False
                self.error = "Mutation returned None"
                return False
        except Exception as e:
            self.success = False
            self.error = str(e)
            self.excerpt = None
            return False
    
    def set_path(self, path: str):
        """Establece la ruta del archivo de audio generado."""
        self.path = path
    
    def __str__(self):
        status = "✓" if self.success else "✗"
        return f"{status} {self.name}: {self.description}"


@dataclass
class MutationCategory:
    """Representa una categoría de mutaciones."""
    name: str
    description: str
    mutations: Dict[str, MutationResult] = field(default_factory=dict)
    
    def add_mutation(self, mutation: MutationResult):
        """Añade una mutación a la categoría."""
        self.mutations[mutation.name] = mutation
    
    def apply_all(self, original_excerpt: pd.DataFrame) -> Dict[str, bool]:
        """
        Aplica todas las mutaciones de la categoría.
        
        Returns:
            Dict[str, bool]: Diccionario con el resultado de cada mutación.
        """
        results = {}
        for name, mutation in self.mutations.items():
            results[name] = mutation.apply(original_excerpt)
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


class MutationManager:
    """Gestor principal de todas las mutaciones musicales."""
    
    def __init__(self):
        self.categories: Dict[str, MutationCategory] = {}
        self._initialize_mutations()
    
    def _initialize_mutations(self):
        """Inicializa todas las categorías y mutaciones."""
        
        # Errores de altura
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
        
        # Errores de tempo
        tempo_category = MutationCategory(
            name="tempo_errors", 
            description="Errores relacionados con el tempo"
        )
        tempo_mutations = [
            ("faster_tempo", "Tempo más rápido", faster_tempo_mutation),
            ("a_lot_faster_tempo", "Tempo mucho más rápido", a_lot_faster_tempo_mutation),
            ("slower_tempo", "Tempo más lento", slower_tempo_mutation),
            ("a_lot_slower_tempo", "Tempo mucho más lento", a_lot_slower_tempo_mutation),
            ("accelerando", "Accelerando - incremento gradual del tempo", accelerando_tempo_mutation),
            ("ritardando", "Ritardando - disminución gradual del tempo", ritardando_tempo_mutation),
            ("tempo_fluctuation", "Fluctuaciones aleatorias del tempo", tempo_fluctuation_mutation),
        ]
        for name, desc, func in tempo_mutations:
            tempo_category.add_mutation(MutationResult(name=name, description=desc, function=func))
        self.categories["tempo_errors"] = tempo_category
        
        # Errores de timing
        timing_category = MutationCategory(
            name="timing_errors",
            description="Errores de timing de las notas"
        )
        timing_mutations = [
            ("note_too_soon_time", "Nota tocada demasiado pronto (tiempo)", note_played_too_soon_time_mutation),
            ("note_too_late_time", "Nota tocada demasiado tarde (tiempo)", note_played_too_late_time_mutation),
            ("note_too_soon_onset", "Nota tocada demasiado pronto (onset)", note_played_too_soon_onset_mutation),
            ("note_too_late_onset", "Nota tocada demasiado tarde (onset)", note_played_too_late_onset_mutation),
        ]
        for name, desc, func in timing_mutations:
            timing_category.add_mutation(MutationResult(name=name, description=desc, function=func))
        self.categories["timing_errors"] = timing_category
        
        # Errores de duración
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
        
        # Errores de notas
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
        
        # Errores de articulación
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
    
    def apply_all_mutations(self, original_excerpt: pd.DataFrame) -> Dict[str, Dict[str, bool]]:
        """
        Aplica todas las mutaciones a un excerpt original.
        
        Returns:
            Dict con los resultados organizados por categoría.
        """
        results = {}
        for category_name, category in self.categories.items():
            results[category_name] = category.apply_all(original_excerpt)
        return results
    
    def get_mutation(self, category_name: str, mutation_name: str) -> Optional[MutationResult]:
        """Obtiene una mutación específica."""
        if category_name in self.categories:
            return self.categories[category_name].mutations.get(mutation_name)
        return None
    
    def get_all_mutations(self) -> List[MutationResult]:
        """Retorna todas las mutaciones en una lista plana."""
        all_mutations = []
        for category in self.categories.values():
            all_mutations.extend(category.mutations.values())
        return all_mutations
    
    def get_successful_mutations(self) -> List[MutationResult]:
        """Retorna todas las mutaciones exitosas."""
        return [mut for mut in self.get_all_mutations() if mut.success]
    
    def get_failed_mutations(self) -> List[MutationResult]:
        """Retorna todas las mutaciones que fallaron."""
        return [mut for mut in self.get_all_mutations() if not mut.success]
    
    def print_summary(self):
        """Imprime un resumen de todas las mutaciones."""
        print("=== RESUMEN DE MUTACIONES ===")
        for category in self.categories.values():
            print(f"\n{category}")
            for mutation in category.mutations.values():
                print(f"  {mutation}")
        
        total_mutations = len(self.get_all_mutations())
        successful_mutations = len(self.get_successful_mutations())
        print(f"\nTOTAL: {successful_mutations}/{total_mutations} mutaciones exitosas")
    
    def export_paths(self) -> Dict[str, str]:
        """Exporta todas las rutas de archivos generados."""
        paths = {}
        for category_name, category in self.categories.items():
            for mutation_name, mutation in category.mutations.items():
                if mutation.path:
                    paths[f"{category_name}.{mutation_name}"] = mutation.path
        return paths
