"""
Clases de resultado para mutaciones musicales.
"""

from dataclasses import dataclass
from typing import Optional, Callable
import pandas as pd
import re


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
    
    def apply(self, original_excerpt: pd.DataFrame, tempo: int = 120) -> bool:
        """
        Aplica la mutación al excerpt original.
        
        Args:
            original_excerpt: DataFrame con el excerpt musical original
            tempo: Tempo en BPM del MIDI original (usado para mutaciones que requieren tempo)
        
        Returns:
            bool: True si la mutación fue exitosa, False en caso contrario.
        """
        try:
            # Verificar si la función acepta el parámetro tempo
            import inspect
            sig = inspect.signature(self.function)
            
            if 'tempo' in sig.parameters:
                # Función acepta tempo como parámetro
                self.excerpt = self.function(original_excerpt, tempo=tempo)
            else:
                # Función no acepta tempo
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
    
    def get_mutation_tempo(self, base_tempo: int = 120) -> int:
        """
        Calcula el tempo correspondiente a la mutación aplicada.
        
        Args:
            base_tempo: Tempo base del MIDI original (BPM)
            
        Returns:
            int: Tempo calculado para la mutación en BPM
        """
        if not self.success or self.excerpt is None:
            return base_tempo
        
        # Detectar tipo de mutación de tempo
        mutation_type = self._detect_tempo_mutation_type()
        
        if mutation_type is None:
            return base_tempo
        
        return self._calculate_tempo_for_mutation(base_tempo, mutation_type)
    
    def _detect_tempo_mutation_type(self) -> Optional[str]:
        """
        Detecta el tipo de mutación de tempo basado en el nombre.
        
        Returns:
            str: Tipo de mutación o None si no es una mutación de tempo
        """
        tempo_patterns = {
            'faster_tempo': 'faster',
            'a_lot_faster_tempo': 'a_lot_faster',
            'slower_tempo': 'slower', 
            'a_lot_slower_tempo': 'a_lot_slower',
            'accelerando': 'accelerando',
            'ritardando': 'ritardando',
            'tempo_fluctuation': 'fluctuation'
        }
        
        for pattern, mutation_type in tempo_patterns.items():
            if pattern in self.name:
                return mutation_type
        
        return None
    
    def _calculate_tempo_for_mutation(self, base_tempo: int, mutation_type: str) -> int:
        """
        Calcula el tempo específico para cada tipo de mutación.
        
        Args:
            base_tempo: Tempo base en BPM
            mutation_type: Tipo de mutación detectada
            
        Returns:
            int: Tempo calculado en BPM
        """
        from mutations.config import (
            FASTER, A_LOT_FASTER, SLOWER, A_LOT_SLOWER, 
            ACCELERANDO, RITARDANDO
        )
        
        tempo_factors = {
            'faster': FASTER,           # 1.2
            'a_lot_faster': A_LOT_FASTER,   # 1.5
            'slower': SLOWER,           # 0.8
            'a_lot_slower': A_LOT_SLOWER,   # 0.5
            'accelerando': ACCELERANDO,     # 1.3 (tempo final)
            'ritardando': RITARDANDO,       # 0.7 (tempo final)
            'fluctuation': 1.0          # Tempo promedio (sin cambio base)
        }
        
        factor = tempo_factors.get(mutation_type, 1.0)
        
        # Para accelerando y ritardando, usar el tempo final
        if mutation_type in ['accelerando', 'ritardando']:
            calculated_tempo = int(base_tempo * factor)
        else:
            # Para cambios constantes de tempo
            calculated_tempo = int(base_tempo * factor)
        
        # Asegurar que el tempo esté en un rango razonable (40-200 BPM)
        return max(40, min(200, calculated_tempo))
    
    def is_tempo_mutation(self) -> bool:
        """
        Verifica si la mutación afecta el tempo.
        
        Returns:
            bool: True si es una mutación de tempo
        """
        return self._detect_tempo_mutation_type() is not None
    
    def __str__(self):
        status = "✓" if self.success else "✗"
        return f"{status} {self.name}: {self.description}"
