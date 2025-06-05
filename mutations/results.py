"""
Clases de resultado para mutaciones musicales.
"""

from dataclasses import dataclass
from typing import Optional, Callable
import pandas as pd


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
