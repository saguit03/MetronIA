"""
Configuración para el análisis de audio musical.
"""

from dataclasses import dataclass

VERBOSE_LOGGING = False
TOLERANCE_MS = 0.1

@dataclass
class AudioAnalysisConfig:
    """Configuración para el análisis de audio."""
    hop_length: int = 1024
    n_chroma: int = 12
    n_mfcc: int = 20
    tempo_threshold: float = 5.0
    compas_duration: float = 2.0
    beat_spectrum_threshold: float = 0.2
    plot_dpi: int = 300
    tolerance_ms: float = TOLERANCE_MS
    round_decimals: int = 1
