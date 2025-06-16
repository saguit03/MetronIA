"""
Configuración para el análisis de audio musical.
"""

from dataclasses import dataclass


@dataclass
class AudioAnalysisConfig:
    """Configuración para el análisis de audio."""
    hop_length: int = 512
    n_mfcc: int = 20
    onset_margin: float = 0.005
    tempo_threshold: float = 5.0
    dtw_tolerance: float = 0.03
    compas_duration: float = 2.0
    beat_spectrum_threshold: float = 0.2
    plot_dpi: int = 300
