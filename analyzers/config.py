"""
Configuración para el análisis de audio musical.
"""

from dataclasses import dataclass


ANALYSIS_PLOTS_PATH = "analysis_plots"

@dataclass
class AudioAnalysisConfig:
    """Configuración para el análisis de audio."""
    hop_length: int = 512
    n_mfcc: int = 20
    onset_margin: float = 0.05
    tempo_threshold: float = 5.0
    dtw_tolerance: float = 0.3
    compas_duration: float = 2.0
    beat_spectrum_threshold: float = 0.2
    plot_dpi: int = 300
