"""
Paquete de analizadores musicales modularizados.

Este paquete contiene clases especializadas para análisis de audio musical:
- config: Configuración del análisis
- results: Estructuras de datos para resultados
- feature_extractor: Extracción de características de audio
- dtw_aligner: Alineamiento usando Dynamic Time Warping
- onset_dtw_analyzer: Análisis de onsets musicales con DTW
- tempo_analyzer: Análisis de tempo y estructura
- beat_spectrum_analyzer: Análisis de beat spectrum
- visualizer: Visualización de resultados
- result_visualizer: Visualización y exportación de resultados DTW
- music_analyzer: Analizador principal que integra todos los componentes
"""

from .beat_spectrum_analyzer import BeatSpectrumAnalyzer, BeatSpectrumResult
from .config import AudioAnalysisConfig
from .feature_extractor import AudioFeatureExtractor
from .music_analyzer import MusicAnalyzer
from .onset_dtw_analyzer import OnsetDTWAnalyzer
from .onset_results import OnsetDTWAnalysisResult, OnsetMatch
from .tempo_analyzer import TempoAnalyzer, TempoAnalysisResult
from .visualizer import Visualizer

__all__ = [
    'AudioAnalysisConfig',
    'BeatSpectrumResult', 
    'TempoAnalysisResult',
    'OnsetDTWAnalysisResult',
    'OnsetMatch',
    'AudioFeatureExtractor',
    'OnsetDTWAnalyzer',
    'TempoAnalyzer',
    'BeatSpectrumAnalyzer',
    'Visualizer',
    'MusicAnalyzer',
]
