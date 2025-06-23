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

from .beat_spectrum_analyzer import BeatSpectrumAnalyzer
from .config import AudioAnalysisConfig
from .dtw_aligner import DTWAligner
from .feature_extractor import AudioFeatureExtractor
from .music_analyzer import MusicAnalyzer
from .music_analyzer import analyze_performance
from .onset_dtw_analyzer import OnsetDTWAnalyzer
from .onset_results import OnsetDTWAnalysisResult, OnsetMatch
from .result_visualizer import ResultVisualizer
from .results import BeatSpectrumResult, TempoAnalysisResult
from .tempo_analyzer import TempoAnalyzer
from .visualizer import AudioVisualizer

__all__ = [
    'AudioAnalysisConfig',
    'BeatSpectrumResult', 
    'TempoAnalysisResult',
    'OnsetDTWAnalysisResult',
    'OnsetMatch',
    'AudioFeatureExtractor',
    'DTWAligner',
    'OnsetDTWAnalyzer',
    'TempoAnalyzer',
    'BeatSpectrumAnalyzer',
    'AudioVisualizer',
    'ResultVisualizer',
    'MusicAnalyzer',
    'analyze_performance'
]
