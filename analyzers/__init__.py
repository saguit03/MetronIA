"""
Paquete de analizadores musicales modularizados.

Este paquete contiene clases especializadas para análisis de audio musical:
- config: Configuración del análisis
- results: Estructuras de datos para resultados
- feature_extractor: Extracción de características de audio
- dtw_aligner: Alineamiento usando Dynamic Time Warping
- onset_analyzer: Análisis de onsets musicales
- tempo_analyzer: Análisis de tempo y estructura
- beat_spectrum_analyzer: Análisis de beat spectrum
- visualizer: Visualización de resultados
- music_analyzer: Analizador principal que integra todos los componentes
"""

from .config import AudioAnalysisConfig
from .results import OnsetAnalysisResult, BeatSpectrumResult, TempoAnalysisResult
from .feature_extractor import AudioFeatureExtractor
from .dtw_aligner import DTWAligner
from .onset_analyzer import OnsetAnalyzer
from .tempo_analyzer import TempoAnalyzer
from .beat_spectrum_analyzer import BeatSpectrumAnalyzer
from .visualizer import AudioVisualizer
from .music_analyzer import MusicAnalyzer

# Funciones de conveniencia
from .music_analyzer import analyze_performance, show_beat_spectrum

__all__ = [
    'AudioAnalysisConfig',
    'OnsetAnalysisResult', 
    'BeatSpectrumResult', 
    'TempoAnalysisResult',
    'AudioFeatureExtractor',
    'DTWAligner',
    'OnsetAnalyzer',
    'TempoAnalyzer',
    'BeatSpectrumAnalyzer',
    'AudioVisualizer',
    'MusicAnalyzer',
    'analyze_performance',
    'show_beat_spectrum'
]
