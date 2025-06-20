"""
Analizador principal de interpretaciones musicales que integra todos los componentes.
"""

import numpy as np
import librosa
import pandas as pd
import soundfile as sf
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from pydub import AudioSegment
from pyrubberband import pyrb

from .config import AudioAnalysisConfig
from .results import BeatSpectrumResult, TempoAnalysisResult
from .onset_results import OnsetDTWAnalysisResult
from .dtw_aligner import DTWAligner
from .onset_dtw_analyzer import OnsetDTWAnalyzer
from .onset_utils import OnsetUtils
from .tempo_analyzer import TempoAnalyzer
from .beat_spectrum_analyzer import BeatSpectrumAnalyzer
from .visualizer import AudioVisualizer
from .result_visualizer import ResultVisualizer
from .feature_extractor import AudioFeatureExtractor
from utils.audio_utils import load_audio_files, stretch_audio, calculate_warping_path

class MusicAnalyzer:
    """Analizador principal de interpretaciones musicales."""
    
    def __init__(self, config: Optional[AudioAnalysisConfig] = None):
        self.config = config or AudioAnalysisConfig()
        self.dtw_aligner = DTWAligner(self.config)
        self.onset_dtw_analyzer = OnsetDTWAnalyzer(self.config)
        self.tempo_analyzer = TempoAnalyzer(self.config)
        self.beat_spectrum_analyzer = BeatSpectrumAnalyzer(self.config)
        self.visualizer = AudioVisualizer(self.config)
        self.result_visualizer = ResultVisualizer()
        self.feature_extractor = AudioFeatureExtractor(self.config)
        self.onset_utils = OnsetUtils()
 
    def comprehensive_analysis(self, reference_path: str, live_path: str, 
                             save_name: Optional[str] = None,
                             reference_tempo: Optional[float] = None,
                             save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Realiza un análisis completo de las interpretaciones.
        
        Args:
            reference_path: Ruta al audio de referencia
            live_path: Ruta al audio en vivo
            save_name: Nombre para guardar gráficos (opcional)
            reference_tempo: Tempo conocido del MIDI original (opcional, para análisis más robusto)            
            save_dir: Directorio específico donde guardar los resultados (opcional)
        """
        audio_ref, audio_live, sr = load_audio_files(reference_path, live_path)

        distance, wp, wp_s = calculate_warping_path(audio_ref, audio_live, sr,  self.config.hop_length)

        tempo_result = self.tempo_analyzer.analyze_tempo_with_reference(audio_ref, audio_live, sr, reference_tempo)

        if tempo_result.is_similar:
            aligned_audio_live = audio_live
        else:
            print(f"Ajustando tempo en vivo de {tempo_result.tempo_live:.1f} BPM a {tempo_result.tempo_ref:.1f} BPM")
            aligned_audio_live = stretch_audio(audio_ref, audio_live, wp_s, sr, self.config.hop_length, save_name="aligned_audio", save_dir=save_dir)

        beat_result = self.beat_spectrum_analyzer.beat_spectrum(audio_ref, aligned_audio_live, sr)

        dtw_analysis = self.dtw_aligner.evaluate_dtw_path_enhanced(wp, audio_ref, aligned_audio_live, sr)
        dtw_regular = dtw_analysis['is_regular_combined']
        
        dtw_onset_result = self.onset_dtw_analyzer.match_onsets_with_dtw(
            audio_ref, aligned_audio_live, sr, wp, distance
        )
        
        segment_result = self.tempo_analyzer.validate_segments(audio_ref, aligned_audio_live, sr)
        
        # Generar visualizaciones
        if save_name:
            # Usar save_dir si se proporciona, sino usar estructura por defecto
            if save_dir:
                analysis_dir = save_dir
            else:
                analysis_dir = f"results/{save_name}"
            
            self.visualizer.plot_beat_spectrum_comparison(result=beat_result, sr=sr, save_name="beat_spectrum", dir_path=analysis_dir, show=False)
            self.visualizer.plot_timeline_onset_errors_detailed(result=dtw_onset_result, save_name="timeline", dir_path=analysis_dir, show=False)
            self.result_visualizer.plot_onset_errors_detailed(dtw_onset_result=dtw_onset_result, save_name="onset_errors_detailed", dir_path=analysis_dir)
            self.onset_utils.save_onsets_analysis_to_csv(dtw_onset_result=dtw_onset_result, save_name="analysis", dir_path=analysis_dir)

        return {
            'beat_spectrum': beat_result,
            'dtw_onsets': dtw_onset_result,  # Análisis DTW principal
            'tempo': tempo_result,
            'segments': segment_result,
            'dtw_regular': dtw_regular,
            'dtw_analysis': dtw_analysis,  # Análisis completo del DTW
            'audio_ref':  audio_ref,  # Audio original
            'audio_live': audio_live,
            'sample_rate': sr,
        }
    
def analyze_performance(reference_path: str, live_path: str, save_name: Optional[str] = None,  config: Optional[AudioAnalysisConfig] = None, reference_tempo: Optional[float] = None, save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Función de conveniencia para realizar un análisis completo de interpretación.
    
    Args:
        reference_path: Ruta al archivo de audio de referencia
        live_path: Ruta al archivo de audio en vivo
        save_name: Nombre base para guardar gráficos (opcional)
        config: Configuración de análisis (opcional)
        reference_tempo: Tempo conocido del MIDI original (opcional, para análisis más robusto)
        save_dir: Directorio específico donde guardar los resultados (opcional)
    
    Returns:
        Diccionario con todos los resultados del análisis
    """
    if config is None:
        config = AudioAnalysisConfig()
    analyzer = MusicAnalyzer(config)
    return analyzer.comprehensive_analysis(reference_path, live_path, save_name, reference_tempo, save_dir)

