"""
Analizador principal de interpretaciones musicales que integra todos los componentes.
"""

from typing import Optional, Dict, Any
import matplotlib.pyplot as plt

from utils.audio_utils import load_audio_files, stretch_audio, calculate_warping_path
from .beat_spectrum_analyzer import BeatSpectrumAnalyzer
from .config import AudioAnalysisConfig, VERBOSE_LOGGING
from .dtw_aligner import DTWAligner
from .feature_extractor import AudioFeatureExtractor
from .onset_dtw_analyzer import OnsetDTWAnalyzer
from .onset_utils import OnsetUtils
from .result_visualizer import ResultVisualizer
from .tempo_analyzer import TempoAnalyzer
from .visualizer import AudioVisualizer


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
                             save_dir: Optional[str] = None,
                             mutation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Realiza un análisis completo de las interpretaciones.
        
        Args:
            reference_path: Ruta al audio de referencia
            live_path: Ruta al audio en vivo
            save_name: Nombre para guardar gráficos (opcional)
            reference_tempo: Tempo conocido del MIDI original (opcional, para análisis más robusto)            
            save_dir: Directorio específico donde guardar los resultados (opcional)
            mutation_name: Nombre de la mutación para el archivo CSV (opcional)
        """
        audio_ref, audio_live, sr = load_audio_files(reference_path, live_path)

        distance, wp, wp_s = calculate_warping_path(audio_ref, audio_live, sr,  self.config.hop_length)

        tempo_result = self.tempo_analyzer.analyze_tempo_with_reference(audio_ref, audio_live, sr, reference_tempo)

        aligned_audio_live = stretch_audio(audio_ref, audio_live, wp_s, sr, self.config.hop_length, save_name="aligned_audio", save_dir=save_dir)

        distance, wp, wp_s = calculate_warping_path(audio_ref, aligned_audio_live, sr,  self.config.hop_length)

        beat_result = self.beat_spectrum_analyzer.beat_spectrum(audio_ref, aligned_audio_live, sr)

        dtw_onset_result = self.onset_dtw_analyzer.match_onsets_with_dtw(
            audio_ref, aligned_audio_live, sr, wp, distance
        )
        
        segment_result = self.tempo_analyzer.validate_segments(audio_ref, aligned_audio_live, sr)
        
        if save_name:
            if save_dir:
                analysis_dir = save_dir
            else:
                analysis_dir = f"results/{save_name}"
            
            self.visualizer.plot_beat_spectrum_comparison(result=beat_result, sr=sr, save_name="beat_spectrum", dir_path=analysis_dir)
            fig_timeline, ax = self.visualizer.plot_timeline_onset_errors_detailed(result=dtw_onset_result, save_name="timeline", dir_path=analysis_dir)
            self.result_visualizer.plot_onset_errors_detailed(dtw_onset_result=dtw_onset_result, save_name="onset_errors_detailed", dir_path=analysis_dir)
            OnsetUtils.save_onsets_analysis_to_csv(dtw_onset_result=dtw_onset_result, save_name="analysis", dir_path=analysis_dir, mutation_name=mutation_name)
              # Cerrar la figura para liberar memoria
            plt.close(fig_timeline)

        return {
            'beat_spectrum': beat_result,
            'dtw_onsets': dtw_onset_result,
            'tempo': tempo_result,
            'segments': segment_result,
            'audio_ref':  audio_ref,
            'audio_live': audio_live,
            'sample_rate': sr
        }
