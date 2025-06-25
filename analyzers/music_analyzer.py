"""
Analizador principal de interpretaciones musicales que integra todos los componentes.
"""

from typing import Optional, Dict, Any
import matplotlib.pyplot as plt

from utils.audio_utils import load_audio_files, stretch_audio, calculate_warping_path
from .beat_spectrum_analyzer import BeatSpectrumAnalyzer
from .config import AudioAnalysisConfig
from .onset_dtw_analyzer import OnsetDTWAnalyzer
from .onset_utils import OnsetUtils
from .tempo_analyzer import TempoAnalyzer
from .visualizer import Visualizer
import librosa


class MusicAnalyzer:
    """Analizador principal de interpretaciones musicales."""
    
    def __init__(self, config: Optional[AudioAnalysisConfig] = None):
        self.config = config or AudioAnalysisConfig()
        self.onset_dtw_analyzer = OnsetDTWAnalyzer(self.config)
        self.tempo_analyzer = TempoAnalyzer(self.config)
        self.beat_spectrum_analyzer = BeatSpectrumAnalyzer(self.config)
        self.visualizer = Visualizer(self.config)

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
        
        # 1. Carga de audios
        audio_ref, audio_live, sr = load_audio_files(reference_path, live_path)
        
        # 2. Eliminación de silencios al inicio y al final de los audios
        trimmed_reference_audio, (reference_start_index, reference_end_index) = librosa.effects.trim(audio_ref)
        trimmed_live_audio, (live_start_index, live_end_index) = librosa.effects.trim(audio_live)

        # 3. Alineamiento de audios
        aligned_audio_live = stretch_audio(trimmed_reference_audio, trimmed_live_audio, sr, self.config.hop_length, save_name="aligned_audio", save_dir=save_dir)
        
        # 4. Cálculo de DTW con el audio en vivo alineado
        distance, wp, wp_s = calculate_warping_path(trimmed_reference_audio, aligned_audio_live, sr,  self.config.hop_length)

        # 5. ANÁLISIS Y OBTENCIÓN DE RESULTADOS       
        # 5.1 Análisis de tempi del audio de referencia y del audio en vivo (sin alinear para obtener el tempo original)
        tempo_result = self.tempo_analyzer.analyze_tempo_with_reference(trimmed_reference_audio, trimmed_live_audio, sr, reference_tempo)
        # 5.2 Obtención del beat spectrum
        beat_result = self.beat_spectrum_analyzer.beat_spectrum(trimmed_reference_audio, aligned_audio_live, sr)
        # 5.3 Detección y alineamiento de onsets con DTW        
        dtw_onset_result = self.onset_dtw_analyzer.match_onsets_with_dtw(
            trimmed_reference_audio, aligned_audio_live, sr, wp, distance, tempo_result.tempo_ref, tempo_result.tempo_live
        )
        # 5.4 Análisis de la estructura de los audios (por compases)
        segment_result = self.tempo_analyzer.validate_segments(trimmed_reference_audio, aligned_audio_live, sr)
        
        # 6. Almacenamiento de los resultados de análisis
        if save_name:
            if save_dir:
                analysis_dir = save_dir
            else:
                analysis_dir = f"results/{save_name}"
            # 6.1 Gráfica de beat spectrum
            self.visualizer.plot_beat_spectrum_comparison(result=beat_result, sr=sr, save_name="beat_spectrum", dir_path=analysis_dir)
            # 6.2 Línea temporal de onsets
            fig_timeline, ax = self.visualizer.plot_onsets(result=dtw_onset_result, save_name="onset_timeline", dir_path=analysis_dir)
            # 6.3 Distribución de errores de onsets en todo el audio
            self.visualizer.plot_onset_distribution(dtw_onset_result=dtw_onset_result, save_name="onset_distribution", dir_path=analysis_dir)
            # 6.4 Análisis completo de onsets
            OnsetUtils.save_onsets_analysis_to_csv(dtw_onset_result=dtw_onset_result, save_name="onset_analysis", dir_path=analysis_dir, mutation_name=mutation_name)
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
