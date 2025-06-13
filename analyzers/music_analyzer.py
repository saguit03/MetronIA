"""
Analizador principal de interpretaciones musicales que integra todos los componentes.
"""

import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from IPython.display import Audio, display

from .config import AudioAnalysisConfig
from .results import BeatSpectrumResult, TempoAnalysisResult
from .dtw_results import OnsetDTWAnalysisResult
from .dtw_aligner import DTWAligner
from .onset_dtw_analyzer import OnsetDTWAnalyzer
from .tempo_analyzer import TempoAnalyzer
from .beat_spectrum_analyzer import BeatSpectrumAnalyzer
from .visualizer import AudioVisualizer
from .result_visualizer import ResultVisualizer


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
        self.visualizer = AudioVisualizer(self.config)
    
    def load_audio_files(self, reference_path: str, live_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """Carga archivos de audio."""
        reference_audio, sr = librosa.load(reference_path)
        live_audio, _ = librosa.load(live_path, sr=sr)  # Usar mismo sr        
        return reference_audio, live_audio, sr
    
    def save_onsets_analysis_to_csv(self, dtw_onset_result, save_name: str) -> None:
        """
        Guarda el análisis detallado de onsets en un archivo CSV.
        
        Args:
            dtw_onset_result: Resultado del análisis DTW de onsets
            save_name: Nombre base para el archivo
            reference_path: Ruta del archivo de referencia
            live_path: Ruta del archivo en vivo
        """
        # Crear directorio de resultados si no existe
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Nombre del archivo CSV
        csv_filename = results_dir / f"onsets_analysis_{save_name}.csv"
        
        # Crear lista de datos para el CSV
        csv_data = []
        
        # Procesar todos los matches (correctos, tarde, adelantado)
        for match in dtw_onset_result.matches:
            csv_data.append({
                'onset_type': match.classification.value,
                'ref_timestamp': round(match.ref_onset, 4),
                'live_timestamp': round(match.live_onset, 4),
                'time_difference_ms': round(match.time_adjustment, 2),
                'ref_pitch_hz': round(match.ref_pitch, 2) if match.ref_pitch > 0 else None,
                'live_pitch_hz': round(match.live_pitch, 2) if match.live_pitch > 0 else None,
                'pitch_similarity': round(match.pitch_similarity, 3),
            })
        
        # Procesar onsets perdidos (missing)
        for ref_time, ref_pitch in dtw_onset_result.missing_onsets:
            csv_data.append({
                'onset_type': 'missing',
                'ref_timestamp': round(ref_time, 4),
                'live_timestamp': None,
                'time_difference_ms': None,
                'ref_pitch_hz': round(ref_pitch, 2) if ref_pitch > 0 else None,
                'live_pitch_hz': None,
                'pitch_similarity': None,
            })
        
        # Procesar onsets extra
        for live_time, live_pitch in dtw_onset_result.extra_onsets:
            csv_data.append({
                'onset_type': 'extra',
                'ref_timestamp': None,
                'live_timestamp': round(live_time, 4),
                'time_difference_ms': None,
                'ref_pitch_hz': None,
                'live_pitch_hz': round(live_pitch, 2) if live_pitch > 0 else None,
                'pitch_similarity': None,
            })
        
        # Crear DataFrame y ordenar por timestamp de referencia
        df = pd.DataFrame(csv_data)
        
        # Ordenar por timestamp de referencia (poner None al final)
        df_sorted = df.sort_values(
            by=['ref_timestamp', 'live_timestamp'], 
            na_position='last'
        )
        
        # Guardar CSV
        df_sorted.to_csv(csv_filename, index=False, encoding='utf-8')
      
    def comprehensive_analysis(self, reference_path: str, live_path: str, 
                             save_name: Optional[str] = None,
                             reference_tempo: Optional[float] = None) -> Dict[str, Any]:
        """
        Realiza un análisis completo de las interpretaciones.
        
        Args:
            reference_path: Ruta al audio de referencia
            live_path: Ruta al audio en vivo
            save_name: Nombre para guardar gráficos (opcional)
            reference_tempo: Tempo conocido del MIDI original (opcional, para análisis más robusto)
        """       
         # Cargar audios
        audio_ref, audio_live, sr = self.load_audio_files(reference_path, live_path)
        
        # ========== ESTRATEGIA DE ALINEAMIENTO SEPARADO ==========
        # Usar el nuevo método que separa el análisis:
        # - Beat spectrum: CON alineamiento DTW (para comparar patrones rítmicos globales)
        # - Onsets: SIN alineamiento DTW (para detectar errores de timing reales)
        ref_feat, aligned_live_feat_for_beat, wp, unaligned_live_feat = \
            self.dtw_aligner.align_features_for_tempo_comparison(audio_ref, audio_live, sr)
        # Evaluación mejorada del DTW que incluye consistencia con onsets
        dtw_analysis = self.dtw_aligner.evaluate_dtw_path_enhanced(wp, audio_ref, audio_live, sr)
        dtw_regular = dtw_analysis['is_regular_combined']
        
        # Análisis de beat spectrum (CON alineamiento DTW)
        beat_result = self.beat_spectrum_analyzer.analyze_beat_spectrum(ref_feat, aligned_live_feat_for_beat)
        
        # Análisis de onsets usando DTW con consistencia de ritmo
        dtw_onset_result = self.onset_dtw_analyzer.analyze_onsets_with_rhythm_consistency(
            audio_ref, audio_live, sr, tolerance_ms=1.0
        )
        
        # Análisis de tempo usando método robusto si se proporciona tempo de referencia
        if reference_tempo is not None:
            tempo_result = self.tempo_analyzer.analyze_tempo_with_reference(
                audio_ref, audio_live, sr, reference_tempo
            )
        else:
            # Usar análisis robusto estándar
            tempo_result = self.tempo_analyzer.analyze_tempo_robust(audio_ref, audio_live, sr)
        
        segment_result = self.tempo_analyzer.validate_segments(audio_ref, audio_live, sr)
          # Generar visualizaciones
        if save_name:
            self.visualizer.plot_beat_spectrum_comparison(beat_result, sr, save_name)
            # Usar el nuevo visualizador para onsets DTW
            self.result_visualizer.plot_onset_errors_detailed(dtw_onset_result, save_name)
            
            # Generar CSV de análisis de onsets
            self.save_onsets_analysis_to_csv(dtw_onset_result, save_name)
        
        return {
            'beat_spectrum': beat_result,
            'dtw_onsets': dtw_onset_result,  # Análisis DTW principal
            'tempo': tempo_result,
            'segments': segment_result,
            'dtw_regular': dtw_regular,
            'dtw_analysis': dtw_analysis,  # Análisis completo del DTW
            'audio_ref': audio_ref,
            'audio_live': audio_live,
            'sample_rate': sr
        }
    
def analyze_performance(reference_path: str, live_path: str, save_name: Optional[str] = None, 
                       config: Optional[AudioAnalysisConfig] = None,
                       reference_tempo: Optional[float] = None) -> Dict[str, Any]:
    """
    Función de conveniencia para realizar un análisis completo de interpretación.
    
    Args:
        reference_path: Ruta al archivo de audio de referencia
        live_path: Ruta al archivo de audio en vivo
        save_name: Nombre base para guardar gráficos (opcional)
        config: Configuración de análisis (opcional)
        reference_tempo: Tempo conocido del MIDI original (opcional, para análisis más robusto)
    
    Returns:
        Diccionario con todos los resultados del análisis
    """
    if config is None:
        config = AudioAnalysisConfig()
    analyzer = MusicAnalyzer(config)
    return analyzer.comprehensive_analysis(reference_path, live_path, save_name, reference_tempo)

