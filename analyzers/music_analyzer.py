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
from utils.audio_utils import load_audio_files, stretch_audio

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
                             save_name: Optional[str] = None, verbose: bool = True,
                             reference_tempo: Optional[float] = 120,
                             save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Realiza un análisis completo de las interpretaciones.
        
        Args:
            reference_path: Ruta al audio de referencia
            live_path: Ruta al audio en vivo
            save_name: Nombre para guardar gráficos (opcional)
            verbose: Si mostrar resultados detallados por pantalla
            reference_tempo: Tempo conocido del MIDI original (opcional, para análisis más robusto)            
            save_dir: Directorio específico donde guardar los resultados (opcional)
        """
        # Cargar audios
        audio_ref, audio_live, sr = load_audio_files(reference_path, live_path)
        aligned_audio_live, wp, wp_s = stretch_audio(audio_ref, audio_live, sr, self.config.hop_length, save_name="aligned_audio_live", save_dir=save_dir)

        beat_result = self.beat_spectrum_analyzer.analyze_beat_spectrum(audio_ref, aligned_audio_live, sr)

        dtw_analysis = self.dtw_aligner.evaluate_dtw_path_enhanced(wp, audio_ref, aligned_audio_live, sr)
        dtw_regular = dtw_analysis['is_regular_combined']
        
        # audio_ref_resampled para mejorar el alineamiento DTW
        dtw_onset_result = self.onset_dtw_analyzer.analyze_onsets_with_rhythm_consistency(
            audio_ref, aligned_audio_live, sr
        )
        # Análisis de tempo usando método robusto si se proporciona tempo de referencia
        tempo_result = self.tempo_analyzer.analyze_tempo_with_reference(
            audio_ref, aligned_audio_live, sr, reference_tempo
        )

        # Actualizar resultado con información de proporción de tempo        
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
            self.onset_utils.save_onsets_analysis_to_csv(dtw_onset_result=dtw_onset_result, save_name="", dir_path=analysis_dir)

        if verbose:
            self._print_analysis_results(beat_result, tempo_result, 
                                       segment_result, dtw_analysis, dtw_onset_result)
        
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
    
    def _print_analysis_results(self, beat_result, tempo_result, segment_result, 
                              dtw_analysis, dtw_onset_result):
        """
        Imprime un resumen de los resultados del análisis.
        
        Args:
            beat_result: Resultado del análisis de beat spectrum
            tempo_result: Resultado del análisis de tempo
            segment_result: Resultado del análisis de segmentos
            dtw_analysis: Resultado del análisis DTW
            dtw_onset_result: Resultado del análisis DTW de onsets
        """
        print(f"\n📊 RESUMEN DEL ANÁLISIS")
        print("=" * 50)
        
        # Resumen de onsets
        if dtw_onset_result:
            total_matches = len(dtw_onset_result.matches)
            correct_matches = len([m for m in dtw_onset_result.matches 
                                 if m.classification.value == 'correct'])
            late_matches = len([m for m in dtw_onset_result.matches 
                              if m.classification.value == 'late'])
            early_matches = len([m for m in dtw_onset_result.matches 
                               if m.classification.value == 'early'])
            missing_onsets = len(dtw_onset_result.missing_onsets)
            extra_onsets = len(dtw_onset_result.extra_onsets)
            
            print(f"🎯 Análisis de Onsets:")
            print(f"   ✅ Correctos: {correct_matches}")
            print(f"   ⏰ Tarde: {late_matches}")
            print(f"   ⚡ Adelantados: {early_matches}")
            print(f"   ❌ Perdidos: {missing_onsets}")
            print(f"   ➕ Extra: {extra_onsets}")
            print(f"   📈 Total emparejados: {total_matches}")
            
            if total_matches > 0:
                accuracy = (correct_matches / total_matches) * 100
                print(f"   🎯 Precisión: {accuracy:.1f}%")
          # Resumen de tempo
        if tempo_result:
            print(f"\n🎵 Análisis de Tempo:")
            print(f"   📄 Referencia: {tempo_result.tempo_ref:.1f} BPM")
            print(f"   🎤 En vivo: {tempo_result.tempo_live:.1f} BPM")
            print(f"   📊 Diferencia: {tempo_result.difference:.1f} BPM")
            print(f"   ✅ Similar: {'Sí' if tempo_result.is_similar else 'No'}")
            
            # Mostrar información de proporción de tempo si está disponible
            if hasattr(tempo_result, 'tempo_proportion'):
                print(f"   🔄 Proporción (live/ref): {tempo_result.tempo_proportion:.3f}")
                if hasattr(tempo_result, 'resampling_applied') and tempo_result.resampling_applied:
                    print(f"   ⚡ Re-sampling aplicado: Sí")
                    if hasattr(tempo_result, 'original_ref_tempo'):
                        print(f"   📄 Tempo original ref: {tempo_result.original_ref_tempo:.1f} BPM")
                    if hasattr(tempo_result, 'original_live_tempo'):
                        print(f"   🎤 Tempo original live: {tempo_result.original_live_tempo:.1f} BPM")
                else:
                    print(f"   ✅ Re-sampling aplicado: No (proporción en rango [0.95, 1.05])")
        
        # Resumen de beat spectrum
        if beat_result:
            print(f"\n🎼 Análisis de Beat Spectrum:")
            print(f"   📊 Diferencia máxima: {beat_result.max_difference:.3f}")
            print(f"   ✅ Similar: {'Sí' if beat_result.is_similar else 'No'}")
          # Resumen DTW
        if dtw_analysis:
            print(f"\n🔀 Análisis DTW:")
            print(f"   📊 Evaluación: {dtw_analysis.get('overall_assessment', 'N/A')}")
            print(f"   ✅ Regular: {'Sí' if dtw_analysis.get('is_regular_combined', False) else 'No'}")

def analyze_performance(reference_path: str, live_path: str, save_name: Optional[str] = None, 
                       config: Optional[AudioAnalysisConfig] = None, verbose: bool = True,
                       reference_tempo: Optional[float] = None, save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Función de conveniencia para realizar un análisis completo de interpretación.
    
    Args:
        reference_path: Ruta al archivo de audio de referencia
        live_path: Ruta al archivo de audio en vivo
        save_name: Nombre base para guardar gráficos (opcional)
        config: Configuración de análisis (opcional)
        verbose: Si mostrar resultados detallados por pantalla
        reference_tempo: Tempo conocido del MIDI original (opcional, para análisis más robusto)
        save_dir: Directorio específico donde guardar los resultados (opcional)
    
    Returns:
        Diccionario con todos los resultados del análisis
    """
    if config is None:
        config = AudioAnalysisConfig()
    analyzer = MusicAnalyzer(config)
    return analyzer.comprehensive_analysis(reference_path, live_path, save_name, verbose, reference_tempo, save_dir)

