"""
Analizador principal de interpretaciones musicales que integra todos los componentes.
"""

import numpy as np
import librosa
from typing import Optional, Dict, Any, Tuple
from IPython.display import Audio, display

from .config import AudioAnalysisConfig
from .results import BeatSpectrumResult, OnsetAnalysisResult, TempoAnalysisResult
from .dtw_aligner import DTWAligner
from .onset_analyzer import OnsetAnalyzer
from .tempo_analyzer import TempoAnalyzer
from .beat_spectrum_analyzer import BeatSpectrumAnalyzer
from .visualizer import AudioVisualizer


class MusicAnalyzer:
    """Analizador principal de interpretaciones musicales."""
    
    def __init__(self, config: Optional[AudioAnalysisConfig] = None):
        self.config = config or AudioAnalysisConfig()
        self.dtw_aligner = DTWAligner(self.config)
        self.onset_analyzer = OnsetAnalyzer(self.config)
        self.tempo_analyzer = TempoAnalyzer(self.config)
        self.beat_spectrum_analyzer = BeatSpectrumAnalyzer(self.config)
        self.visualizer = AudioVisualizer(self.config)
    
    def load_audio_files(self, reference_path: str, live_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """Carga archivos de audio."""
        reference_audio, sr = librosa.load(reference_path)
        live_audio, _ = librosa.load(live_path, sr=sr)  # Usar mismo sr
        return reference_audio, live_audio, sr
    
    def comprehensive_analysis(self, reference_path: str, live_path: str, 
                             save_name: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
        """Realiza un análisis completo de las interpretaciones."""
        # Cargar audios
        audio_ref, audio_live, sr = self.load_audio_files(reference_path, live_path)
        
        # Alineamiento DTW
        ref_feat, aligned_live_feat, wp = self.dtw_aligner.align_features(audio_ref, audio_live, sr)
        dtw_deviations, dtw_regular = self.dtw_aligner.evaluate_dtw_path(wp)
        
        # Análisis de beat spectrum
        beat_result = self.beat_spectrum_analyzer.analyze_beat_spectrum(ref_feat, aligned_live_feat)
        
        # Análisis de onsets
        onset_result = self.onset_analyzer.compare_onsets_detailed(audio_ref, audio_live, sr)
        rhythm_errors = self.onset_analyzer.detect_rhythm_pattern_errors(
            onset_result.onsets_ref, onset_result.onsets_live
        )
        
        # Análisis de tempo
        tempo_result = self.tempo_analyzer.analyze_tempo(audio_ref, audio_live, sr)
        segment_result = self.tempo_analyzer.validate_segments(audio_ref, audio_live, sr)
        
        # Generar visualizaciones
        if save_name:
            self.visualizer.plot_beat_spectrum_comparison(beat_result, sr, save_name)
            self.visualizer.plot_onset_errors_detailed(onset_result, save_name)
        
        # Imprimir resultados si es verbose
        if verbose:
            self._print_analysis_results(beat_result, onset_result, tempo_result, 
                                       segment_result, dtw_regular, rhythm_errors)
        
        return {
            'beat_spectrum': beat_result,
            'onsets': onset_result,
            'tempo': tempo_result,
            'segments': segment_result,
            'dtw_regular': dtw_regular,
            'dtw_deviations': dtw_deviations,
            'rhythm_errors': rhythm_errors,
            'audio_ref': audio_ref,
            'audio_live': audio_live,
            'sample_rate': sr
        }
    
    def _print_analysis_results(self, beat_result: BeatSpectrumResult, onset_result: OnsetAnalysisResult,
                               tempo_result: TempoAnalysisResult, segment_result: Dict,
                               dtw_regular: bool, rhythm_errors: Tuple):
        """Imprime resultados del análisis."""
        print("=" * 50)
        print("ANÁLISIS COMPLETO DE INTERPRETACIÓN MUSICAL")
        print("=" * 50)
        
        # Beat Spectrum
        status = "✅" if beat_result.is_similar else "⚠️"
        print(f"\n🎵 BEAT SPECTRUM:")
        print(f"  {status} Similitud: {'Similar' if beat_result.is_similar else 'Diferencias significativas'}")
        print(f"  📊 Diferencia máxima: {beat_result.max_difference:.3f}")
        
        # Onsets
        stats = onset_result.stats
        print(f"\n🎯 ANÁLISIS DE ONSETS:")
        print(f"  ✅ Onsets correctos: {stats['correct']}")
        print(f"  ⚡ Onsets adelantados: {stats['early']}")
        print(f"  🐌 Onsets atrasados: {stats['late']}")
        print(f"  ❌ Notas faltantes: {stats['missing']}")
        print(f"  ➕ Notas extras: {stats['extra']}")
        print(f"  📈 Precisión: {stats['correct']/(stats['total_ref'] or 1)*100:.1f}%")
        
        # Tempo
        print(f"\n🎼 ANÁLISIS DE TEMPO:")
        print(f"  🎵 Tempo referencia: {tempo_result.tempo_ref:.2f} BPM")
        print(f"  🎵 Tempo en vivo: {tempo_result.tempo_live:.2f} BPM")
        print(f"  📏 Diferencia: {tempo_result.difference:.2f} BPM")
        status = "✅" if tempo_result.is_similar else "⚠️"
        print(f"  {status} {'Tempo similar' if tempo_result.is_similar else 'Diferencia significativa de tempo'}")
        
        # Estructura
        print(f"\n🏗️ ESTRUCTURA MUSICAL:")
        print(f"  📏 Compases referencia: {segment_result['measures_ref']}")
        print(f"  📏 Compases en vivo: {segment_result['measures_live']}")
        status = "✅" if segment_result['overall_compatible'] else "⚠️"
        print(f"  {status} Estructura {'compatible' if segment_result['overall_compatible'] else 'incompatible'}")
        
        # DTW
        status = "✅" if dtw_regular else "⚠️"
        print(f"\n🔄 ALINEAMIENTO DTW:")
        print(f"  {status} Camino DTW {'regular' if dtw_regular else 'con desviaciones anómalas'}")
        
        # Errores rítmicos
        repeats, gaps = rhythm_errors
        print(f"\n🎶 PATRONES RÍTMICOS:")
        print(f"  🔁 Repeticiones detectadas: {len(repeats)}")
        print(f"  🕳️ Huecos grandes detectados: {len(gaps)}")


    def extract_analysis_for_csv(self, beat_result: BeatSpectrumResult, onset_result: OnsetAnalysisResult,
                                tempo_result: TempoAnalysisResult, segment_result: Dict,
                                dtw_regular: bool, rhythm_errors: Tuple) -> Dict[str, Any]:
        """
        Extrae los resultados del análisis en formato para CSV.
        
        Returns:
            Diccionario con los datos formateados para CSV
        """
        stats = onset_result.stats
        repeats, gaps = rhythm_errors
        
        return {
            # Beat Spectrum
            'beat_spectrum_similar': 'Similar' if beat_result.is_similar else 'Diferencias significativas',
            'beat_spectrum_max_difference': f"{beat_result.max_difference:.3f}",
            
            # Onsets
            'onsets_correct': stats['correct'],
            'onsets_early': stats['early'],
            'onsets_late': stats['late'],
            'onsets_missing': stats['missing'],
            'onsets_extra': stats['extra'],
            'onsets_precision': f"{stats['correct']/(stats['total_ref'] or 1)*100:.1f}%",
            
            # Tempo
            'tempo_reference_bpm': f"{tempo_result.tempo_ref:.2f}",
            'tempo_live_bpm': f"{tempo_result.tempo_live:.2f}",
            'tempo_difference_bpm': f"{tempo_result.difference:.2f}",
            'tempo_similar': 'Tempo similar' if tempo_result.is_similar else 'Diferencia significativa de tempo',
            
            # Estructura
            'structure_measures_ref': segment_result['measures_ref'],
            'structure_measures_live': segment_result['measures_live'],
            'structure_compatible': 'Estructura compatible' if segment_result['overall_compatible'] else 'Estructura incompatible',
            
            # DTW
            'dtw_regular': 'Camino DTW regular' if dtw_regular else 'Camino DTW con desviaciones anómalas',
            
            # Patrones rítmicos
            'rhythm_repeats': len(repeats),
            'rhythm_large_gaps': len(gaps)
        }


# Función de conveniencia para análisis rápido
def analyze_performance(reference_path: str, live_path: str, save_name: Optional[str] = None, 
                       config: Optional[AudioAnalysisConfig] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Función de conveniencia para realizar un análisis completo de interpretación.
    
    Args:
        reference_path: Ruta al archivo de audio de referencia
        live_path: Ruta al archivo de audio en vivo
        save_name: Nombre base para guardar gráficos (opcional)
        config: Configuración de análisis (opcional)
        verbose: Si mostrar resultados por pantalla (opcional)
    
    Returns:
        Diccionario con todos los resultados del análisis
    """
    analyzer = MusicAnalyzer(config)
    return analyzer.comprehensive_analysis(reference_path, live_path, save_name, verbose)


# Función compatible con el script original
def show_beat_spectrum(reference_path: str, live_path: str, 
                      comparacion_1: bool = True, comparacion_2: bool = True, 
                      nombre: Optional[str] = None):
    """
    Función compatible con la interfaz original de comparaciones.py
    """
    config = AudioAnalysisConfig()
    analyzer = MusicAnalyzer(config)
    
    # Cargar audios
    audio_ref, audio_live, sr = analyzer.load_audio_files(reference_path, live_path)
    
    # Análisis DTW
    ref_feat, aligned_live_feat, wp = analyzer.dtw_aligner.align_features(audio_ref, audio_live, sr)
    
    # Beat spectrum
    beat_result = analyzer.beat_spectrum_analyzer.analyze_beat_spectrum(ref_feat, aligned_live_feat)
    analyzer.visualizer.plot_beat_spectrum_comparison(beat_result, sr, nombre)
    
    if comparacion_1:
        print("======== Comparación de Beat Spectrums ========")
        status = "✅ Beat spectrum similar." if beat_result.is_similar else "⚠️ Diferencias significativas en el beat spectrum."
        print(status)
        
        # Onsets básicos
        onsets_data = analyzer.onset_analyzer.compare_onsets_basic(audio_ref, audio_live, sr)
        onsets_ref, onsets_live, matched, unmatched_ref, unmatched_live = onsets_data
        print(f"✅ Onsets emparejados: {len(matched)}")
        print(f"❌ Notas faltantes (en vivo): {len(unmatched_ref)}")
        print(f"❌ Notas extras (en vivo): {len(unmatched_live)}")
        
        # Tempo
        tempo_result = analyzer.tempo_analyzer.analyze_tempo(audio_ref, audio_live, sr)
        print(f"🎼 Tempo referencia: {tempo_result.tempo_ref:.2f} BPM")
        print(f"🎼 Tempo en vivo: {tempo_result.tempo_live:.2f} BPM")
        status = "✅ Tempo similar." if tempo_result.is_similar else "⚠️ Diferencia significativa de tempo."
        print(status)
        
        # DTW
        _, dtw_regular = analyzer.dtw_aligner.evaluate_dtw_path(wp)
        status = "✅ Camino DTW razonablemente regular." if dtw_regular else "⚠️ Camino DTW con desviaciones anómalas."
        print(status)
        
        # Segmentos
        segment_result = analyzer.tempo_analyzer.validate_segments(audio_ref, audio_live, sr)
        print(f"🎵 Compases en referencia: {segment_result['measures_ref']}")
        print(f"🎵 Compases en vivo: {segment_result['measures_live']}")
        if segment_result['overall_compatible']:
            print("✅ Estructura de compases compatible.")
        else:
            print("⚠️ Desajuste en la estructura de compases.")
    
    if comparacion_2:
        print("======== Comparación de onsets y errores rítmicos ========")
        onsets_data = analyzer.onset_analyzer.compare_onsets_basic(audio_ref, audio_live, sr)
        analyzer.visualizer.plot_onset_errors_basic(*onsets_data, save_name=nombre)
        
        print("======== Análisis detallado de onsets ========")
        onset_result = analyzer.onset_analyzer.compare_onsets_detailed(audio_ref, audio_live, sr)
        analyzer.visualizer.plot_onset_errors_detailed(onset_result, save_name=nombre)
        
        stats = onset_result.stats
        print(f"✅ Onsets correctos: {stats['correct']}")
        print(f"⚠️ Onsets adelantados: {stats['early']}")
        print(f"⚠️ Onsets atrasados: {stats['late']}")
        print(f"❌ Notas faltantes (en vivo): {stats['missing']}")
        print(f"❌ Notas extras (en vivo): {stats['extra']}")
        
        repeats, gaps = analyzer.onset_analyzer.detect_rhythm_pattern_errors(
            onset_result.onsets_ref, onset_result.onsets_live
        )
        print(f"Repeticiones detectadas en vivo (intervalos < 100 ms): {len(repeats)}")
        print(f"Huecos grandes detectados en vivo: {len(gaps)}")
    
    # Reproducir audios
    display(Audio(data=audio_ref, rate=sr))
    display(Audio(data=audio_live, rate=sr))
