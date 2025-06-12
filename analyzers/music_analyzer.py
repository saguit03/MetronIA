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
                             save_name: Optional[str] = None, verbose: bool = True,
                             reference_tempo: Optional[float] = None) -> Dict[str, Any]:
        """
        Realiza un análisis completo de las interpretaciones.
        
        Args:
            reference_path: Ruta al audio de referencia
            live_path: Ruta al audio en vivo
            save_name: Nombre para guardar gráficos (opcional)
            verbose: Si mostrar resultados por pantalla
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
        
        # Análisis de onsets (SIN alineamiento DTW para preservar errores de timing reales)
        onset_result = self.onset_analyzer.compare_onsets_without_alignment(
            audio_ref, audio_live, sr, reference_tempo
        )
        rhythm_errors = self.onset_analyzer.detect_rhythm_pattern_errors(
            onset_result.onsets_ref, onset_result.onsets_live
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
            self.visualizer.plot_onset_errors_detailed(onset_result, save_name)
          # Imprimir resultados si es verbose
        if verbose:
            self._print_analysis_results(beat_result, onset_result, tempo_result, 
                                       segment_result, dtw_analysis, rhythm_errors)
        
        return {
            'beat_spectrum': beat_result,
            'onsets': onset_result,
            'tempo': tempo_result,
            'segments': segment_result,
            'dtw_regular': dtw_regular,
            'dtw_analysis': dtw_analysis,  # Análisis completo del DTW
            'rhythm_errors': rhythm_errors,
            'audio_ref': audio_ref,
            'audio_live': audio_live,
            'sample_rate': sr
        }
    def _print_analysis_results(self, beat_result: BeatSpectrumResult, onset_result: OnsetAnalysisResult,
                               tempo_result: TempoAnalysisResult, segment_result: Dict,
                               dtw_analysis: Dict, rhythm_errors: Tuple):
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
          # DTW Análisis Mejorado
        dtw_regular = dtw_analysis['is_regular_combined']
        print(f"\n🔄 ALINEAMIENTO DTW:")
        print(f"  📊 Evaluación tradicional: {'Regular' if dtw_analysis['is_regular_traditional'] else 'Irregular'}")
        
        if 'overall_assessment' in dtw_analysis:
            print(f"  🎯 Consistencia DTW-Onsets: {dtw_analysis['overall_assessment']}")
            print(f"  📈 Onsets bien alineados: {dtw_analysis['well_aligned_ratio']*100:.1f}%")
            print(f"  ⏱️ Desplazamiento máximo: {dtw_analysis['max_displacement']*1000:.1f}ms")
            
        status = "✅" if dtw_regular else "⚠️"
        print(f"  {status} Evaluación final: {'DTW y onsets consistentes' if dtw_regular else 'Inconsistencias detectadas'}")
        
        # Errores rítmicos
        repeats, gaps = rhythm_errors
        print(f"\n🎶 PATRONES RÍTMICOS:")
        print(f"  🔁 Repeticiones detectadas: {len(repeats)}")
        print(f"  🕳️ Huecos grandes detectados: {len(gaps)}")    
        
    
    def extract_analysis_for_csv(self, beat_result: BeatSpectrumResult, onset_result: OnsetAnalysisResult,
                                tempo_result: TempoAnalysisResult, segment_result: Dict,
                                dtw_data, rhythm_errors: Tuple, 
                                mutation_category: str = "", mutation_name: str = "") -> Dict[str, Any]:
        """
        Extrae los resultados del análisis en formato para CSV.
        
        Args:
            beat_result: Resultado del análisis de beat spectrum
            onset_result: Resultado del análisis de onsets
            tempo_result: Resultado del análisis de tempo
            segment_result: Resultado del análisis de segmentos
            dtw_data: Datos DTW (puede ser bool legacy o dict del análisis mejorado)
            rhythm_errors: Errores rítmicos detectados
            mutation_category: Categoría de la mutación
            mutation_name: Nombre de la mutación
              Returns:
            Diccionario con los datos formateados para CSV (columnas de mutación primero)
        """
        stats = onset_result.stats
        repeats, gaps = rhythm_errors
        
        # Manejar dtw_data tanto en formato legacy (bool) como nuevo (dict)
        if isinstance(dtw_data, dict):
            dtw_regular = dtw_data.get('is_regular_combined', dtw_data.get('is_regular_traditional', False))
            dtw_assessment = dtw_data.get('overall_assessment', 'Análisis DTW estándar')
        else:
            # Formato legacy: dtw_data es un booleano
            dtw_regular = dtw_data
            dtw_assessment = 'Camino DTW regular' if dtw_regular else 'Camino DTW con desviaciones anómalas'
        
        return {
            # Información de mutación (primeras columnas)
            'mutation_category': mutation_category,
            'mutation_name': mutation_name,
            
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
            'dtw_regular': dtw_assessment,
            
            # Patrones rítmicos
            'rhythm_repeats': len(repeats),
            'rhythm_large_gaps': len(gaps)
        }


# Función de conveniencia para análisis rápido
def analyze_performance(reference_path: str, live_path: str, save_name: Optional[str] = None, 
                       config: Optional[AudioAnalysisConfig] = None, verbose: bool = True,
                       reference_tempo: Optional[float] = None) -> Dict[str, Any]:
    """
    Función de conveniencia para realizar un análisis completo de interpretación.
    
    Args:
        reference_path: Ruta al archivo de audio de referencia
        live_path: Ruta al archivo de audio en vivo
        save_name: Nombre base para guardar gráficos (opcional)
        config: Configuración de análisis (opcional)
        verbose: Si mostrar resultados por pantalla (opcional)
        reference_tempo: Tempo conocido del MIDI original (opcional, para análisis más robusto)
    
    Returns:
        Diccionario con todos los resultados del análisis
    """
    if config is None:
        config = AudioAnalysisConfig()
    analyzer = MusicAnalyzer(config)
    return analyzer.comprehensive_analysis(reference_path, live_path, save_name, verbose, reference_tempo)


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
          # Onsets básicos con alineamiento DTW
        onsets_data = analyzer.onset_analyzer.compare_onsets_basic(audio_ref, audio_live, sr, wp)
        onsets_ref, onsets_live, matched, unmatched_ref, unmatched_live = onsets_data
        print(f"✅ Onsets emparejados: {len(matched)}")
        print(f"❌ Notas faltantes (en vivo): {len(unmatched_ref)}")
        print(f"❌ Notas extras (en vivo): {len(unmatched_live)}")
          # Tempo usando análisis robusto
        tempo_result = analyzer.tempo_analyzer.analyze_tempo_robust(audio_ref, audio_live, sr)
        print(f"🎼 Tempo referencia: {tempo_result.tempo_ref:.2f} BPM")
        print(f"🎼 Tempo en vivo: {tempo_result.tempo_live:.2f} BPM")
        status = "✅ Tempo similar." if tempo_result.is_similar else "⚠️ Diferencia significativa de tempo."
        print(status)
          # DTW con análisis mejorado
        dtw_analysis = analyzer.dtw_aligner.evaluate_dtw_path_enhanced(wp, audio_ref, audio_live, sr)
        dtw_regular = dtw_analysis['is_regular_combined']
        status = "✅ Camino DTW razonablemente regular." if dtw_regular else "⚠️ Camino DTW con desviaciones anómalas."
        print(status)
        
        if 'overall_assessment' in dtw_analysis:
            print(f"📊 {dtw_analysis['overall_assessment']}")
        
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
        onsets_data = analyzer.onset_analyzer.compare_onsets_basic(audio_ref, audio_live, sr, wp)
        analyzer.visualizer.plot_onset_errors_basic(*onsets_data, save_name=nombre)
        
        print("======== Análisis detallado de onsets ========")
        onset_result = analyzer.onset_analyzer.compare_onsets_detailed(audio_ref, audio_live, sr, wp)
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
