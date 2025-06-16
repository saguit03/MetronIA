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
    
    def load_audio_files(self, reference_path: str, live_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """Carga archivos de audio."""
        reference_audio, sr = librosa.load(reference_path)
        live_audio, _ = librosa.load(live_path, sr=sr)  # Usar mismo sr          
        return reference_audio, live_audio, sr
    
    def save_onsets_analysis_to_csv(self, dtw_onset_result: OnsetDTWAnalysisResult, 
                                   save_name: str, reference_path: str = "", 
                                   live_path: str = "", dir_path: Optional[str] = None) -> None:
        """
        Guarda el análisis detallado de onsets en un archivo CSV.
        
        Args:
            dtw_onset_result: Resultado del análisis DTW de onsets
            save_name: Nombre base para el archivo
            reference_path: Ruta del archivo de referencia (opcional)
            live_path: Ruta del archivo en vivo (opcional)
            dir_path: Directorio donde guardar el archivo (opcional, por defecto "results")
        """
        # Crear directorio de resultados si no existe
        results_dir = Path(dir_path if dir_path else "results")
        results_dir.mkdir(parents=True, exist_ok=True)
          # Nombre del archivo CSV
        if save_name:
            csv_filename = results_dir / f"{save_name}_analysis.csv"
        else:
            csv_filename = results_dir / "analysis.csv"
        
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
                'is_matched': True
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
                'is_matched': False
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
                'is_matched': False
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
          # Logging mejorado
        print(f"💾 Análisis de onsets guardado en: {csv_filename}")
        print(f"   📊 Total de onsets analizados: {len(csv_data)}")
        print(f"   ✅ Correctos: {len([d for d in csv_data if d['onset_type'] == 'correct'])}")
        print(f"   ⏰ Tarde: {len([d for d in csv_data if d['onset_type'] == 'late'])}")
        print(f"   ⚡ Adelantado: {len([d for d in csv_data if d['onset_type'] == 'early'])}")
        print(f"   ❌ Perdidos: {len([d for d in csv_data if d['onset_type'] == 'missing'])}")
        print(f"   ➕ Extra: {len([d for d in csv_data if d['onset_type'] == 'extra'])}")
    
    def comprehensive_analysis(self, reference_path: str, live_path: str, 
                             save_name: Optional[str] = None, verbose: bool = True,
                             reference_tempo: Optional[float] = None,
                             save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Realiza un análisis completo de las interpretaciones.
        
        Args:
            reference_path: Ruta al audio de referencia
            live_path: Ruta al audio en vivo
            save_name: Nombre para guardar gráficos (opcional)
            verbose: Si mostrar resultados detallados por pantalla
            reference_tempo: Tempo conocido del MIDI original (opcional, para análisis más robusto)            save_dir: Directorio específico donde guardar los resultados (opcional)
        """
        # Cargar audios
        audio_ref, audio_live, sr = self.load_audio_files(reference_path, live_path)
          # Calcular proporción de tempo y re-samplear si es necesario
        audio_ref_resampled, tempo_proportion, actual_ref_tempo, actual_live_tempo, resampled_path = self.calculate_tempo_proportion_and_resample(
            audio_ref, audio_live, sr, reference_tempo, reference_path, save_dir
        )
        
        # ========== ESTRATEGIA DE ALINEAMIENTO SEPARADO ==========
        # Usar el nuevo método que separa el análisis:
        # - Beat spectrum: CON alineamiento DTW (para comparar patrones rítmicos globales)
        # - Onsets: SIN alineamiento DTW (para detectar errores de timing reales)
        # IMPORTANTE: Usar audio_ref_resampled para mejorar el alineamiento DTW
        ref_feat, aligned_live_feat_for_beat, wp, unaligned_live_feat = self.dtw_aligner.align_features_for_tempo_comparison(audio_ref_resampled, audio_live, sr)
        
        # Evaluación mejorada del DTW que incluye consistencia con onsets
        dtw_analysis = self.dtw_aligner.evaluate_dtw_path_enhanced(wp, audio_ref_resampled, audio_live, sr)
        dtw_regular = dtw_analysis['is_regular_combined']
        
        # Análisis de beat spectrum (CON alineamiento DTW)
        beat_result = self.beat_spectrum_analyzer.analyze_beat_spectrum(ref_feat, aligned_live_feat_for_beat)
        
        # Análisis de onsets usando DTW con consistencia de ritmo
        # Usar audio_ref_resampled para mejor alineamiento
        dtw_onset_result = self.onset_dtw_analyzer.analyze_onsets_with_rhythm_consistency(
            audio_ref_resampled, audio_live, sr, tolerance_ms=1.0
        )
          # Análisis de tempo usando método robusto si se proporciona tempo de referencia
        if reference_tempo is not None:
            tempo_result = self.tempo_analyzer.analyze_tempo_with_reference(
                audio_ref_resampled, audio_live, sr, reference_tempo
            )
        else:
            # Usar análisis robusto estándar con audio re-sampled
            tempo_result = self.tempo_analyzer.analyze_tempo_robust(audio_ref_resampled, audio_live, sr)
        
        # Actualizar resultado con información de proporción de tempo        tempo_result.tempo_proportion = tempo_proportion
        tempo_result.original_ref_tempo = actual_ref_tempo
        tempo_result.original_live_tempo = actual_live_tempo
        tempo_result.resampling_applied = abs(tempo_proportion - 1.0) > 0.05
        
        segment_result = self.tempo_analyzer.validate_segments(audio_ref_resampled, audio_live, sr)
        
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
            self.save_onsets_analysis_to_csv(dtw_onset_result=dtw_onset_result, save_name="", reference_path=reference_path, live_path=live_path, dir_path=analysis_dir)
          # Mostrar resultados si se solicita
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
            'audio_ref': audio_ref,  # Audio original
            'audio_ref_resampled': audio_ref_resampled,  # Audio re-sampled para DTW
            'audio_live': audio_live,
            'sample_rate': sr,
            'tempo_proportion': tempo_proportion,
            'resampling_applied': abs(tempo_proportion - 1.0) > 0.05,
            'resampled_audio_path': resampled_path  # Ruta del archivo re-sampleado guardado
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
    
    def calculate_tempo_proportion_and_resample(self, audio_ref: np.ndarray, audio_live: np.ndarray, 
                                              sr: int, reference_tempo: Optional[float] = None,
                                              reference_path: Optional[str] = None, 
                                              save_dir: Optional[str] = None) -> tuple:
        """
        Calcula la proporción de tempo (live_tempo/reference_tempo) y re-samplea el audio de referencia
        si la proporción está fuera del rango [0.95, 1.05] para mejorar el alineamiento DTW.
        Guarda el audio re-sampleado en un archivo para usar con OnsetDTW.
        
        Args:
            audio_ref: Audio de referencia
            audio_live: Audio en vivo
            sr: Sample rate
            reference_tempo: Tempo conocido del MIDI original (opcional)
            reference_path: Ruta del archivo de referencia original (para generar nombre del resampled)
            save_dir: Directorio donde guardar el audio re-sampleado (opcional)
            
        Returns:
            tuple: (audio_ref_resampled, tempo_proportion, actual_ref_tempo, actual_live_tempo, resampled_path)
        """
        # Detectar tempos
        if reference_tempo is not None:
            # Si tenemos tempo de referencia del MIDI, usarlo como verdad fundamental
            actual_ref_tempo = reference_tempo
            
            # Detectar tempo del audio en vivo usando análisis robusto
            candidates_live = self.tempo_analyzer.extract_multiple_tempo_candidates(audio_live, sr)
            if not candidates_live:
                tempo_live, _ = librosa.beat.beat_track(y=audio_live, sr=sr)
                actual_live_tempo = float(tempo_live.item())
            else:
                tempo_live_raw = candidates_live[0]
                actual_live_tempo = self.tempo_analyzer.correct_tempo_octave_errors(actual_ref_tempo, tempo_live_raw)
        else:
            # Si no tenemos tempo de referencia, detectar ambos
            tempo_ref, _ = librosa.beat.beat_track(y=audio_ref, sr=sr)
            tempo_live, _ = librosa.beat.beat_track(y=audio_live, sr=sr)
            actual_ref_tempo = float(tempo_ref.item())
            actual_live_tempo = float(tempo_live.item())
        
        # Calcular proporción de tempo
        tempo_proportion = actual_live_tempo / actual_ref_tempo
        
        print(f"🎵 Tempos detectados:")
        print(f"   Referencia: {actual_ref_tempo:.2f} BPM")
        print(f"   En vivo: {actual_live_tempo:.2f} BPM")
        print(f"   Proporción (live/ref): {tempo_proportion:.3f}")
        
        resampled_path = None
        
        # Determinar si necesitamos re-samplear
        if tempo_proportion < 0.95 or tempo_proportion > 1.05:
            print(f"⚡ Aplicando re-sampling del audio de referencia (factor: {tempo_proportion:.3f})")
            
            # Re-samplear el audio de referencia para que coincida con el tempo del live
            # Si live es más rápido (proportion > 1), aceleramos el reference
            # Si live es más lento (proportion < 1), ralentizamos el reference
            audio_ref_resampled = librosa.effects.time_stretch(audio_ref, rate=tempo_proportion)
            
            # Guardar el audio re-sampleado como archivo
            resampled_path = self._save_resampled_audio(
                audio_ref_resampled, sr, tempo_proportion, 
                reference_path, save_dir
            )
            
            print(f"   Duración original ref: {len(audio_ref)/sr:.2f}s")
            print(f"   Duración re-sampled ref: {len(audio_ref_resampled)/sr:.2f}s")
            print(f"   Duración live: {len(audio_live)/sr:.2f}s")
            print(f"   💾 Audio re-sampleado guardado en: {resampled_path}")
        else:
            print(f"✅ No se requiere re-sampling (proporción dentro del rango [0.95, 1.05])")
            audio_ref_resampled = audio_ref
        
        return audio_ref_resampled, tempo_proportion, actual_ref_tempo, actual_live_tempo, resampled_path
    
    def _save_resampled_audio(self, audio_resampled: np.ndarray, sr: int, 
                             tempo_proportion: float, reference_path: Optional[str] = None,
                             save_dir: Optional[str] = None) -> str:
        """
        Guarda el audio re-sampleado en un archivo.
        
        Args:
            audio_resampled: Audio re-sampleado
            sr: Sample rate
            tempo_proportion: Factor de re-sampling aplicado
            reference_path: Ruta del archivo de referencia original
            save_dir: Directorio donde guardar
            
        Returns:
            Ruta del archivo guardado
        """
        # Determinar el directorio de destino
        if save_dir:
            output_dir = Path(save_dir)
        else:
            output_dir = Path("results")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar nombre del archivo
        if reference_path:
            ref_path = Path(reference_path)
            base_name = ref_path.stem
            extension = ".wav"  # Always save as WAV for better compatibility
        else:
            base_name = "reference_resampled"
            extension = ".wav"
        
        # Crear nombre con factor de re-sampling
        resampled_filename = f"{base_name}_{tempo_proportion:.1f}{extension}"
        resampled_path = output_dir / resampled_filename
        
        # Guardar usando soundfile para mejor control de calidad
        sf.write(str(resampled_path), audio_resampled, sr)
        
        return str(resampled_path)

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

