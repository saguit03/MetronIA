import librosa
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

from utils.audio_utils import load_audio_files, stretch_audio, calculate_warping_path
from .beat_spectrum_analyzer import BeatSpectrumAnalyzer
from utils.config import AudioAnalysisConfig
from .onset_dtw_analyzer import OnsetDTWAnalyzer
from utils.onset_utils import OnsetUtils
from .tempo_analyzer import TempoAnalyzer
from utils.visualizer import Visualizer


class MetronIA:
    def __init__(self, config: Optional[AudioAnalysisConfig] = None):
        self.config = config or AudioAnalysisConfig()
        self.onset_dtw_analyzer = OnsetDTWAnalyzer(self.config)
        self.tempo_analyzer = TempoAnalyzer(self.config)
        self.beat_spectrum_analyzer = BeatSpectrumAnalyzer(self.config)
        self.visualizer = Visualizer(self.config)

    def comprehensive_analysis(self, reference_path: str, live_path: str,
                               save_name,
                               save_dir,
                               reference_tempo: Optional[float] = None,
                               mutation_name: Optional[str] = None,
                               verbose: Optional[bool] = False) -> Dict[str, Any]:
        # 1. Carga de audios
        reference_audio, live_audio, sampling_rate = load_audio_files(reference_path, live_path)

        # 2. Eliminación de silencios al inicio y al final de los audios
        trimmed_reference_audio, (reference_start_index, reference_end_index) = librosa.effects.trim(reference_audio)
        trimmed_live_audio, (live_start_index, live_end_index) = librosa.effects.trim(live_audio)
        if verbose: print(f"\n🎧 Audios cargados y silencios eliminados.")

        # 3. Ajuste de audios
        aligned_live_audio = stretch_audio(trimmed_reference_audio, trimmed_live_audio, sampling_rate, self.config.hop_length, save_name="aligned_audio", save_dir=save_dir)
        if verbose:
            print(f"⛓️‍💥 Ajuste de audios completado.")

        # 4. Análisis de tempo
        tempo_result = self.tempo_analyzer.analyze_tempo_with_reference(trimmed_reference_audio, trimmed_live_audio, sampling_rate, reference_tempo)
        if verbose: print(f"⌛ Tempo de referencia: {tempo_result.tempo_ref} BPM \n⌛ Tempo en vivo: {tempo_result.tempo_live} BPM")

        # 5. Cálculo de DTW con el audio en vivo alineado
        distance, wp, wp_s = calculate_warping_path(trimmed_reference_audio, aligned_live_audio, sampling_rate,
                                                    self.config.hop_length)
        if verbose:
            print(f"🧮 DTW calculado.")
            print(f"📊 Comenzando análisis...")

        # 6. ANÁLISIS Y OBTENCIÓN DE RESULTADOS: Detección y alineamiento de onsets con DTW   
        if verbose: print(f"🕰️ Detección y emparejamiento de onsets con DTW...")
        dtw_onset_result = self.onset_dtw_analyzer.match_onsets_with_dtw(
            trimmed_reference_audio, aligned_live_audio, sampling_rate, wp
        )
        if verbose:
            print(f"* Onsets correctos: {len(dtw_onset_result.correct_matches)}")
            print(f"* Onsets atrasados: {len(dtw_onset_result.late_matches)}")
            print(f"* Onsets adelantados: {len(dtw_onset_result.early_matches)}")
            print(f"* Onsets perdidos: {len(dtw_onset_result.missing_onsets)}")
            print(f"* Onsets extras: {len(dtw_onset_result.extra_onsets)}")

        # 7. Almacenamiento de los resultados de análisis
        if save_name:
            analysis_dir = save_dir or f"results/{save_name}"
            if verbose: print(f"💾 Almacenando resultados en {analysis_dir}...")
            # 7.1 Gráfica de beat spectrum
            beat_result = self.beat_spectrum_analyzer.beat_spectrum(trimmed_reference_audio, aligned_live_audio, sampling_rate)
            self.visualizer.plot_beat_spectrum_comparison(result=beat_result, sr=sampling_rate, save_name="beat_spectrum", dir_path=analysis_dir)
            # 7.2 Línea temporal de onsets
            self.visualizer.plot_onsets(result=dtw_onset_result, save_name="onset_timeline", dir_path=analysis_dir)
            # 7.3 Gráfico de tarta de onsets
            self.visualizer.plot_onset_pie(dtw_onset_result=dtw_onset_result, save_name="onset_pie", dir_path=analysis_dir)
            # 7.4 Análisis completo de onsets
            OnsetUtils.save_onsets_analysis_to_csv(dtw_onset_result=dtw_onset_result, save_name="onset_analysis", dir_path=analysis_dir, mutation_name=mutation_name)
            # 7.5 Informe de análisis
            self.save_analysis_summary(tempo_result=tempo_result, dtw_onset_results=dtw_onset_result, dir_path=analysis_dir)

        return {
            'dtw_onsets': dtw_onset_result,
            'tempo': tempo_result,
        }

    def save_analysis_summary(self, tempo_result, dtw_onset_results, dir_path):
        data = {
            'tempo_ref': tempo_result.tempo_ref,
            'tempo_live': tempo_result.tempo_live,
            'tempo_difference': tempo_result.difference,
            'total_onsets_ref': len(dtw_onset_results.correct_matches) + len(dtw_onset_results.late_matches) + len(
                dtw_onset_results.early_matches) + len(dtw_onset_results.missing_onsets),
            'correct_onsets': len(dtw_onset_results.correct_matches),
            'late_onsets': len(dtw_onset_results.late_matches),
            'early_onsets': len(dtw_onset_results.early_matches),
            'ref_missing_onsets': len(dtw_onset_results.missing_onsets),
            'live_extra_onsets': len(dtw_onset_results.extra_onsets),
            'correct_notes': dtw_onset_results.correct_notes
        }
        df = pd.DataFrame([data])
        results_dir = Path(dir_path)
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_filename = results_dir / f"analysis_summary.csv"
        df.to_csv(csv_filename, index=False)

# Por si las dudas, he elegido yo misma los emoticonos, no ha sido ninguna IA :sob:
