"""
Visualizador de análisis de audio musical.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Optional
from .config import AudioAnalysisConfig
from .results import BeatSpectrumResult, OnsetAnalysisResult
from pathlib import Path
from analyzers.config import ANALYSIS_PLOTS_PATH

class AudioVisualizer:
    """Visualizador de análisis de audio."""
    
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
    
    def plot_beat_spectrum_comparison(self, result: BeatSpectrumResult, sr: int, 
                                    save_name: Optional[str] = None, show: bool = False) -> Optional[plt.Figure]:
        """Plotea comparación de beat spectrums."""
        times = np.arange(1, len(result.beat_ref) + 1) * self.config.hop_length / sr
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times, result.beat_ref, label='Referencia', linewidth=2)
        ax.plot(times, result.beat_aligned, label='En vivo (alineado por DTW)', linewidth=2)
        ax.set_xlabel("Time Lag (s)")
        ax.set_ylabel("Similarity")
        ax.set_title("Comparación de Beat Spectrums alineados con DTW")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            Path(ANALYSIS_PLOTS_PATH).mkdir(exist_ok=True)
            fig_path = Path(ANALYSIS_PLOTS_PATH) / f"{save_name}_dtw.png"
            plt.savefig(fig_path, dpi=self.config.plot_dpi)
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig if not show else None
    
    def plot_onset_errors_basic(self, onsets_ref: np.ndarray, onsets_live: np.ndarray,
                               matched: List[Tuple], unmatched_ref: List, unmatched_live: List,
                               save_name: Optional[str] = None, show: bool = False) -> Optional[plt.Figure]:
        """Plotea errores básicos de onsets."""
        fig, ax = plt.subplots(figsize=(12, 3))
        
        ax.vlines(onsets_ref, 0.8, 1.0, color='blue', label='Onsets referencia', linewidth=2)
        matched_live = [live for _, live in matched]
        ax.vlines(matched_live, 0.6, 0.8, color='green', label='Onsets emparejados', linewidth=2)
        ax.vlines(unmatched_ref, 0.4, 0.6, color='black', label='Notas faltantes', linewidth=2)
        ax.vlines(unmatched_live, 0.2, 0.4, color='red', label='Notas extras', linewidth=2)
        
        ax.set_ylim(0, 1.1)
        ax.set_yticks([])
        ax.set_xlabel('Tiempo (segundos)')
        ax.set_title('Detección de errores rítmicos nota por nota')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            Path(ANALYSIS_PLOTS_PATH).mkdir(exist_ok=True)
            fig_path = Path(ANALYSIS_PLOTS_PATH) / f"{save_name}_basic_onsets.png"
            plt.savefig(fig_path, dpi=self.config.plot_dpi)
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig if not show else None
    
    def plot_onset_errors_detailed(self, result: OnsetAnalysisResult,
                                  save_name: Optional[str] = None, show: bool = False) -> Optional[plt.Figure]:
        """Plotea análisis detallado de errores de onsets."""
        fig, ax = plt.subplots(figsize=(14, 3))
        
        ax.vlines(result.onsets_ref, 0.8, 1.0, color='blue', label='Onsets referencia', linewidth=2)
        ax.vlines([live for _, live in result.matched_correct], 0.6, 0.8, color='green', 
                 label='Onsets correctos', linewidth=2)
        ax.vlines([live for _, live in result.matched_early], 0.4, 0.6, color='orange', 
                 label='Onsets adelantados', linewidth=2)        
        ax.vlines([live for _, live in result.matched_late], 0.2, 0.4, color='purple', 
                 label='Onsets atrasados', linewidth=2)
        ax.vlines(result.unmatched_ref, 0.0, 0.2, color='black', label='Notas faltantes', linewidth=2)
        ax.vlines(result.unmatched_live, -0.2, 0.0, color='red', label='Notas extras', linewidth=2)
        
        ax.set_ylim(-0.3, 1.1)
        ax.set_yticks([])
        ax.set_xlabel('Tiempo (segundos)')
        ax.set_title('Errores de ejecución nota por nota: adelantados, atrasados, extras y faltantes')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            Path(ANALYSIS_PLOTS_PATH).mkdir(exist_ok=True)
            fig_path = Path(ANALYSIS_PLOTS_PATH) / f"{save_name}_detailed_onsets.png"
            plt.savefig(fig_path, dpi=self.config.plot_dpi)
            csv_path = self.export_onset_analysis_to_csv(result, save_name)
            print(f"📊 Análisis de onsets exportado a CSV: {csv_path}")
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig if not show else None
    
    def export_onset_analysis_to_csv(self, result: OnsetAnalysisResult, 
                                   save_name: str, 
                                   output_dir: Optional[str] = None) -> str:
        """
        Exporta los resultados del análisis de onsets a un archivo CSV.
        
        Args:
            result: Resultado del análisis de onsets
            save_name: Nombre base para el archivo
            output_dir: Directorio de salida (opcional, usa ANALYSIS_PLOTS_PATH por defecto)
        
        Returns:
            Ruta del archivo CSV generado
        """
        if output_dir is None:
            output_dir = ANALYSIS_PLOTS_PATH
        
        # Crear directorio si no existe
        Path(output_dir).mkdir(exist_ok=True)
        
        # Preparar datos para CSV
        onset_data = []
        
        # Onsets correctos
        for ref_time, live_time in result.matched_correct:
            onset_data.append({
                'reference_timestamp': ref_time,
                'live_timestamp': live_time,
                'category': 'correcto',
                'time_difference': live_time - ref_time,
                'abs_time_difference': abs(live_time - ref_time)
            })
        
        # Onsets tempranos (adelantados)
        for ref_time, live_time in result.matched_early:
            onset_data.append({
                'reference_timestamp': ref_time,
                'live_timestamp': live_time,
                'category': 'temprano',
                'time_difference': live_time - ref_time,  # Será negativo
                'abs_time_difference': abs(live_time - ref_time)
            })
        
        # Onsets tardíos (atrasados)
        for ref_time, live_time in result.matched_late:
            onset_data.append({
                'reference_timestamp': ref_time,
                'live_timestamp': live_time,
                'category': 'tarde',
                'time_difference': live_time - ref_time,  # Será positivo
                'abs_time_difference': abs(live_time - ref_time)
            })
        
        # Onsets faltantes (solo en referencia)
        for ref_time in result.unmatched_ref:
            onset_data.append({
                'reference_timestamp': ref_time,
                'live_timestamp': None,
                'category': 'faltante',
                'time_difference': None,
                'abs_time_difference': None
            })
        
        # Onsets extra (solo en vivo)
        for live_time in result.unmatched_live:
            onset_data.append({
                'reference_timestamp': None,
                'live_timestamp': live_time,
                'category': 'extra',
                'time_difference': None,
                'abs_time_difference': None
            })
        
        # Crear DataFrame y ordenar por timestamp de referencia
        df = pd.DataFrame(onset_data)
        
        # Ordenar por reference_timestamp, poniendo los NaN al final
        df_sorted = df.sort_values(
            by=['reference_timestamp', 'live_timestamp'], 
            na_position='last'
        ).reset_index(drop=True)
        
        # Añadir columnas adicionales de estadísticas
        df_sorted['onset_index'] = range(len(df_sorted))
        df_sorted['analysis_name'] = save_name
        
        # Guardar CSV
        csv_path = Path(output_dir) / f"{save_name}_onset_analysis.csv"
        df_sorted.to_csv(csv_path, index=False, float_format='%.6f')
        
        # Crear también un archivo de resumen
        self._create_onset_summary_csv(result, save_name, output_dir)
        
        return str(csv_path)
    
    def _create_onset_summary_csv(self, result: OnsetAnalysisResult, 
                                 save_name: str, output_dir: str):
        """
        Crea un archivo CSV con el resumen estadístico del análisis de onsets.
        """
        stats = result.stats
        
        summary_data = {
            'analysis_name': [save_name],
            'total_reference_onsets': [stats['total_ref']],
            'total_live_onsets': [stats['total_live']],
            'correct_onsets': [stats['correct']],
            'early_onsets': [stats['early']],
            'late_onsets': [stats['late']],
            'missing_onsets': [stats['missing']],
            'extra_onsets': [stats['extra']],
            'precision_percentage': [stats['correct']/(stats['total_ref'] or 1)*100],
            'recall_percentage': [stats['correct']/(stats['total_live'] or 1)*100],
        }
        
        # Calcular estadísticas de tiempo si hay onsets emparejados
        all_matched = result.matched_correct + result.matched_early + result.matched_late
        if all_matched:
            time_diffs = [live - ref for ref, live in all_matched]
            summary_data.update({
                'mean_time_difference': [np.mean(time_diffs)],
                'std_time_difference': [np.std(time_diffs)],
                'min_time_difference': [np.min(time_diffs)],
                'max_time_difference': [np.max(time_diffs)],
                'median_time_difference': [np.median(time_diffs)]
            })
        else:
            summary_data.update({
                'mean_time_difference': [None],
                'std_time_difference': [None],
                'min_time_difference': [None],
                'max_time_difference': [None],
                'median_time_difference': [None]
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = Path(output_dir) / f"{save_name}_onset_summary.csv"
        summary_df.to_csv(summary_path, index=False, float_format='%.6f')
