"""
Visualizador de análisis de audio musical.
"""

import numpy as np
import matplotlib.pyplot as plt
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
            fig_path = Path(ANALYSIS_PLOTS_PATH) / f"{save_name}.png"
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
            fig_path = Path(ANALYSIS_PLOTS_PATH) / f"plt_onsets_errors_comparisons_{save_name}.png"
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
            fig_path = Path(ANALYSIS_PLOTS_PATH) / f"plt_onsets_detailed_comparisons_{save_name}.png"
            plt.savefig(fig_path, dpi=self.config.plot_dpi)
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig if not show else None
