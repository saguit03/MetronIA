"""
Visualizador de análisis de audio musical.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path

from .config import AudioAnalysisConfig
from .results import BeatSpectrumResult
from .onset_results import OnsetDTWAnalysisResult

class AudioVisualizer:
    """Visualizador de análisis de audio."""
    
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
    
    def plot_beat_spectrum_comparison(self, result: BeatSpectrumResult, sr: int, 
                                    save_name: Optional[str] = None, dir_path: Optional[str] = "results", show: bool = False) -> Optional[plt.Figure]:
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
            plots_dir = Path(dir_path)
            plots_dir.mkdir(parents=True, exist_ok=True)
            fig_path = plots_dir / f"{save_name}_beat_spectrum.png"
            plt.savefig(fig_path, dpi=self.config.plot_dpi)
        
        if show:
            plt.show()
        else:
            plt.close()
        return fig if not show else None
    
    def plot_timeline_onset_errors_detailed(self, result: OnsetDTWAnalysisResult,
                                  save_name: Optional[str] = None, dir_path: Optional[str] = "results", show: bool = False) -> Optional[plt.Figure]:
        """Plotea análisis detallado de errores de onsets."""
        fig, ax = plt.subplots(figsize=(14, 3))
        
        # Extraer onsets de referencia de matches y missing_onsets
        ref_onsets_from_matches = [m.ref_onset for m in result.matches]
        ref_onsets_from_missing = [ref_time for ref_time, _ in result.missing_onsets]
        all_ref_onsets = ref_onsets_from_matches + ref_onsets_from_missing
        
        # Extraer onsets en vivo por categoría
        correct_live_onsets = [m.live_onset for m in result.matches if m.classification.value == 'correct']
        early_live_onsets = [m.live_onset for m in result.matches if m.classification.value == 'early']
        late_live_onsets = [m.live_onset for m in result.matches if m.classification.value == 'late']
        extra_live_onsets = [live_time for live_time, _ in result.extra_onsets]
        
        # Plotear
        ax.vlines(all_ref_onsets, 0.8, 1.0, color='blue', label='Onsets referencia', linewidth=2)
        ax.vlines(correct_live_onsets, 0.6, 0.8, color='green', 
                 label='Onsets correctos', linewidth=2)
        ax.vlines(early_live_onsets, 0.4, 0.6, color='orange', 
                 label='Onsets adelantados', linewidth=2)        
        ax.vlines(late_live_onsets, 0.2, 0.4, color='purple', 
                 label='Onsets atrasados', linewidth=2)
        ax.vlines(ref_onsets_from_missing, 0.0, 0.2, color='black', label='Notas faltantes', linewidth=2)
        ax.vlines(extra_live_onsets, -0.2, 0.0, color='red', label='Notas extras', linewidth=2)
        
        ax.set_ylim(-0.3, 1.1)
        ax.set_yticks([])
        ax.set_xlabel('Tiempo (segundos)')
        ax.set_title('Errores de ejecución nota por nota: adelantados, atrasados, extras y faltantes')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            plots_dir = Path(dir_path)
            plots_dir.mkdir(parents=True, exist_ok=True)
            fig_path = plots_dir / f"{save_name}_timeline.png"
            plt.savefig(fig_path, dpi=self.config.plot_dpi)
            
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig if not show else None
