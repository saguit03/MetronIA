"""
Visualizador de análisis de audio musical.
"""

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
import numpy as np

from .config import AudioAnalysisConfig
from .onset_results import OnsetDTWAnalysisResult
from .results import BeatSpectrumResult


class AudioVisualizer:
    """Visualizador de análisis de audio."""
    
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
    
    def plot_beat_spectrum_comparison(self, result: BeatSpectrumResult, sr: int, 
                                    save_name: Optional[str] = None, dir_path: Optional[str] = "results") -> Optional[plt.Figure]:
        """Plotea comparación de beat spectrums."""
        times = np.arange(1, len(result.beat_ref) + 1) * self.config.hop_length / sr
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times, result.beat_ref, label='Referencia', linewidth=2)
        ax.plot(times, result.beat_aligned, label='En vivo (alineado por DTW)', linewidth=2)
        ax.set_xlabel("Time Lag (s)")
        ax.set_ylabel("Similarity")
        ax.set_title("Comparación de Beat Spectrums alineados con DTW")
        fig.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_name:
            plots_dir = Path(dir_path)
            plots_dir.mkdir(parents=True, exist_ok=True)
            fig_path = plots_dir / f"{save_name}.png"
            plt.savefig(fig_path, dpi=self.config.plot_dpi)
        plt.close(fig)
            
    def plot_timeline_onset_errors_detailed(self, result: OnsetDTWAnalysisResult,
                                  save_name: Optional[str] = None, dir_path: Optional[str] = "results") -> Optional[plt.Figure]:
        """Plotea análisis detallado de errores de onsets."""
        fig, ax = plt.subplots(figsize=(14, 3))
        ref_onsets_from_matches = [m.ref_onset for m in result.matches]
        ref_onsets_from_missing = [ref_time for ref_time, _ in result.missing_onsets]
        all_ref_onsets = ref_onsets_from_matches + ref_onsets_from_missing
        correct_live_onsets = [m.live_onset for m in result.matches if m.classification.value == 'correct']
        early_live_onsets = [m.live_onset for m in result.matches if m.classification.value == 'early']
        late_live_onsets = [m.live_onset for m in result.matches if m.classification.value == 'late']
        extra_live_onsets = [live_time for live_time, _ in result.extra_onsets]
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
        fig.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            plots_dir = Path(dir_path)
            plots_dir.mkdir(parents=True, exist_ok=True)
            fig_path = plots_dir / f"{save_name}.png"
            plt.savefig(fig_path, dpi=self.config.plot_dpi)

        return fig, ax
    
    def plot_onsets(self, reference, corrects, early, late, extras, missed, save_name: Optional[str] = None, dir_path: Optional[str] = "results"):
        """
        Dibuja los onsets en tres líneas: referencia, correctos+adelantados+atrasados, extras+perdidos.
        """
        # Alineación por capas:
        #  Línea 0: referencia
        #  Línea 1: correctos, adelantados, atrasados
        #  Línea 2: extras, perdidos
        fig, ax = plt.subplots(figsize=(12, 4))

        # Cada evento es una lista (incluso si hay uno solo) por capa
        def plot_layer(events, y, color, label):
            if events:
                ax.eventplot(events, lineoffsets=y, colors=color, linelengths=0.8, label=label)

        # Línea 0: referencia
        plot_layer(reference, y=0, color='black', label='Referencia')

        # Línea 1: correctos (verde), adelantados (naranja), atrasados (azul)
        plot_layer(corrects, y=1, color='green', label='Correctos')
        plot_layer(early, y=1, color='orange', label='Adelantados')
        plot_layer(late, y=1, color='blue', label='Atrasados')

        # Línea 2: extras (rojo), perdidos (magenta)
        plot_layer(extras, y=2, color='red', label='Extras')
        plot_layer(missed, y=2, color='magenta', label='Perdidos')

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Referencia', 'Alineados', 'Errores'])
        ax.set_xlabel("Tiempo (s)")
        ax.set_title('Errores de ejecución nota por nota: adelantados, atrasados, extras y faltantes')
        fig.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            plots_dir = Path(dir_path)
            plots_dir.mkdir(parents=True, exist_ok=True)
            fig_path = plots_dir / f"{save_name}.png"
            plt.savefig(fig_path, dpi=self.config.plot_dpi)

        return fig, ax
