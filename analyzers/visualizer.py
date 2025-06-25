"""
Visualizador de análisis de audio musical.
"""

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.use('Agg') 
import numpy as np

from .config import AudioAnalysisConfig
from .onset_results import OnsetDTWAnalysisResult
from .beat_spectrum_analyzer import BeatSpectrumResult

class Visualizer:
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config
    
    def plot_beat_spectrum_comparison(self, result: BeatSpectrumResult, sr: int, save_name,dir_path) -> Optional[plt.Figure]:
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
        
    def plot_onset_distribution(self, dtw_onset_result: OnsetDTWAnalysisResult, save_name: str, 
                                dir_path: Optional[str] = None) -> str:
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ['Correctos', 'Tarde', 'Adelantados', 'Faltantes', 'Extras']
        counts = [
            len(dtw_onset_result.correct_matches),
            len(dtw_onset_result.late_matches),
            len(dtw_onset_result.early_matches),
            len(dtw_onset_result.missing_onsets),
            len(dtw_onset_result.extra_onsets)
        ]
        colors = ['green', 'orange', 'red', 'gray', 'purple']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.7)
        ax.set_title('Distribución de Tipos de Onsets')
        ax.set_ylabel('Cantidad')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5, f'{count}',
                    ha='center', va='bottom', fontsize=10)
    
        plt.tight_layout()
        
        output_dir = Path(dir_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{save_name}.png"
        
        plt.savefig(output_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close(fig)

    def plot_onset_timeline(self, result: OnsetDTWAnalysisResult, save_name, dir_path) -> Optional[plt.Figure]:
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
        
        plots_dir = Path(dir_path)
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plots_dir / f"{save_name}.png"
        plt.savefig(fig_path, dpi=self.config.plot_dpi)

        return fig, ax
    
    def plot_onsets(self, result: OnsetDTWAnalysisResult, save_name, dir_path):
        fig, ax = plt.subplots(figsize=(16, 4))
        ref_onsets_from_matches = [m.ref_onset for m in result.matches]
        ref_onsets_from_missing = [ref_time for ref_time, _ in result.missing_onsets]
        all_ref_onsets = ref_onsets_from_matches + ref_onsets_from_missing

        correct_live_onsets = [m.live_onset for m in result.matches if m.classification.value == 'correct']
        early_live_onsets = [m.live_onset for m in result.matches if m.classification.value == 'early']
        late_live_onsets = [m.live_onset for m in result.matches if m.classification.value == 'late']
        extra_live_onsets = [live_time for live_time, _ in result.extra_onsets]

        positions = [
            all_ref_onsets,              # y = 2.0
            correct_live_onsets,         # y = 1.0
            early_live_onsets,           # y = 1.0
            late_live_onsets,            # y = 1.0
            ref_onsets_from_missing,     # y = 0.5
            extra_live_onsets            # y = 0.5
        ]

        lineoffsets = [2.0, 1.0, 1.0, 1.0, 0.5, 0.5]
        colors = ['blue', 'green', 'orange', 'purple', 'gray', 'red']
        linelengths = [0.8, 0.8, 0.8, 0.8, 1.0, 1.0]
        linestyles = ['solid', 'solid', 'solid', 'solid', 'dotted', 'dotted']

        ax.eventplot(
            positions,
            lineoffsets=lineoffsets,
            colors=colors,
            linelengths=linelengths,
            linestyle=linestyles,
        )

        for m in result.matches:
            if m.classification.value == 'correct':
                ax.plot([m.ref_onset, m.live_onset], [2.0, 1.0],
                        color='black', linewidth=1.2, alpha=0.8, linestyle='dotted')

        ax.set_yticks([0.5, 1.0, 2.0])
        ax.set_yticklabels(['Extras/Perdidos', 'En vivo', 'Referencia'])
        ax.set_xlabel("Tiempo (s)")
        ax.set_title(f"Análisis de onsets:\ncorrectos, adelantados, atrasados,\nextras y perdidos")  # Se usará una caja de texto en su lugar
        ax.grid(True, alpha=0.3)

        # ➡️ Leyenda a la derecha, en tres columnas
        legend_elements = [
            Line2D([0], [0], color='blue', label='Referencia'),
            Line2D([0], [0], color='green', label='Correctos'),
            Line2D([0], [0], color='orange', label='Adelantados'),
            Line2D([0], [0], color='purple', label='Atrasados'),
            Line2D([0], [0], color='gray', label='Perdidos'),
            Line2D([0], [0], color='red', label='Extras'),
        ]
        
        fig.legend(handles=legend_elements, loc='upper right', ncol=3, bbox_to_anchor=(0.99, 0.97))
        plt.tight_layout()
        # fig.set_tight_layout(True)
        # plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.9])  # Deja espacio arriba
        # Guardar si es necesario
        if save_name:
            plots_dir = Path(dir_path)
            plots_dir.mkdir(parents=True, exist_ok=True)
            fig_path = plots_dir / f"{save_name}.png"
            plt.savefig(fig_path, dpi=self.config.plot_dpi)

        return fig, ax
