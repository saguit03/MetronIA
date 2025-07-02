import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Optional

matplotlib.use('Agg')
import numpy as np

from utils.config import AudioAnalysisConfig
from analyzers.onset_results import OnsetDTWAnalysisResult
from analyzers.beat_spectrum_analyzer import BeatSpectrumResult

ONSET_COLORS = ['blue', 'green', 'orange', 'purple', 'gray', 'red']


class Visualizer:
    def __init__(self, config: AudioAnalysisConfig):
        self.config = config

    def plot_beat_spectrum_comparison(self, result: BeatSpectrumResult, sr: int, save_name, dir_path) -> Optional[
        plt.Figure]:
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

    def get_similarity_notes(self, matches):
        return [
            len([m for m in matches if m.note_similarity == 1.0]),
            len([m for m in matches if m.note_similarity == 0.8]),
            len([m for m in matches if m.note_similarity == 0.6]),
            len([m for m in matches if m.note_similarity == 0.4]),
            len([m for m in matches if m.note_similarity == 0.2]),
            len([m for m in matches if m.note_similarity < 0.2])
        ]

    def plot_onset_pie(self, dtw_onset_result: OnsetDTWAnalysisResult, save_name: str,
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

        wedges, texts, autotexts = ax.pie(
            counts,
            labels=categories,
            colors=ONSET_COLORS[1:5],
            autopct='%1.1f%%',
            startangle=90
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)

        for text in texts:
            text.set_fontsize(11)
            text.set_fontweight('bold')

        legend_labels = [f'{cat}: {count}' for cat, count in zip(categories, counts)]
        ax.legend(wedges, legend_labels, title="Cantidad", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        ax.set_title('Distribución de Tipos de Onsets', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        output_dir = Path(dir_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{save_name}.png"

        plt.savefig(output_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close(fig)

    def plot_onset_timeline(self, result: OnsetDTWAnalysisResult, save_name, dir_path) -> Optional[plt.Figure]:
        fig, ax = plt.subplots(figsize=(14, 3))
        onsets_ref_from_matches = [m.onset_ref for m in result.matches]
        onsets_ref_from_missing = [ref_time for ref_time, _ in result.missing_onsets]
        all_onsets_ref = onsets_ref_from_matches + onsets_ref_from_missing
        correct_onsets_live = [m.onset_live for m in result.matches if m.classification.value == 'correct']
        early_onsets_live = [m.onset_live for m in result.matches if m.classification.value == 'early']
        late_onsets_live = [m.onset_live for m in result.matches if m.classification.value == 'late']
        extra_onsets_live = [live_time for live_time, _ in result.extra_onsets]
        ax.vlines(all_onsets_ref, 0.8, 1.0, color='blue', label='Onsets referencia', linewidth=2)
        ax.vlines(correct_onsets_live, 0.6, 0.8, color='green',
                  label='Onsets correctos', linewidth=2)
        ax.vlines(early_onsets_live, 0.4, 0.6, color='orange',
                  label='Onsets adelantados', linewidth=2)
        ax.vlines(late_onsets_live, 0.2, 0.4, color='purple',
                  label='Onsets atrasados', linewidth=2)
        ax.vlines(onsets_ref_from_missing, 0.0, 0.2, color='black', label='Notas faltantes', linewidth=2)
        ax.vlines(extra_onsets_live, -0.2, 0.0, color='red', label='Notas extras', linewidth=2)

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

        plt.close()

    def plot_onsets(self, result: OnsetDTWAnalysisResult, save_name, dir_path):
        fig, ax = plt.subplots(figsize=(16, 4))
        onsets_ref_from_matches = [m.onset_ref for m in result.matches]
        onsets_ref_from_missing = [ref_time for ref_time, _ in result.missing_onsets]
        all_onsets_ref = onsets_ref_from_matches + onsets_ref_from_missing

        correct_onsets_live = [m.onset_live for m in result.matches if m.classification.value == 'correct']
        early_onsets_live = [m.onset_live for m in result.matches if m.classification.value == 'early']
        late_onsets_live = [m.onset_live for m in result.matches if m.classification.value == 'late']
        extra_onsets_live = [live_time for live_time, _ in result.extra_onsets]

        positions = [
            all_onsets_ref,  # y = 2.0
            correct_onsets_live,  # y = 1.0
            early_onsets_live,  # y = 1.0
            late_onsets_live,  # y = 1.0
            onsets_ref_from_missing,  # y = 0.5
            extra_onsets_live  # y = 0.5
        ]

        lineoffsets = [2.0, 1.0, 1.0, 1.0, 0.5, 0.5]
        colors = ONSET_COLORS
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
                ax.plot([m.onset_ref, m.onset_live], [2.0, 1.0],
                        color='black', linewidth=1.2, alpha=0.8, linestyle='dotted')

        ax.set_yticks([0.5, 1.0, 2.0])
        ax.set_yticklabels(['Extras/Perdidos', 'En vivo', 'Referencia'])
        ax.set_xlabel("Tiempo (s)")
        ax.set_title(f"Análisis de onsets:\ncorrectos, adelantados, atrasados,\nextras y perdidos")
        ax.grid(True, alpha=0.3)

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
        if save_name:
            plots_dir = Path(dir_path)
            plots_dir.mkdir(parents=True, exist_ok=True)
            fig_path = plots_dir / f"{save_name}.png"
            plt.savefig(fig_path, dpi=self.config.plot_dpi)

        plt.close(fig)
