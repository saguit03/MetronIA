import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

matplotlib.use('Agg')

from dataclasses import dataclass
from typing import Optional, Callable

from mdtk.utils import plot_from_df
from mutations.globals import MUTATIONS_PLOTS_PATH
from mutations.logs import save_mutation_logs_to_csv


@dataclass
class MutationResult:
    name: str
    description: str
    function: Callable
    excerpt: Optional[pd.DataFrame] = None
    audio_path: Optional[str] = None
    midi_path: Optional[str] = None
    success: bool = False
    error: Optional[str] = None

    def apply(self, original_excerpt: pd.DataFrame, tempo: int = 120, output_dir: Optional[str] = "mutts") -> bool:
        try:
            import inspect
            sig = inspect.signature(self.function)

            if 'tempo' in sig.parameters:
                self.excerpt, logs = self.function(original_excerpt, tempo=tempo)
            else:
                self.excerpt, logs = self.function(original_excerpt)

            if self.excerpt is not None:
                self.success = True
                mutation_tempo = self.get_mutation_tempo(tempo)
                save_mutation_logs_to_csv(logs, output_dir, self.name)
            else:
                self.success = False
                self.error = "Mutation returned None"
        except Exception as e:
            self.success = False
            self.error = str(e)
            self.excerpt = None
        return self.success

    def set_audio_path(self, path: str):
        self.audio_path = path

    def set_midi_path(self, path: str):
        self.midi_path = path

    def get_mutation_tempo(self, base_tempo: int = 120) -> int:
        if not self.success or self.excerpt is None:
            return base_tempo

        mutation_type = self._detect_tempo_mutation_type()

        if mutation_type is None:
            return base_tempo

        return self._calculate_tempo_for_mutation(base_tempo, mutation_type)

    def _detect_tempo_mutation_type(self) -> Optional[str]:
        tempo_patterns = {
            'faster_tempo': 'faster',
            'a_lot_faster_tempo': 'a_lot_faster',
            'slower_tempo': 'slower',
            'a_lot_slower_tempo': 'a_lot_slower',
            'accelerando': 'accelerando',
            'ritardando': 'ritardando',
            'tempo_fluctuation': 'fluctuation'
        }

        for pattern, mutation_type in tempo_patterns.items():
            if pattern in self.name:
                return mutation_type

        return None

    def _calculate_tempo_for_mutation(self, base_tempo: int, mutation_type: str) -> int:
        from mutations.globals import (
            FASTER, A_LOT_FASTER, SLOWER, A_LOT_SLOWER,
            ACCELERANDO, RITARDANDO
        )

        tempo_factors = {
            'faster': FASTER,
            'a_lot_faster': A_LOT_FASTER,
            'slower': SLOWER,
            'a_lot_slower': A_LOT_SLOWER,
            'accelerando': ACCELERANDO,
            'ritardando': RITARDANDO,
            'fluctuation': 1.0
        }

        factor = tempo_factors.get(mutation_type, 1.0)
        calculated_tempo = int(base_tempo * factor)

        return max(40, min(200, calculated_tempo))

    def is_tempo_mutation(self) -> bool:
        return self._detect_tempo_mutation_type() is not None

    def __str__(self):
        status = "✓" if self.success else "✗"
        return f"{status} {self.name}: {self.description}"


def save_plot_against_orig(orig_excerpt, list_of_diff_excerpts, save_name=None):
    nr_diffs = len(list_of_diff_excerpts)
    fig, ax = plt.subplots(
        1, nr_diffs + 1, figsize=(6 * (nr_diffs + 1), 4), sharex=True, sharey=True
    )
    plt.sca(ax[0])
    plot_from_df(orig_excerpt, alpha=0.3)
    plt.title("original")

    for ii in range(nr_diffs):
        plt.sca(ax[ii + 1])
        plot_from_df(orig_excerpt, alpha=0.3)
        plot_from_df(list_of_diff_excerpts[ii])
        plt.title(f"deg {ii + 1}")

    if save_name is not None:
        Path(MUTATIONS_PLOTS_PATH).mkdir(parents=True, exist_ok=True)
        fig_path = Path(MUTATIONS_PLOTS_PATH) / f"{save_name}.png"
        plt.savefig(fig_path, bbox_inches="tight")

    plt.close(fig)
