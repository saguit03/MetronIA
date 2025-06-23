from pathlib import Path

import matplotlib.pyplot as plt

from mdtk.utils import plot_from_df
from mutations.config import MUTATIONS_PLOTS_PATH


def save_plot_against_orig(orig_excerpt, list_of_diff_excerpts, save_name=None):
    """
    Guarda un gráfico comparativo de un fragmento original y sus modificaciones.

    Parameters
    ----------
    orig_excerpt : pd.DataFrame
        Fragmento original.

    list_of_diff_excerpts : list[pd.DataFrame]
        Lista de fragmentos modificados.

    save_name : str or None
        Nombre base del archivo a guardar. Si es None, no se guarda.
    """
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
