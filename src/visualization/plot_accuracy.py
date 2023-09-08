import os
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from ..align_representations import AlignRepresentations


def plot_accuracy(
    align_representation: AlignRepresentations,
    eval_type: str = "ot_plan",
    scatter: bool = True,
    fig_dir: Optional[str] = None,
    fig_name: str = "matching_rate_ot_plan",
    *,
    figsize: Tuple[int, int] = (5, 3),
    fig_ext: str = "png",
    show_figure: bool = False,
) -> matplotlib.axes.Axes:
    """Plot the accuracy of the unsupervised alignment for each top_k

    Args:
        eval_type (str, optional):
            Specifies the method used to evaluate accuracy. Can be "ot_plan", "k_nearest", or "category".
            Defaults to "ot_plan".

        fig_dir (Optional[str], optional):
            Directory path where the generated figure will be saved. If None, the figure will not be saved
            but displayed. Defaults to None.

        fig_name (str, optional):
            Name of the saved figure if `fig_dir` is specified. Defaults to "Accuracy_ot_plan.png".

        scatter (bool, optional):
            If True, the accuracy will be visualized as a swarm plot. Otherwise, a line plot will be used.
            Defaults to True.
    """
    # close all figures
    plt.clf()
    plt.close()

    # default setting
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(figsize=figsize)

    if scatter:
        df = align_representation._get_dataframe(eval_type, melt=True)
        sns.set_style("darkgrid")
        sns.set_palette("pastel")
        sns.swarmplot(data=df, x="top_n", y="matching rate", ax=ax, size=5, dodge=True)

    else:
        df = align_representation._get_dataframe(eval_type, melt=False)
        for group in df.columns:
            ax.plot(df.index, df[group], c="blue")

    ax.set_ylim(0, 100)
    ax.set_title(eval_type)
    ax.set_xlabel("top k")
    ax.set_ylabel("Matching rate (%)")
    # plt.legend(loc = "best")
    ax.tick_params(axis="both", which="major")

    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2)

    if fig_dir is None:
        fig_dir = os.path.join(align_representation.main_results_dir, "matching_rate")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

    plt.savefig(os.path.join(fig_dir, f"{fig_name}_{eval_type}.{fig_ext}"))

    if show_figure:
        plt.show()

    return ax
