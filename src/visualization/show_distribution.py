import os
from typing import Any, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
#%%
from ..align_representations import AlignRepresentations, Representation


def show_distribution(
    align_representation: AlignRepresentations,
    fig_dir: Optional[str] = None,
    **kwargs
) -> Tuple[matplotlib.axes.Axes]:

    if fig_dir is None:
        fig_dir = align_representation.main_results_dir + "/individual_distribution/"
        os.makedirs(fig_dir, exist_ok=True)

    axis = []
    for representation in align_representation.representations_list:
        ax = show_distribution_rep(
            sim_mat=representation.sim_mat,
            title=f"Distribution of RDM ({representation.name})",
            fig_name=f"distribution_{representation.name}",
            fig_dir=fig_dir,
            **kwargs
        )
        axis.append(ax)

    return axis


def show_distribution_rep(
    sim_mat: Any,
    title: str,
    fig_name: Optional[str] = None,
    fig_dir: Optional[str] = None,
    *,
    figsize: Tuple[int, int] = (4, 3),
    title_size: int = 20,
    xlabel_size: int = 15,
    ylabel_size: int = 15,
    alpha: float = 1.,
    bins: int = 100,
    color: str = "C0",
    font_size: int = 20,
    fig_ext: str = "png",
    show_figure: bool = True,
):

    sim_vector = sp.spatial.distance.squareform(sim_mat, checks=False)

    # default setting
    plt.rcParams.update(plt.rcParamsDefault)
    sns.set(style='darkgrid')
    plt.rcParams["font.size"] = font_size

    # plot
    _, ax = plt.subplots(figsize=figsize)
    sns.histplot(sim_vector, alpha=alpha, bins=bins, color=color)

    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel("RDM value", fontsize=xlabel_size)
    ax.set_ylabel("Count", fontsize=ylabel_size)

    if fig_dir is not None:
        plt.savefig(os.path.join(fig_dir, fig_name + "." + fig_ext))

    if show_figure:
        plt.show()

    return ax
