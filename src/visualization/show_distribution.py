# Standard Library
import os
from typing import Any, List, Optional, Tuple

# Third Party Library
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns

# Local Library
#%%
from ..align_representations import AlignRepresentations


def show_distribution(
    align_representation: AlignRepresentations,
    fig_dir: Optional[str] = None,
    **kwargs
) -> List[matplotlib.axes.Axes]:
    """Visualize the distribution of RDMs.

    Args:
        align_representation (AlignRepresentations):
            AlignRepresentations object.
        fig_dir (Optional[str], optional):
            Directory to save the heatmap. If None, the heatmap won't be saved.
            Defaults to None.

    Keyword Args:
        These keyword arguments are passed internally to `show_distribution_rep`.

        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (4, 3).
        title_size (int, optional): The size of the title. Defaults to 20.
        xlabel_size (int, optional): The size of the x-axis label. Defaults to 15.
        ylabel_size (int, optional): The size of the y-axis label.  Defaults to 15.
        alpha (float, optional): The transparency of the histogram. Defaults to 1..
        bins (int, optional): The number of bins. Defaults to 100.
        color (str, optional): The color of the histogram. Defaults to "C0".
        font_size (int, optional): The size of the font. Defaults to 20.
        fig_ext (str, optional): The extension of the figure. Defaults to "png".
        show_figure (bool, optional): Show the figure or not. Defaults to True.

    Returns:
        List[matplotlib.axes.Axes]: histgram of the similarity matrices.
    """

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
) -> matplotlib.axes.Axes:
    """Visualize the distribution of RDM of a single representation.

    Args:
        sim_mat (Any): The similarity matrix.
        title (str): The title of the figure.
        fig_name (Optional[str], optional): File name to save the figure. Defaults to None.
        fig_dir (Optional[str], optional): Directory to save the figure. Defaults to None.

    Keyword Args:
        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (4, 3).
        title_size (int, optional): The size of the title. Defaults to 20.
        xlabel_size (int, optional): The size of the x-axis label. Defaults to 15.
        ylabel_size (int, optional): The size of the y-axis label.  Defaults to 15.
        alpha (float, optional): The transparency of the histogram. Defaults to 1..
        bins (int, optional): The number of bins. Defaults to 100.
        color (str, optional): The color of the histogram. Defaults to "C0".
        font_size (int, optional): The size of the font. Defaults to 20.
        fig_ext (str, optional): The extension of the figure. Defaults to "png".
        show_figure (bool, optional): Show the figure or not. Defaults to True.

    Returns:
        matplotlib.axes.Axes: Histogram of the similarity matrix.
    """

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
