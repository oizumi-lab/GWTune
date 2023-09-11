# Standard Library
import os
from typing import List, Optional, Tuple

# Third Party Library
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Local Library
from ..align_representations import AlignRepresentations, PairwiseAnalysis


def show_optimization_log(
    align_representation: AlignRepresentations,
    fig_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[matplotlib.axes.Axes], List[matplotlib.axes.Axes]]:
    """Display dissimilarity matrices for a given AlignRepresentations object.

    Args:
        align_representation (AlignRepresentations): AlignRepresentations object.
        sim_mat_format (str, optional): _description_. Defaults to "default".
        ticks (str, optional): _description_. Defaults to "number".
        fig_dir (Optional[str], optional): _description_. Defaults to None.

    Keyword Args:
        These keyword arguments are passed internally to `show_heatmap`.
        For more details, please refer to the documentation.

    Returns:
        axis (List[matplotlib.axes.Axes]): heatmap of the similarity matrices.
    """

    # default setting
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use("seaborn-darkgrid")

    eps_gwd_axs, acc_gwd_axs = [], []

    for pairwise in align_representation.pairwise_list:
        eps_gwd_ax, acc_gwd_ax = show_optimization_log_rep(
            pairwise=pairwise, fig_dir=fig_dir, **kwargs
        )
        eps_gwd_axs.append(eps_gwd_ax)
        acc_gwd_axs.append(acc_gwd_ax)

    return eps_gwd_axs, acc_gwd_axs


def show_optimization_log_rep(
    pairwise: PairwiseAnalysis,
    fig_dir: Optional[str] = None,
    *,
    figsize: Tuple[int, int] = (8, 6),
    title_size: int = 20,
    xlabel_size: int = 20,
    ylabel_size: int = 20,
    xticks_rotation: int = 0,
    cbar_ticks_size: int = 20,
    xticks_size: int = 10,
    yticks_size: int = 10,
    cbar_format: Optional[str] = None,
    cbar_label_size: int = 20,
    cmap: str = "viridis",
    marker_size: int = 20,
    plot_eps_log: bool = False,
    lim_eps: Optional[Tuple[float, float]] = None,
    lim_gwd: Optional[Tuple[float, float]] = None,
    lim_acc: Optional[Tuple[float, float]] = None,
    fig_ext: str = "png",
    show_figure: bool = False,
) -> Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]:
    """Display a heatmap of the given matrix with various customization options.

    Args:
        matrix (Any): The matrix to be visualized as a heatmap.
        title (str, optional): The title of the heatmap.
        save_file_name (str, optional): File name to save the heatmap. If None, the heatmap won't be saved.
        ticks (str, optional): Determines how ticks should be displayed. Options are "objects", "category", or "numbers".
        category_name_list (List[str], optional): List of category names if `ot_category_tick` is True.
        num_category_list (List[int], optional): List of the number of items in each category.
        object_labels (List[str], optional): Labels for individual objects, used if `ot_object_tick` is True.

    Raises:
        ValueError: If both `ot_object_tick` and `ot_category_tick` are True.
        ValueError: If `ticks` is "category" but `ot_category_tick` is False.

    Returns:
        None: Displays or saves the heatmap.
    """

    # get trial history
    study = pairwise._run_optimization(compute_OT=False)
    df_trial = study.trials_dataframe()

    # figure plotting epsilon as x-axis and GWD as y-axis
    _, ax1 = plt.subplots(figsize=figsize)
    sc1 = ax1.scatter(
        df_trial["params_eps"],
        df_trial["value"],
        c=100 * df_trial["user_attrs_best_acc"],
        s=marker_size,
        cmap=cmap,
    )
    ax1.set_title(
        f"epsilon - GWD ({pairwise.pair_name.replace('_', ' ')})", fontsize=title_size
    )
    ax1.set_xlabel("epsilon", fontsize=xlabel_size)
    ax1.set_ylabel("GWD", fontsize=ylabel_size)

    if lim_eps is not None:
        ax1.set_xlim(lim_eps)

    if lim_gwd is not None:
        ax1.set_ylim(lim_gwd)

    if plot_eps_log:
        ax1.set_xscale("log")

    ax1.tick_params(
        axis="x", which="both", labelsize=xticks_size, rotation=xticks_rotation
    )
    ax1.tick_params(axis="y", which="major", labelsize=yticks_size)

    ax1.grid(True, which="both")
    cbar = plt.colorbar(sc1, ax=ax1)
    cbar.set_label(label="Matching Rate (%)", size=cbar_label_size)
    cbar.ax.tick_params(labelsize=cbar_ticks_size)

    plt.tight_layout()

    if fig_dir is None:
        fig_dir = pairwise.figure_path

    # save figure
    plt.savefig(
        os.path.join(fig_dir, f"Optim_log_eps_GWD_{pairwise.pair_name}.{fig_ext}")
    )

    # show figure
    if show_figure:
        plt.show()

    # figure plotting accuracy as x-axis and GWD as y-axis
    if plot_eps_log:
        norm = LogNorm

    else:
        norm = None

    _, ax2 = plt.subplots(figsize=figsize)
    sc2 = ax2.scatter(
        100 * df_trial["user_attrs_best_acc"],
        df_trial["value"].values,
        c=df_trial["params_eps"],
        cmap=cmap,
        norm=norm,
    )
    ax2.set_title(
        f"Matching Rate - GWD ({pairwise.pair_name.replace('_', ' ')})",
        fontsize=title_size,
    )
    ax2.set_xlabel("Matching Rate (%)", fontsize=xlabel_size)
    ax2.tick_params(axis="x", labelsize=xticks_size)
    ax2.set_ylabel("GWD", fontsize=ylabel_size)
    ax2.tick_params(axis="y", labelsize=yticks_size)

    if lim_acc is not None:
        ax2.set_xlim(lim_acc)

    if lim_gwd is not None:
        ax2.set_ylim(lim_gwd)

    ax2.grid(True)
    cbar = plt.colorbar(sc2, ax=ax2, format=cbar_format, norm=norm)
    cbar.set_label(label="epsilon", size=cbar_label_size)
    cbar.ax.tick_params(labelsize=cbar_ticks_size)

    plt.tight_layout()

    plt.savefig(os.path.join(fig_dir, f"acc_gwd_eps({pairwise.pair_name}).{fig_ext}"))

    if show_figure:
        plt.show()

    return ax1, ax2
