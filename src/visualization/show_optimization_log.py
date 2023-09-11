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
    """Show both the relationships between epsilons and GWD, and between accuracy and GWD.

    Args:
        align_representation (AlignRepresentations):
            AlignRepresentations object.
        fig_dir (Optional[str], optional):
            Directory to save the heatmap. If None, the heatmap won't be saved.

    Keyword Args:
        These keyword arguments are passed internally to `show_optimization_log_rep`.

        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (8, 6).
        title_size (int, optional): The size of the title. Defaults to 20.
        xlabel_size (int, optional): The size of the x-axis label. Defaults to 20.
        ylabel_size (int, optional): The size of the y-axis label. Defaults to 20.
        xticks_rotation (int, optional): The rotation of the x-axis ticks. Defaults to 0.
        cbar_ticks_size (int, optional): The size of the colorbar ticks. Defaults to 20.
        xticks_size (int, optional): The size of the x-axis ticks. Defaults to 10.
        yticks_size (int, optional): The size of the y-axis ticks. Defaults to 10.
        cbar_format (Optional[str], optional): The format of the colorbar ticks. Defaults to None.
        cbar_label_size (int, optional): The size of the colorbar label. Defaults to 20.
        cmap (str, optional): The colormap to use. Defaults to "viridis".
        marker_size (int, optional): The size of the markers. Defaults to 20.
        plot_eps_log (bool, optional): Whether to plot epsilon in log scale. Defaults to False.
        lim_eps (Optional[Tuple[float, float]], optional): The limits of the range of epsilon. Defaults to None.
        lim_gwd (Optional[Tuple[float, float]], optional): The limits of the range of GWD. Defaults to None.
        lim_acc (Optional[Tuple[float, float]], optional): The limits of the range of accuracy. Defaults to None.
        fig_ext (str, optional): The extension of the saved figure. Defaults to "png".
        show_figure (bool, optional): Whether to show the figure. Defaults to True.

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
        pairwise (PairwiseAnalysis):
            PairwiseAnalysis object.
        fig_dir (Optional[str], optional):
            Directory to save the heatmap. If None, the heatmap won't be saved.

    Keyword Args:
        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (8, 6).
        title_size (int, optional): The size of the title. Defaults to 20.
        xlabel_size (int, optional): The size of the x-axis label. Defaults to 20.
        ylabel_size (int, optional): The size of the y-axis label. Defaults to 20.
        xticks_rotation (int, optional): The rotation of the x-axis ticks. Defaults to 0.
        cbar_ticks_size (int, optional): The size of the colorbar ticks. Defaults to 20.
        xticks_size (int, optional): The size of the x-axis ticks. Defaults to 10.
        yticks_size (int, optional): The size of the y-axis ticks. Defaults to 10.
        cbar_format (Optional[str], optional): The format of the colorbar ticks. Defaults to None.
        cbar_label_size (int, optional): The size of the colorbar label. Defaults to 20.
        cmap (str, optional): The colormap to use. Defaults to "viridis".
        marker_size (int, optional): The size of the markers. Defaults to 20.
        plot_eps_log (bool, optional): Whether to plot epsilon in log scale. Defaults to False.
        lim_eps (Optional[Tuple[float, float]], optional): The limits of the range of epsilon. Defaults to None.
        lim_gwd (Optional[Tuple[float, float]], optional): The limits of the range of GWD. Defaults to None.
        lim_acc (Optional[Tuple[float, float]], optional): The limits of the range of accuracy. Defaults to None.
        fig_ext (str, optional): The extension of the saved figure. Defaults to "png".
        show_figure (bool, optional): Whether to show the figure. Defaults to True.

    Returns:
        ax1 (matplotlib.axes.Axes): The axes of the epsilon-GWD figure.
        ax2 (matplotlib.axes.Axes): The axes of the matching_rate-GWD figure.
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
