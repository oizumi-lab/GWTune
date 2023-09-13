# Standard Library
import os
from typing import List, Optional

# Third Party Library
import matplotlib
import matplotlib.pyplot as plt

# Local Library
from ..align_representations import AlignRepresentations, PairwiseAnalysis
from .show_sim_mat import show_heatmap


def show_OT(
    align_representation: AlignRepresentations,
    OT_format: str = "default",
    ticks: Optional[str] = None,
    fig_dirs: Optional[List[str]] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Visualize the OT.

    Args:
        align_representation (AlignRepresentations):
            AlignRepresentations object.
        OT_format (str, optional):
            format of sim_mat to visualize.
            Options are "default", "sorted", and "both". Defaults to "default".
        ticks (str, optional):
            Determines how ticks should be displayed. Options are "objects", "category", or "numbers".
            Defaults to "number".
        fig_dir (Optional[str], optional):
            Directory to save the heatmap. If None, the heatmap won't be saved.

    Keyword Args:
        These keyword arguments are passed internally to `show_heatmap`.

        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (8, 6).
        title_size (int, optional): The size of the title. Defaults to 60.
        xlabel (Optional[str], optional): The label for the x-axis. Defaults to None.
        ylabel (Optional[str], optional): The label for the y-axis. Defaults to None.
        xlabel_size (int, optional): The size of the x-axis label. Defaults to 40.
        ylabel_size (int, optional): The size of the y-axis label. Defaults to 40.
        xticks_rotation (int, optional): The rotation of the x-axis ticks. Defaults to 90.
        yticks_rotation (int, optional): The rotation of the y-axis ticks. Defaults to 0.
        cbar_ticks_size (int, optional): The size of the colorbar ticks. Defaults to 20.
        xticks_size (int, optional): The size of the x-axis ticks. Defaults to 20.
        yticks_size (int, optional): The size of the y-axis ticks. Defaults to 20.
        cbar_format (Optional[str], optional): The format of the colorbar ticks. Defaults to None.
        cbar_label (Optional[str], optional): The label for the colorbar. Defaults to None.
        cbar_label_size (int, optional): The size of the colorbar label. Defaults to 20.
        cmap (str, optional): The colormap to use. Defaults to "cividis".
        ot_object_tick (bool, optional): Whether to use object labels for the ticks. Defaults to False.
        ot_category_tick (bool, optional): Whether to use category labels for the ticks. Defaults to False.
        draw_category_line (bool, optional): Whether to draw lines between categories. Defaults to False.
        category_line_alpha (float, optional): The alpha value of the category lines. Defaults to 0.2.
        category_line_style (str, optional): The style of the category lines. Defaults to "dashed".
        category_line_color (str, optional): The color of the category lines. Defaults to "C2".
        fig_ext (str, optional): The extension of the saved figure. Defaults to "png".
        show_figure (bool, optional): Whether to show the figure. Defaults to True.

    Returns:
        axes (List[matplotlib.axes.Axes]): heatmap of the similarity matrices.
    """

    # plot OTs
    if fig_dirs is None:
        fig_dirs = [None] * len(align_representation.pairwise_list)

    axes = []
    for pairwise, fig_dir in zip(align_representation.pairwise_list, fig_dirs):
        ax = show_OT_single(
            pairwise,
            OT_format=OT_format,
            title=f"$\Gamma$ ({pairwise.pair_name.replace('_', ' ')})",
            ticks=ticks,
            fig_dir=fig_dir,
            **kwargs,
        )
        axes.append(ax)

    return axes


def show_OT_single(
    pairwise: PairwiseAnalysis,
    OT_format: str = "default",
    title: Optional[str] = None,
    ticks: Optional[str] = None,
    fig_dir: Optional[str] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Visualize the OT for a single PairwiseAnalysis object.

    Args:
        pairwise (PairwiseAnalysis):
            PairwiseAnalysis object.
        OT_format (str, optional):
            format of sim_mat to visualize.
            Options are "default", "sorted", and "both". Defaults to "default".
        title (Optional[str], optional):
            The title of the heatmap. Defaults to None.
        ticks (Optional[str], optional):
            Determines how ticks should be displayed. Options are "objects", "category", or "numbers".
            Defaults to "number".
        fig_dir (Optional[str], optional):
            Directory to save the heatmap. If None, the heatmap won't be saved.

    Keyword Args:
        These keyword arguments are passed internally to `show_heatmap`.

    Returns:
        ax (matplotlib.axes.Axes): heatmap of the GWOT.
    """

    # figure path setting
    if fig_dir is None:
        fig_dir = pairwise.figure_path

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # plot OT
    fig_name = pairwise.data_name + "_" + pairwise.pair_name

    if OT_format == "default":
        ax = show_heatmap(
            pairwise.OT,
            title=title,
            ticks=ticks,
            category_name_list=None,
            num_category_list=None,
            object_labels=pairwise.source.object_labels,
            fig_name=fig_name,
            fig_dir=fig_dir,
            **kwargs,
        )

    elif OT_format == "sorted":
        ax = show_heatmap(
            pairwise.OT_sorted,
            title=title,
            ticks=ticks,
            category_name_list=pairwise.source.category_name_list,
            num_category_list=pairwise.source.num_category_list,
            object_labels=pairwise.source.object_labels,
            fig_name=fig_name,
            fig_dir=fig_dir,
            **kwargs,
        )

    else:
        raise ValueError("OT_format must be either 'default', 'sorted', or 'both'.")

    plt.clf()
    plt.close()
    return ax
