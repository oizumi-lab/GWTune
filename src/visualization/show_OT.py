# Standard Library
import os
from typing import List, Optional

# Third Party Library
import matplotlib

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
        align_representation (AlignRepresentations): AlignRepresentations object.
        sim_mat_format (str, optional): _description_. Defaults to "default".
        ticks (str, optional): _description_. Defaults to "number".
        fig_dir (Optional[str], optional): _description_. Defaults to None.

        title (str, optional):
            the title of OT figure.
            Defaults to None. If None, this will be automatically defined.

        OT_format (str, optional):
            format of sim_mat to visualize.
            Options are "default", "sorted", and "both". Defaults to "default".

            return_data (bool, optional):
            return the computed OT. Defaults to False.

        return_figure (bool, optional):
            make the result figures or not. Defaults to True.

        visualization_config (VisualizationConfig, optional):
            container of parameters used for figure. Defaults to VisualizationConfig().

        fig_dir (Optional[str], optional):
            you can define the path to which you save the figures (.png).
            If None, the figures will be saved in the same subfolder in "results_dir". Defaults to None.

        ticks (Optional[str], optional):
            you can use "objects" or "category (if existed)" or "None". Defaults to None.

    Returns:
        OT : the result of GWOT or sorted OT. This depends on OT_format.

    Raises:
        ValueError: If an invalid OT_format is provided.
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

    return ax
