import os
from typing import Any, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..align_representations import AlignRepresentations, PairwiseAnalysis


def show_OT(
    align_representation: AlignRepresentations,
    OT_format: str = "default",
    fig_dir: Optional[str] = None,
    ticks: Optional[str] = None
) -> matplotlib.axes.Axes :
    """Visualize the OT.

    Args:
        ot_to_plot (Optional[np.ndarray], optional):
            the OT to visualize. Defaults to None.
            If None, the OT computed as GWOT will be used.

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

    # get ot list
    if OT_format == "default":
        ot_to_plot_list = align_representation.OT_list
    elif OT_format == "sorted":
        ot_to_plot_list = align_representation.OT_sorted_list

    if fig_dir is None:
        fig_dir = align_representation.figure_path

    # get cate

    # plot OTs
    axes = []
    for OT in ot_to_plot_list:
        ax = show_OT_single(
            OT,
            fig_dir=fig_dir,
            )
        axes.append(ax)

    return axes


def show_OT_single(
    pairwise: PairwiseAnalysis,
    title: Optional[str] = None,
    fig_dir: Optional[str] = None,
    ticks: Optional[str] = None,
):

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    for pairwise, OT in zip(align_representation.pairwise_list, align_representation.OT_list):
        if OT_format == "sorted" or OT_format == "both":
        assert pairwise.source.sorted_sim_mat is not None, "No label info to sort the 'sim_mat'."
        OT_sorted = pairwise.source.func_for_sort_sim_mat(
            OT,
            category_idx_list=pairwise.source.category_idx_list
        )

    if return_figure:
        save_file = self.data_name + "_" + self.pair_name
        if fig_dir is not None:
            fig_ext=visualization_config.visualization_params["fig_ext"]
            fig_path = os.path.join(fig_dir, f"{save_file}.{fig_ext}")
        else:
            fig_path = None

        if OT_format == "default" or OT_format == "both":
            if OT_format == "default":
                assert self.source.category_name_list is None, "please set the 'sim_mat_format = sorted'. "

            visualize_functions.show_heatmap(
                ot_to_plot,
                title=title,
                save_file_name=fig_path,
                ticks=ticks,
                category_name_list=None,
                num_category_list=None,
                object_labels=self.source.object_labels,
                **visualization_config(),
            )

        elif OT_format == "sorted" or OT_format == "both":
            visualize_functions.show_heatmap(
                OT_sorted,
                title=title,
                save_file_name=fig_path,
                ticks=ticks,
                category_name_list=self.source.category_name_list,
                num_category_list=self.source.num_category_list,
                object_labels=self.source.object_labels,
                **visualization_config(),
            )

        else:
            raise ValueError("OT_format must be either 'default', 'sorted', or 'both'.")

    if return_data:
        if OT_format == "default":
            return ot_to_plot

        elif OT_format == "sorted":
            return OT_sorted

        elif OT_format == "both":
            return ot_to_plot, OT_sorted

        else:
            raise ValueError("OT_format must be either 'default', 'sorted', or 'both'.")
