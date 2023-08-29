# visualizationをやるための関数を定義する
import copy
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import optuna
from dataclasses import dataclass
import ot
import seaborn as sns
import torch

from ..align_representations import AlignRepresentations

def show_sim_mat(
    align_representation: AlignRepresentations,
    fig_dir: Optional[str] = None,
    ticks: str = "number",
    sim_mat_format: str = "default",
    *,
    show_figure: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    title_size: int = 60,
    xlabel: str = "target",
    ylabel: str = "source",
    xlabel_size: int = 40,
    ylabel_size: int = 40,
    xticks_rotation: int = 90,
    yticks_rotation: int = 0,
    xticks_size: int = 20,
    yticks_size: int = 20,
    cbar_format: Optional[str] = None,
    cbar_ticks_size: int = 20,
    ot_object_tick: bool = False,
    ot_category_tick: bool = False,
    draw_category_line: bool = False,
    category_line_alpha: float = 0.2,
    category_line_style: str = "dashed",
    category_line_color: str = "C2",
    cmap: str = "cividis",
    aspect: str = "equal",
    fig_ext: str = "png"
) -> List[matplotlib.axes.Axes]:




def show_sim_mat(
    align_representation: AlignRepresentations,
    fig_dir: Optional[str] = None,
    ticks: str = "number",
    sim_mat_format: str = "default",
    *,
    show_figure: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    title_size: int = 60,
    xlabel: str = "target",
    ylabel: str = "source",
    xlabel_size: int = 40,
    ylabel_size: int = 40,
    xticks_rotation: int = 90,
    yticks_rotation: int = 0,
    xticks_size: int = 20,
    yticks_size: int = 20,
    cbar_format: Optional[str] = None,
    cbar_ticks_size: int = 20,
    ot_object_tick: bool = False,
    ot_category_tick: bool = False,
    draw_category_line: bool = False,
    category_line_alpha: float = 0.2,
    category_line_style: str = "dashed",
    category_line_color: str = "C2",
    cmap: str = "cividis",
    aspect: str = "equal",
    fig_ext: str = "png"
) -> matplotlib.axes.Axes:

    if fig_dir is None:
        fig_dir = align_representation.main_results_dir + "/" + align_representation.data_name + "/individual_sim_mat/"
        os.makedirs(fig_dir, exist_ok=True)

        for representation in self.representations_list:
            representation.show_sim_mat(
                sim_mat_format=sim_mat_format,
                visualization_config=visualization_config,
                fig_dir=fig_dir,
                ticks=ticks,
            )

            if show_distribution:
                representation.show_sim_mat_distribution(
                    **visualization_config_hist())


    if fig_dir is not None:
        fig_path = os.path.join(fig_dir, f"RDM_{align_representation.name}.{fig_ext}")
    else:
        fig_path = None

    if sim_mat_format == "default" or sim_mat_format == "both":
        sim_mat = align_representation.sim_mat
        title = align_representation.name
        category_name_list = None
        num_category_list = None

    elif sim_mat_format == "sorted" or sim_mat_format == "both":
        assert align_representation.category_idx_list is not None, "No label info to sort the 'sim_mat'."
        sim_mat = align_representation.sorted_sim_mat
        title = align_representation.name + "_sorted",
        category_name_list = align_representation.category_name_list
        num_category_list = align_representation.num_category_list

    else:
        raise ValueError("sim_mat_format must be either 'default', 'sorted', or 'both'.")

    ax = _show_sim_mat(
        sim_mat,
        save_file_name=fig_path,
        title=title,
        ticks=ticks,
        category_name_list=category_name_list,
        num_category_list=num_category_list,
        object_labels=align_representation.object_labels,
        # plot parameter setting
        show_figure=show_figure,
        figsize=figsize,
        title_size=title_size,
        xlabel=xlabel,
        ylabel=ylabel,
        xlabel_size=xlabel_size,
        ylabel_size=ylabel_size,
        xticks_rotation=xticks_rotation,
        yticks_rotation=yticks_rotation,
        xticks_size=xticks_size,
        yticks_size=yticks_size,
        cbar_format=cbar_format,
        cbar_ticks_size=cbar_ticks_size,
        ot_object_tick=ot_object_tick,
        ot_category_tick=ot_category_tick,
        draw_category_line=draw_category_line,
        category_line_alpha=category_line_alpha,
        category_line_style=category_line_style,
        category_line_color=category_line_color,
        cmap=cmap,
        aspect=aspect
    )
    return ax


def _show_sim_mat(
    matrix: Any,
    save_file_name: Optional[str],
    title: str,
    ticks: str,
    category_name_list: Optional[List[str]],
    num_category_list: Optional[List[int]],
    object_labels: Optional[List[str]],
    *,
    # plot parameter setting
    show_figure: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    title_size: int = 60,
    xlabel: str = "target",
    ylabel: str = "source",
    xlabel_size: int = 40,
    ylabel_size: int = 40,
    xticks_rotation: int = 90,
    yticks_rotation: int = 0,
    xticks_size: int = 20,
    yticks_size: int = 20,
    cbar_format: Optional[str] = None,
    cbar_ticks_size: int = 20,
    ot_object_tick: bool = False,
    ot_category_tick: bool = False,
    draw_category_line: bool = False,
    category_line_alpha: float = 0.2,
    category_line_style: str = "dashed",
    category_line_color: str = "C2",
    cmap: str = "cividis",
    aspect: str = "equal"
) -> None:
    """Display a heatmap of the given matrix with various customization options.

    Args:
        matrix (Any): The matrix to be visualized as a heatmap.
        title (str, optional): The title of the heatmap.
        save_file_name (str, optional): File name to save the heatmap. If None, the heatmap won't be saved.
        ticks (str, optional): Determines how ticks should be displayed. Options are "objects", "category", or "numbers".
        category_name_list (List[str], optional): List of category names if `ot_category_tick` is True.
        num_category_list (List[int], optional): List of the number of items in each category.
        object_labels (List[str], optional): Labels for individual objects, used if `ot_object_tick` is True.
        **kwargs: Other keyword arguments for customizing the heatmap.

    Raises:
        ValueError: If both `ot_object_tick` and `ot_category_tick` are True.
        ValueError: If `ticks` is "category" but `ot_category_tick` is False.

    Returns:
        None: Displays or saves the heatmap.
    """

    # Set up the graph style
    plt.style.use("default")
    plt.rcParams["grid.color"] = "black"

    fig, ax = plt.subplots(figsize = figsize)

    if title is not None:
        ax.set_title(title, size = title_size)

    aximg = ax.imshow(matrix, cmap = cmap, aspect = aspect)

    if ot_object_tick and ot_category_tick:
        raise(ValueError, "please turn off either 'ot_category_tick' or 'ot_object_tick'.")

    if not ot_object_tick and ot_category_tick:
        assert category_name_list is not None
        assert num_category_list is not None

        if ticks == "objects":
            plt.xticks(np.arange(sum(num_category_list)) + 0.5, labels = object_labels, rotation = xticks_rotation, size = xticks_size)
            plt.yticks(np.arange(sum(num_category_list)) + 0.5, labels = object_labels, rotation = yticks_rotation, size = yticks_size)

        elif ticks == "category":
            label_pos = [sum(num_category_list[:i + 1]) for i in range(len(category_name_list))]
            plt.xticks(label_pos, labels = category_name_list, rotation = xticks_rotation, size = xticks_size, fontweight = "bold")
            plt.yticks(label_pos, labels = category_name_list, rotation = yticks_rotation, size = yticks_size, fontweight = "bold")

            if draw_category_line:
                for pos in label_pos:
                    plt.axhline(pos, alpha = category_line_alpha, linestyle = category_line_style, color = category_line_color)
                    plt.axvline(pos, alpha = category_line_alpha, linestyle = category_line_style, color = category_line_color)

    if ot_object_tick and not ot_category_tick:

        if ticks == "numbers":
            plt.xticks(ticks = np.arange(len(matrix)) + 0.5, labels = np.arange(len(matrix)) + 1, size = xticks_size, rotation = xticks_rotation)
            plt.yticks(ticks = np.arange(len(matrix)) + 0.5, labels = np.arange(len(matrix)) + 1, size = yticks_size, rotation = yticks_rotation)

        elif ticks == "objects":
            assert object_labels is not None
            plt.xticks(ticks = np.arange(len(matrix)) + 0.5, labels = object_labels, size = xticks_size, rotation = xticks_rotation)
            plt.yticks(ticks = np.arange(len(matrix)) + 0.5, labels = object_labels, size = yticks_size, rotation = yticks_rotation)

        elif ticks == "category":
            raise(ValueError, "please use 'ot_category_tick = True'.")

    if not ot_object_tick and not ot_category_tick:
        plt.xticks([])
        plt.yticks([])

    plt.xlabel(xlabel, size = xlabel_size)
    plt.ylabel(ylabel, size = ylabel_size)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cbar = fig.colorbar(aximg, cax=cax, format = cbar_format)

    cbar.ax.tick_params(axis='y', labelsize = cbar_ticks_size)

    plt.tight_layout()

    if save_file_name is not None:
        plt.savefig(save_file_name)

    if return_figure:
        return ax

    else:
        plt.clf()
        plt.close()
