# Standard Library
import os
from typing import Any, List, Optional, Tuple

# Third Party Library
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Local Library
from ..align_representations import AlignRepresentations, Representation


def show_sim_mat(
    align_representation: AlignRepresentations,
    sim_mat_format: str = "default",
    ticks: str = "number",
    fig_dir: Optional[str] = None,
    **kwargs
) -> List[matplotlib.axes.Axes]:
    """Display dissimilarity matrices for a given AlignRepresentations object.

    Args:
        align_representation (AlignRepresentations):
            AlignRepresentations object.
        sim_mat_format (str, optional):
            "default", "sorted", or "both". Defaults to "default".
        ticks (str, optional):
            Determines how ticks should be displayed. Options are "objects", "category", or "numbers".
            Defaults to "number".
        fig_dir (Optional[str], optional):
            Directory to save the heatmap. If None, the heatmap won't be saved.

    Keyword Args:
        These keyword arguments are passed internally to `visualization.show_heatmap`.

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

    if fig_dir is None:
        fig_dir = os.path.join(
            align_representation.main_results_dir,
            "individual_sim_mat",
            align_representation.config.init_mat_plan
        )
        os.makedirs(fig_dir, exist_ok=True)

    axes = []
    for representation in align_representation.representations_list:
        ax = show_sim_mat_rep(
            representation=representation,
            sim_mat_format=sim_mat_format,
            ticks=ticks,
            fig_dir=fig_dir,
            **kwargs
        )
        axes.append(ax)

    return axes


def show_sim_mat_rep(
    representation: Representation,
    sim_mat_format: str = "default",
    ticks: str = "number",
    fig_dir: Optional[str] = None,
    **kwargs
) -> matplotlib.axes.Axes:
    """Display a heatmap of the given matrix with various customization options.

    Args:
        representation (Representation):
            The representation to be visualized.
        sim_mat_format (str, optional):
            "default", "sorted", or "both". Defaults to "default".
            Default to "default".
        ticks (str, optional):
            Determines how ticks should be displayed. Options are "objects", "category", or "numbers".
            Defaults to "number".
        fig_dir (Optional[str], optional):
            Directory to save the heatmap. If None, the heatmap won't be saved.

    Keyword Args:
        These keyword arguments are passed internally to `visualization.show_heatmap`.

    Returns:
        ax (matplotlib.axes.Axes): heatmap of the similarity matrices.
    """

    if sim_mat_format == "default" or sim_mat_format == "both":
        ax = show_heatmap(
            representation.sim_mat,
            title=representation.name,
            ticks=ticks,
            category_name_list=None,
            num_category_list=None,
            object_labels=representation.object_labels,
            fig_name=f"RDM_{representation.name}_default",
            fig_dir=fig_dir,
            **kwargs
        )
        return ax

    elif sim_mat_format == "sorted" or sim_mat_format == "both":

        assert representation.category_idx_list is not None, "No label info to sort the 'sim_mat'."
        ax = show_heatmap(
            representation.sorted_sim_mat,
            title=representation.name,
            ticks=ticks,
            category_name_list=representation.category_name_list,
            num_category_list=representation.num_category_list,
            object_labels=representation.object_labels,
            fig_name=f"RDM_{representation.name}_sorted",
            fig_dir=fig_dir,
            **kwargs
        )
        return ax

    else:
        raise ValueError("sim_mat_format must be either 'default', 'sorted', or 'both'.")


def show_heatmap(
    sim_mat: Any,
    title: Optional[str],
    ticks: Optional[str] = None,
    category_name_list: Optional[List[str]] = None,
    num_category_list: Optional[List[int]] = None,
    object_labels: Optional[List[str]] = None,
    fig_name: Optional[str] = None,
    fig_dir: Optional[str] = None,
    *,
    figsize: Tuple[int, int] = (8, 6),
    title_size: int = 60,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel_size: int = 40,
    ylabel_size: int = 40,
    xticks_rotation: int = 90,
    yticks_rotation: int = 0,
    cbar_ticks_size: int = 20,
    xticks_size: int = 20,
    yticks_size: int = 20,
    cbar_format: Optional[str] = None,
    cbar_label: Optional[str] = None,
    cbar_label_size: int = 20,
    cmap: str = "cividis",
    ot_object_tick: bool = False,
    ot_category_tick: bool = False,
    draw_category_line: bool = False,
    category_line_alpha: float = 0.2,
    category_line_style: str = "dashed",
    category_line_color: str = "C2",
    fig_ext: str = "png",
    show_figure: bool = True,
) -> matplotlib.axes.Axes:
    """Display a heatmap of the given matrix with various customization options.

    Args:
        sim_mat (Any): The matrix to be visualized as a heatmap.
        title (str, optional): The title of the heatmap.
        ticks (str, optional): Determines how ticks should be displayed. Options are "objects", "category", or "numbers".
        category_name_list (List[str], optional): List of category names if `ot_category_tick` is True.
        num_category_list (List[int], optional): List of the number of items in each category.
        object_labels (List[str], optional): Labels for individual objects, used if `ot_object_tick` is True.
        fig_name (Optional[str], optional): File name to save the heatmap. If None, the heatmap won't be saved.
        fig_dir (Optional[str], optional): Directory to save the heatmap. If None, the heatmap won't be saved.

    Keyword Args:
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
    """

    # Set up the graph style
    plt.style.use("default")
    plt.rcParams["grid.color"] = "black"

    fig, ax = plt.subplots(figsize = figsize)

    if title is not None:
        ax.set_title(title, size = title_size)

    aximg = ax.imshow(sim_mat, cmap=cmap, aspect='equal')

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
            plt.xticks(ticks = np.arange(len(sim_mat)) + 0.5, labels = np.arange(len(sim_mat)) + 1, size = xticks_size, rotation = xticks_rotation)
            plt.yticks(ticks = np.arange(len(sim_mat)) + 0.5, labels = np.arange(len(sim_mat)) + 1, size = yticks_size, rotation = yticks_rotation)

        elif ticks == "objects":
            assert object_labels is not None
            plt.xticks(ticks = np.arange(len(sim_mat)) + 0.5, labels = object_labels, size = xticks_size, rotation = xticks_rotation)
            plt.yticks(ticks = np.arange(len(sim_mat)) + 0.5, labels = object_labels, size = yticks_size, rotation = yticks_rotation)

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
    cbar.set_label(cbar_label, size = cbar_label_size)
    cbar.ax.tick_params(axis='y', labelsize = cbar_ticks_size)

    plt.tight_layout()

    if fig_dir is not None:
        plt.savefig(os.path.join(fig_dir, fig_name + "." + fig_ext), bbox_inches='tight', dpi=300)

    if show_figure:
        plt.show()

    plt.clf()
    plt.close()
    return ax
