# Standard Library
import colorsys
import os
from typing import List, Optional, Tuple

# Third Party Library
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Local Library
from ..align_representations import AlignRepresentations


def plot_embedding(
    align_representation: AlignRepresentations,
    dim: int,
    category_name_list: Optional[List[str]] = None,
    num_category_list: Optional[List[int]] = None,
    title: Optional[str] = None,
    legend: bool = True,
    fig_name: str = "Aligned_embedding",
    fig_dir: Optional[str] = None,
    **kwargs
) -> matplotlib.axes.Axes:
    """Plot the aligned embedding.

    Args:
        align_representation (AlignRepresentations):
            AlignRepresentations object.
        dim (int):
            Dimension of the embedding. Can be 2 or 3.
        category_name_list (Optional[List[str]], optional):
            List of category names. Defaults to None.
        num_category_list (Optional[List[int]], optional):
            List of number of categories. Defaults to None.
        title (Optional[str], optional):
            Title of the plot. Defaults to None.
        legend (bool, optional):
            Whether to show the legend. Defaults to True.
        fig_name (str, optional):
            Name of the saved figure if `fig_dir` is specified. Defaults to "Aligned_embedding".
        fig_dir (Optional[str], optional):
            Directory to save the plot. If None, the figure won't be saved. Defaults to None.

    Keyword Args:
        These keyword arguments are passed internally to `_plot_embedding`.


    Returns:
        ax (matplotlib.axes.Axes): embedding plot.
    """

    if category_name_list is None:
        if align_representation.representations_list[0].category_name_list is not None:
            category_name_list = align_representation.representations_list[0].category_name_list
            num_category_list = align_representation.representations_list[0].num_category_list

    name_list = []
    for representation in align_representation.representations_list:
        name_list.append(representation.name)

    if fig_dir is None:
        fig_dir = os.path.join(align_representation.main_results_dir, "visualize_embedding")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

    ax = _plot_embedding(
        align_representation.low_embedding_list,
        dim=dim,
        name_list=name_list,
        category_name_list=category_name_list,
        num_category_list=num_category_list,
        title=title,
        has_legend=legend,
        fig_name=fig_name,
        fig_dir=fig_dir,
        **kwargs
    )
    return ax

def _plot_embedding(
    embedding_list: List[np.ndarray],
    dim: int,
    name_list: List[str],
    category_name_list: Optional[List[str]] = None,
    num_category_list: Optional[List[int]] = None,
    title: Optional[str] = None,
    has_legend: bool = True,
    fig_name: str = "Aligned_embedding",
    fig_dir: Optional[str] = None,
    *,
    figsize: Tuple[int, int] = (15, 15),
    xlabel: str = "PC1",
    ylabel: str = "PC2",
    zlabel: str = "PC3",
    title_size: int = 60,
    xlabel_size: int = 25,
    ylabel_size: int = 25,
    zlabel_size: int = 25,
    legend_size: Optional[int] = None,
    color_labels: Optional[List[str]] = None,
    color_hue: Optional[str] = None,
    colorbar_label: Optional[str] = None,
    colorbar_range: List[int] = [0, 1],
    colorbar_shrink: float = 0.8,
    markers_list: Optional[List[str]] = None,
    marker_size: int = 30,
    cmap: str = "viridis",
    alpha: float = 1,
    font: str = "Arial",
    fig_ext: str = "png",
    show_figure: bool = True
) -> matplotlib.axes.Axes:
    """Plot the aligned embedding.

    Args:
        embedding_list (List[np.ndarray]):
            low-dimensional embedding list
        dim (int):
            Dimension of the embedding. Can be 2 or 3.
        name_list (List[str]):
            List of names of the embedding/representation.
        category_name_list (Optional[List[str]], optional):
            List of category names. Defaults to None.
        num_category_list (Optional[List[int]], optional):
            List of number of categories. Defaults to None.
        title (Optional[str], optional):
            Title of the plot. Defaults to None.
        has_legend (bool, optional):
            Whether to show the legend. Defaults to True.
        fig_name (str, optional):
            Name of the saved figure if `fig_dir` is specified. Defaults to "Aligned_embedding".
        fig_dir (Optional[str], optional):
            Directory to save the plot. If None, the figure won't be saved. Defaults to None.

    Keyword Args:
        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (15, 15).
        xlabel (str, optional): The label of the x-axis. Defaults to "PC1".
        ylabel (str, optional): The label of the y-axis. Defaults to "PC2".
        zlabel (str, optional): The label of the z-axis. Defaults to "PC3".
        title_size (int, optional): The size of the title. Defaults to 60.
        xlabel_size (int, optional): The size of the x-axis label. Defaults to 25.
        ylabel_size (int, optional): The size of the y-axis label. Defaults to 25.
        zlabel_size (int, optional): The size of the z-axis label. Defaults to 25.
        legend_size (Optional[int], optional): The size of the legend. Defaults to None.
        color_labels (Optional[List[str]], optional): The color labels. Defaults to None.
        color_hue (Optional[str], optional): The hue of the color labels. Defaults to None.
        colorbar_label (Optional[str], optional): The label of the colorbar. Defaults to None.
        colorbar_range (List[int], optional): The range of the colorbar. Defaults to [0, 1].
        colorbar_shrink (float, optional): The shrink of the colorbar. Defaults to 0.8.
        markers_list (Optional[List[str]], optional): The list of markers. Defaults to None.
        marker_size (int, optional): The size of the markers. Defaults to 30.
        cmap (str, optional): The colormap to use. Defaults to "viridis".
        alpha (float, optional): The transparency of the markers. Defaults to 1.
        font (str, optional): The font to use. Defaults to "Arial".
        fig_ext (str, optional): The extension of the saved figure. Defaults to "png".
        show_figure (bool, optional): Whether to show the figure. Defaults to True.

    Raises:
        ValueError: `dim` must be 2 or 3.

    Returns:
        matplotlib.axes.Axes: embedding plot.
    """

    if color_labels is None:

        if num_category_list is None:
            color_labels = get_color_labels(
                embedding_list[0].shape[0],
                hue=color_hue,
                show_labels=False
            )

        else:
            color_labels, main_colors = get_color_labels_for_category(
                num_category_list,
                min_saturation = 1,
                show_labels = False
            )

    if markers_list is None:
        markers_list = ['o', 'x', '^', 's', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'D', 'd', '.', ',', '1', '2', '3', '4', '_', '|'][:len(embedding_list)]

    plt.style.use("default")
    plt.rcParams["grid.color"] = "black"
    plt.rcParams['font.family'] = font
    fig = plt.figure(figsize = figsize)

    # Set the axis
    if dim == 3:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_xlabel(xlabel, fontsize = xlabel_size)
        ax.set_ylabel(ylabel, fontsize = ylabel_size)
        ax.set_zlabel(zlabel, fontsize = zlabel_size)
        ax.view_init(elev = 30, azim = 60)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.w_xaxis.gridlines.set_color('black')
        ax.w_yaxis.gridlines.set_color('black')
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.set_ticklabels([])
        ax.axes.get_zaxis().set_visible(True)
        ax.w_zaxis.gridlines.set_color('black')
        ax.zaxis.pane.set_edgecolor('w')

    elif dim == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(xlabel, fontsize = xlabel_size)
        ax.set_ylabel(ylabel, fontsize = ylabel_size)

    else:
        raise ValueError("'dim' is either 2 or 3")

    ax.grid(True)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.axes.get_xaxis().set_visible(True)
    ax.axes.get_yaxis().set_visible(True)

    # Plot the embedding
    for i in range(len(embedding_list)):
        coords_i = embedding_list[i]
        if dim == 3:
            im = ax.scatter(
                xs = coords_i[:, 0],
                ys = coords_i[:, 1],
                zs = coords_i[:, 2],
                marker = markers_list[i],
                color = color_labels,
                s = marker_size,
                alpha = alpha,
                cmap=cmap,
            )

            ax.scatter([], [], [], marker = markers_list[i], color = "black", s = marker_size, alpha = 1, label = name_list[i].replace("_", " "))

        else:
            im = ax.scatter(
                x = coords_i[:, 0],
                y = coords_i[:, 1],
                marker = markers_list[i],
                color = color_labels,
                s = marker_size,
                alpha = alpha,
                cmap=cmap,
            )

            ax.scatter(x = [], y = [], marker = markers_list[i], color = "black", s = marker_size, alpha = 1, label = name_list[i].replace("_", " "))

    if category_name_list is not None:
        for i, category in enumerate(category_name_list):
            if dim == 3:
                ax.scatter([], [], [], marker = "o", color = main_colors[i], s = marker_size, alpha = 1, label = category)

            else:
                ax.scatter(x = [], y = [], marker = "o", color = main_colors[i], s = marker_size, alpha = 1, label = category)
    ax.set_axisbelow(True)

    if has_legend:
        ax.legend(fontsize = legend_size, loc = "best")

    if title is not None:
        plt.title(title, fontsize = title_size)

    if colorbar_label is not None:
        im.set_cmap(cmap)
        cbar = plt.colorbar(im, shrink=colorbar_shrink, ax=ax)
        cbar.set_label(colorbar_label, size=xlabel_size)
        cbar.ax.tick_params(labelsize=xlabel_size)
        cbar.mappable.set_clim(colorbar_range[0], colorbar_range[1])

    if fig_dir is not None:
        fig_path = os.path.join(fig_dir, f"{fig_name}.{fig_ext}")
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)

    if show_figure:
        plt.show()

    plt.clf()
    plt.close()
    return ax


def get_color_labels(
    n: int,
    hue: Optional[str] = None,
    show_labels: bool = True
) -> List[Tuple[float, float, float]]:
    """Create color labels for n objects

    Args:
        n (int): number of objects
        hue (str): "warm", "cool", or None
        show_labels (bool): whether to show color labels

    Returns:
        color_labels (List): color labels for each objects
    """

    # Set the saturation and lightness values to maximum
    saturation = 1.0
    lightness = 0.5

    if hue == "warm":
        hue_list = np.linspace(-0.2, 0.1, n)
    elif hue == "cool":
        hue_list = np.linspace(0.5, 0.8, n)
    else:
        hue_list = np.linspace(0, 1, n, endpoint = False)

    # Create a list to store the color labels
    color_labels = []

    # Generate the color labels
    for i in range(n):
        hue = hue_list[i]
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        color_labels.append((r, g, b))

    if show_labels:
        # Show color labels
        plt.figure(figsize=(10, 5))
        plt.scatter(np.linspace(0, n-1, n), np.zeros((n,)), c = color_labels)
        plt.xticks(ticks = np.linspace(0, n-1, n))
        plt.title("Show color labels")
        plt.show()

    return color_labels


def get_color_labels_for_category(
    n_category_list: List[int],
    min_saturation: int,
    show_labels: bool = True
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """Create color labels for n categoris

    Args:
        n_category_list (List): number of objects in each category
        min_saturation (int): minimum saturation

    Returns:
        color_labels : color labels for each objects
        main_colors : list of representative color labels for n categories
    """

    # Set the saturation and lightness values to maximum
    lightness = 0.5

    # Calculate the hue step size
    hue_list = np.linspace(0, 1, len(n_category_list), endpoint=False)

    # Create a list to store the color labels
    color_labels = []
    main_colors = []

    # Generate the color labels
    for j, n_category in enumerate(n_category_list):
        saturation = np.linspace(min_saturation, 1, n_category)
        for i in range(n_category):
            hue = hue_list[j]
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation[i])
            color_labels.append((r, g, b))
        r, g, b = colorsys.hls_to_rgb(hue, lightness, 1)
        main_colors.append((r, g, b))

    if show_labels:
        # Show color labels
        n_category = sum(n_category_list)
        plt.figure(figsize=(10, 5))
        plt.scatter(np.linspace(0, n_category-1, n_category), np.zeros((n_category,)), c = color_labels)
        plt.xticks(ticks = np.linspace(0, n_category-1, n_category))
        plt.title("Show color labels")
        plt.show()

    return color_labels, main_colors
