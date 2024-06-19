import colorsys
from typing import Any, List, Tuple, Optional
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, MDS


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


def show_heatmap(
    matrix: Any,
    title: Optional[str],
    save_file_name: Optional[str] = None,
    ticks: Optional[str] = None,
    category_name_list: Optional[List[str]] = None,
    num_category_list: Optional[List[int]] = None,
    x_object_labels: Optional[List[str]] = None,
    y_object_labels: Optional[List[str]] = None,
    **kwargs
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
    figsize = kwargs.get('figsize', (8, 6))
    title_size = kwargs.get('title_size', 60)

    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)

    xlabel_size = kwargs.get('xlabel_size', 40)
    ylabel_size = kwargs.get('ylabel_size', 40)

    xticks_rotation = kwargs.get('xticks_rotation', 90)
    yticks_rotation = kwargs.get('yticks_rotation', 0)

    cbar_ticks_size = kwargs.get("cbar_ticks_size", 20)
    xticks_size = kwargs.get('xticks_size', 20)
    yticks_size = kwargs.get('yticks_size', 20)
    cbar_format = kwargs.get('cbar_format', None)#"%.2e"
    cbar_label = kwargs.get('cbar_label', None)
    cbar_label_size = kwargs.get('cbar_label_size', 20)
    cmap = kwargs.get('cmap', 'cividis')

    ot_object_tick = kwargs.get("ot_object_tick", False)
    ot_category_tick = kwargs.get("ot_category_tick", False)

    draw_category_line  = kwargs.get('draw_category_line', False)
    category_line_alpha = kwargs.get('category_line_alpha', 0.2)
    category_line_style = kwargs.get('category_line_style', 'dashed')
    category_line_color = kwargs.get('category_line_color', 'C2')

    font = kwargs.get('font', 'Noto Sans CJK JP')
    show_figure = kwargs.get('show_figure', True)

    plt.style.use("default")
    plt.rcParams["grid.color"] = "black"
    plt.rcParams['font.family'] = font

    fig, ax = plt.subplots(figsize = figsize)

    if title is not None:
        ax.set_title(title, size = title_size)

    aximg = ax.imshow(matrix, cmap=cmap, aspect='equal')

    if ot_object_tick and ot_category_tick:
        raise(ValueError, "please turn off either 'ot_category_tick' or 'ot_object_tick'.")

    if not ot_object_tick and ot_category_tick:
        assert category_name_list is not None
        assert num_category_list is not None

        if ticks == "objects":
            plt.xticks(np.arange(sum(num_category_list)) + 0.5, labels = x_object_labels, rotation = xticks_rotation, size = xticks_size)
            plt.yticks(np.arange(sum(num_category_list)) + 0.5, labels = y_object_labels, rotation = yticks_rotation, size = yticks_size)

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
            # plt.xticks(ticks = np.arange(len(matrix)) + 0.5, labels = np.arange(len(matrix)) + 1, size = xticks_size, rotation = xticks_rotation)
            # plt.yticks(ticks = np.arange(len(matrix)) + 0.5, labels = np.arange(len(matrix)) + 1, size = yticks_size, rotation = yticks_rotation)
            pass
        elif ticks == "objects":
            # assert object_labels is not None
            plt.xticks(ticks = np.arange(len(x_object_labels)), labels = x_object_labels, size = xticks_size, rotation = xticks_rotation)
            plt.yticks(ticks = np.arange(len(y_object_labels)), labels = y_object_labels, size = yticks_size, rotation = yticks_rotation)
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

    if save_file_name is not None:
        plt.savefig(save_file_name)

    if show_figure:
        plt.show()

    plt.clf()
    plt.close()


def plot_lower_triangular_histogram(matrix: Any, title: str) -> None:
    """Plot a histogram of the values in the lower triangular part of the given matrix.

    Args:
        matrix (Any): The matrix whose lower triangular values will be used for the histogram.
        title (str): The title of the histogram.

    Returns:
        None: Displays the histogram.
    """

    lower_triangular = np.tril(matrix)
    lower_triangular = lower_triangular.flatten()
    plt.hist(lower_triangular)
    plt.title(title)
    plt.show()


def plot_embedding(
    embedding_list: List[np.ndarray],
    dim: int,
    name_list: List[str],
    category_name_list: Optional[List[str]] = None,
    num_category_list: Optional[List[int]] = None,
    title: Optional[str] = None,
    has_legend: bool = True,
    fig_name: str = "Aligned_embedding",
    fig_dir: Optional[str] = None,
    **kwargs,
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
         **kwargs: 
            Other keyword arguments for customizing the plot.

    Raises:
        ValueError: `dim` must be 2 or 3.

    Returns:
        matplotlib.axes.Axes: embedding plot.
    """
    figsize = kwargs.get('figsize', (15, 15))
    xlabel = kwargs.get('xlabel', "PC1")
    xlabel_size = kwargs.get('xlabel_size', 25)
    ylabel = kwargs.get('ylabel', "PC2")
    ylabel_size = kwargs.get('ylabel_size', 25)
    zlabel = kwargs.get('zlabel', "PC3")
    zlabel_size = kwargs.get('zlabel_size', 25)
    title_size = kwargs.get('title_size', 60)
    legend_size = kwargs.get('legend_size')
    color_labels = kwargs.get('color_labels', None)
    color_hue = kwargs.get("color_hue", None)
    colorbar_label = kwargs.get("colorbar_label", None)
    colorbar_range = kwargs.get("colorbar_range", [0, 1])
    colorbar_shrink = kwargs.get("colorbar_shrink", 0.8)
    markers_list = kwargs.get('markers_list', None)
    marker_size = kwargs.get('marker_size', 30)
    cmap = kwargs.get('cmap', "viridis")
    show_figure = kwargs.get('show_figure', True)
    font = kwargs.get('font', 'Noto Sans CJK JP')
    elev = kwargs.get('elev', 30)
    azim = kwargs.get('azim' ,60)
    alpha = kwargs.get('alpha', 1)
    fig_ext = kwargs.get('fig_ext', "png")
    
    if color_labels is None:

        if num_category_list is None:
            color_labels = get_color_labels(
                embedding_list[0].shape[0],
                hue = color_hue, 
                show_labels = False,
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
    
    if dim == 3:
        # _, ax = plt.subplots(figsize = figsize, subplot_kw={'projection': '3d'})

        # # Adjust the scale of the axis.
        # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.9, 0.9, 0.9, 1]))
        
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111, projection='3d')
    
    elif dim == 2:
        _, ax = plt.subplots(figsize = figsize)
    
    else:
        raise ValueError("'dim' is either 2 or 3")

    # Set the axis
    if dim == 3:
        ax.set_xlabel(xlabel, fontsize = xlabel_size)
        ax.set_ylabel(ylabel, fontsize = ylabel_size)
        ax.set_zlabel(zlabel, fontsize = zlabel_size)
        ax.view_init(elev = elev, azim = azim)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # ax.w_xaxis.gridlines.set_color('black')
        # ax.w_yaxis.gridlines.set_color('black')
        # ax.w_zaxis.gridlines.set_color('black')
        
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.set_ticklabels([])
        ax.axes.get_zaxis().set_visible(True)
        ax.zaxis.pane.set_edgecolor('w')

    elif dim == 2:
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

    plt.tight_layout()
    if fig_dir is not None:
        fig_path = os.path.join(fig_dir, f"{fig_name}.{fig_ext}")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    if show_figure:
        plt.show()

    plt.clf()
    plt.close()
    return ax
