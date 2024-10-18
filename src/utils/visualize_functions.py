import colorsys
from typing import Any, List, Tuple, Optional
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib.colors import LogNorm

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

    cm = plt.get_cmap(hue)
    color_labels = [cm(v) for v in np.linspace(0, 1, n)]
    
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

def add_colored_label(ax, x, y, bgcolor, width=1, height=1):
    rect = Rectangle((x, y), width, height, facecolor=bgcolor)
    ax.add_patch(rect)

def show_heatmap(
    matrix: Any,
    title: Optional[str],
    save_file_name: Optional[str] = None,
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
    cbar_format = kwargs.get('cbar_format', None)
    cbar_label = kwargs.get('cbar_label', None)
    cbar_label_size = kwargs.get('cbar_label_size', 20)
    cbar_range = kwargs.get('cbar_range', None)
    cmap = kwargs.get('cmap', 'cividis')

    ticks = kwargs.get('ticks', None)
    ot_object_tick = kwargs.get("ot_object_tick", False)
    ot_category_tick = kwargs.get("ot_category_tick", False)

    draw_category_line  = kwargs.get('draw_category_line', False)
    category_line_alpha = kwargs.get('category_line_alpha', 0.2)
    category_line_style = kwargs.get('category_line_style', 'dashed')
    category_line_color = kwargs.get('category_line_color', 'C2')

    font = kwargs.get('font', 'Arial')
    show_figure = kwargs.get('show_figure', True)
    
    color_labels = kwargs.get('color_labels', None)
    color_label_width = kwargs.get('color_label_width', None)
    
    dpi = kwargs.get('dpi', 300)

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
        assert category_name_list is not None, "please provide 'category_name_list' or set `OT_format` or `sim_mat_format` to `sorted`."
        assert num_category_list is not None

        if ticks == "objects":
            plt.xticks(np.arange(sum(num_category_list)) + 0.5, labels = x_object_labels, rotation = xticks_rotation, size = xticks_size)
            plt.yticks(np.arange(sum(num_category_list)) + 0.5, labels = y_object_labels, rotation = yticks_rotation, size = yticks_size)

        elif ticks == "category":
            label_pos = [sum(num_category_list[:i + 1]) for i in range(len(category_name_list))]
            plt.xticks(label_pos, labels = category_name_list, rotation = xticks_rotation, size = xticks_size)
            plt.yticks(label_pos, labels = category_name_list, rotation = yticks_rotation, size = yticks_size)

            if draw_category_line:
                for pos in label_pos:
                    plt.axhline(pos, alpha = category_line_alpha, linestyle = category_line_style, color = category_line_color)
                    plt.axvline(pos, alpha = category_line_alpha, linestyle = category_line_style, color = category_line_color)


    if ot_object_tick and not ot_category_tick:
        if ticks is None:
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
    
    if color_labels is not None:
        for idx, color in enumerate(color_labels):
            add_colored_label(ax, -color_label_width, idx, color, width=color_label_width)
            add_colored_label(ax, idx, matrix.shape[0], color, height=color_label_width)

        ax.set_aspect('equal')
        ax.set_xlim(-color_label_width, matrix.shape[1])
        ax.set_ylim(matrix.shape[0] + color_label_width, 0)
        
        for spine in ax.spines.values():
            spine.set_visible(False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cbar = fig.colorbar(aximg, cax=cax, format = cbar_format)
    cbar.set_label(cbar_label, size = cbar_label_size)
    
    if cbar_range is not None:
        cmin, cmax = cbar_range
        cbar.mappable.set_clim(cmin, cmax)
    
    
    cbar.ax.tick_params(axis='y', labelsize = cbar_ticks_size)

    plt.tight_layout()

    if save_file_name is not None:
        plt.savefig(save_file_name, bbox_inches='tight', dpi=dpi)

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
    font = kwargs.get('font', 'Arial')
    elev = kwargs.get('elev', 30)
    azim = kwargs.get('azim' ,60)
    alpha = kwargs.get('alpha', 1)
    fig_ext = kwargs.get('fig_ext', "png")
    dpi = kwargs.get('dpi', 300)
    
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

    fig.tight_layout()
    
    if fig_dir is not None:
        fig_path = os.path.join(fig_dir, f"{fig_name}.{fig_ext}")
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')

    if show_figure:
        plt.show()

    plt.clf()
    plt.close()
    return ax


def plot_optimization_log(
    df_trial: pd.DataFrame,
    pair_name: str,
    eps_list : List[float],
    fig_dir: Optional[str] = None,
    *,
    figsize: Tuple[int, int] = (8, 6),
    title_size: int = 20,
    xlabel_size: int = 20,
    ylabel_size: int = 20,
    xticks_rotation: int = 0,
    xticks_size: int = 10,
    yticks_size: int = 10,
    cbar_format: Optional[str] = None,
    cbar_label_size: int = 20,
    cbar_ticks_size: int = 20,
    cmap: str = "viridis",
    grid_alpha : float = 1.0,
    marker_size: int = 20,
    plot_eps_log: bool = False,
    lim_eps: Optional[Tuple[float, float]] = None,
    lim_gwd: Optional[Tuple[float, float]] = None,
    lim_acc: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    font: str = "Arial",
    fig_ext: str = "png",
    show_figure: bool = False,
    edgecolor:Optional[int] = None,
    linewidth:Optional[int] = None,
    **kwargs,
) -> Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]:
    """Display a heatmap of the given matrix with various customization options.

    Args:
        df_trial (pd.DataFrame): 
            The dataframe of the optimization log.
        pair_name (str): 
            The name of the pair.
        fig_dir (Optional[str], optional):
            Directory to save the heatmap. If None, the heatmap won't be saved.

    Keyword Args:
        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (8, 6).
        title (Optional[str], optional): The title of the figure. Defaults to None.
        title_size (int, optional): The size of the title. Defaults to 20.
        xlabel_size (int, optional): The size of the x-axis label. Defaults to 20.
        ylabel_size (int, optional): The size of the y-axis label. Defaults to 20.
        xticks_rotation (int, optional): The rotation of the x-axis ticks. Defaults to 0.
        cbar_ticks_size (int, optional): The size of the colorbar ticks. Defaults to 20.
        xticks_size (int, optional): The size of the x-axis ticks. Defaults to 10.
        yticks_size (int, optional): The size of the y-axis ticks. Defaults to 10.
        cbar_format (Optional[str], optional): The format of the colorbar ticks. Defaults to None.
        cbar_label_size (int, optional): The size of the colorbar label. Defaults to 20.
        cbar_range (Optional[List[float]], optional): The range of the colorbar. Defaults to None.
        cmap (str, optional): The colormap to use. Defaults to "viridis".
        marker_size (int, optional): The size of the markers. Defaults to 20.
        plot_eps_log (bool, optional): Whether to plot epsilon in log scale. Defaults to False.
        lim_eps (Optional[Tuple[float, float]], optional): The limits of the range of epsilon. Defaults to None.
        lim_gwd (Optional[Tuple[float, float]], optional): The limits of the range of GWD. Defaults to None.
        lim_acc (Optional[Tuple[float, float]], optional): The limits of the range of accuracy. Defaults to None.
        fig_ext (str, optional): The extension of the saved figure. Defaults to "png".
        show_figure (bool, optional): Whether to show the figure. Defaults to True.
    """
    plt.style.use("default")
    plt.rcParams["grid.color"] = "black"
    plt.rcParams["grid.alpha"] = str(grid_alpha)
    plt.rcParams['font.family'] = font
    
    # figure plotting epsilon as x-axis and GWD as y-axis
    _, ax1 = plt.subplots(figsize=figsize)
    sc1 = ax1.scatter(
        df_trial["params_eps"],
        df_trial["value"],
        c = 100 * df_trial["user_attrs_best_acc"],
        s=marker_size,
        cmap=cmap,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    
    ax1.set_title(f"epsilon - GWD ({pair_name})", fontsize=title_size)
    ax1.set_xlabel("epsilon", fontsize=xlabel_size)
    ax1.set_ylabel("GWD", fontsize=ylabel_size)

    if lim_eps is not None:
        ax1.set_xlim(lim_eps)

    if lim_gwd is not None:
        ax1.set_ylim(lim_gwd)

    if plot_eps_log:
        ax1.set_xscale("log")

    ax1.tick_params(
        axis="x", 
        which="both", 
        labelsize=xticks_size, 
        rotation=xticks_rotation,
    )
    ax1.tick_params(axis="y", which="major", labelsize=yticks_size)

    ax1.grid(True, which="both")
    cbar = plt.colorbar(sc1, ax=ax1)
    cbar.set_label(label="Matching Rate (%)", size=cbar_label_size)
    
    if lim_acc is not None:
        cmin, cmax = lim_acc
        cbar.mappable.set_clim(cmin, cmax)
    
    cbar.ax.tick_params(labelsize=cbar_ticks_size)

    plt.tight_layout()

    if fig_dir is not None:
        # save figure
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(
            os.path.join(fig_dir, f"Optim_log_eps_GWD_{pair_name.replace(' ', '_')}.{fig_ext}"),
            bbox_inches='tight',
            dpi=dpi,
        )
    
    # show figure
    if show_figure:
        plt.show()
    
    plt.clf()
    plt.close()
    
    plt.style.use("default")
    plt.rcParams["grid.color"] = "black"
    plt.rcParams["grid.alpha"] = str(grid_alpha)
    plt.rcParams['font.family'] = font

    # figure plotting accuracy as x-axis and GWD as y-axis
    if plot_eps_log:
        if lim_eps is not None:
            norm = LogNorm(vmin=lim_eps[0], vmax=lim_eps[1])
        else:
            norm = LogNorm(vmin=eps_list[0], vmax=eps_list[1])
    else:
        norm = None

    _, ax2 = plt.subplots(figsize=figsize)
    sc2 = ax2.scatter(
        100 * df_trial["user_attrs_best_acc"],
        df_trial["value"].values,
        c=df_trial["params_eps"],
        cmap=cmap,
        s=marker_size,
        norm=norm,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    
   
    ax2.set_title(f"Matching Rate - GWD ({pair_name})", fontsize=title_size)

    if lim_acc is not None:
        ax2.set_xlim(lim_acc)

    if lim_gwd is not None:
        ax2.set_ylim(lim_gwd)
        
    ax2.set_xlabel("Matching Rate (%)", fontsize=xlabel_size)
    ax2.tick_params(axis="x", labelsize=xticks_size)
    ax2.set_ylabel("GWD", fontsize=ylabel_size)
    ax2.tick_params(axis="y", labelsize=yticks_size)
    
    cbar2 = plt.colorbar(mappable=sc2, ax=ax2, format=cbar_format)
    cbar2.set_label(label="epsilon", size=cbar_label_size)
    # cbar2.ax.set_yscale('log')
    # cbar2.set_ticks(eps_list, minor=False)
    cbar2.ax.tick_params(labelsize=cbar_ticks_size)
    
    if lim_eps is not None:
        cmin, cmax = lim_eps
        cbar2.mappable.set_clim(cmin, cmax)
    
   
    
    ax2.grid(True)
    plt.tight_layout()

    if fig_dir is not None:
        # save figure
        plt.savefig(
            os.path.join(fig_dir, f"acc_gwd_eps({pair_name.replace(' ', '_')}).{fig_ext}"),
            bbox_inches='tight',
            dpi=dpi,
        )
    
    if show_figure:
        plt.show()
        
    plt.clf()
    plt.close()

    
    