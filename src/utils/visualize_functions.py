import colorsys
from typing import Any, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA


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
            plt.xticks(ticks = np.arange(len(matrix)) + 0.5, labels = np.arange(len(matrix)) + 1, size = xticks_size, rotation = xticks_rotation)
            plt.yticks(ticks = np.arange(len(matrix)) + 0.5, labels = np.arange(len(matrix)) + 1, size = yticks_size, rotation = yticks_rotation)
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


class VisualizeEmbedding():
    """A class to visualize embeddings in either 2D or 3D using PCA.

    This class provides functions to visualize embeddings in a 2D or 3D space. Through the use of PCA, this class allows for
    the reduction of high-dimensional embeddings down to 2 or 3 dimensions for visualization purposes. The class offers various
    customization options, including the ability to color-code and differentiate multiple embeddings based on categories using
    distinct markers and colors.

    Attributes:
        embedding_list (List[np.ndarray]): A list of embeddings to be visualized.
        dim (int): Dimension (either 2 or 3) for the visualization after applying PCA.
        category_name_list (Optional[List[str]]): List of category names.
        num_category_list (Optional[List[int]]): List of the number of items in each category.
        category_idx_list (Optional[List[int]]): Index list for categories.
    """

    def __init__(
        self,
        embedding_list : List[np.ndarray],
        dim: int,
        category_name_list: Optional[List[str]] = None,
        num_category_list: Optional[List[int]] = None,
        category_idx_list: Optional[List[int]] = None
    ) -> None:
        """Initialize the VisualizeEmbedding class.

        Args:
            embedding_list (List[np.ndarray]): A list of embeddings.
            dim (int): Dimension (either 2 or 3) for the visualization after applying PCA.
            category_name_list (Optional[List[str]]): List of category names. Defaults to None.
            num_category_list (Optional[List[int]]): List of the number of items in each category. Defaults to None.
            category_idx_list (Optional[List[int]]): Index list for categories. Defaults to None.
        """

        self.embedding_list = embedding_list
        if category_idx_list is not None:
            category_concat_embedding_list = []
            for embedding in self.embedding_list:
                concatenated_embedding = np.concatenate([embedding[category_idx_list[i]] for i in range(len(category_name_list))])
                category_concat_embedding_list.append(concatenated_embedding)
            self.embedding_list = category_concat_embedding_list

        if self.embedding_list[0].shape[1] > 3:
            self.embedding_list = self.apply_pca_to_embedding_list(n_dim_pca = dim, show_result = False)

        self.dim = dim
        self.category_name_list = category_name_list
        self.num_category_list = num_category_list
        self.category_idx_list = category_idx_list

    def apply_pca_to_embedding_list(self, n_dim_pca: int, show_result: bool = True) -> List[np.ndarray]:
        """Apply pca to the embedding list.

        Args:
            embedding_list (list): A list of embeddings.
            n_dim_pca (int): Dimmension after PCA.
            show_result (bool, optional): If true, show the cumulative contibution rate. Defaults to True.

        Returns:
            embedding_list_pca (list): A list of embeddings after PCA.
        """

        pca = PCA(n_components = n_dim_pca)
        n_object = self.embedding_list[0].shape[0]
        embedding_list_cat = np.concatenate([self.embedding_list[i] for i in range(len(self.embedding_list))], axis = 0)
        embedding_list_pca = pca.fit_transform(embedding_list_cat)
        embedding_list_pca = [embedding_list_pca[i*n_object:(i+1)*n_object] for i in range(len(self.embedding_list))]

        if show_result:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
            ax.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
            plt.xlabel("Number of principal components")
            plt.ylabel("Cumulative contribution rate")
            plt.grid()
            plt.show()

        return embedding_list_pca


    def plot_embedding(
        self,
        name_list: Optional[List[str]] = None,
        legend: bool = True,
        title: Optional[str] = None,
        save_dir: Optional[str] = None,
        **kwargs
    ) -> None:
        """Plot the embeddings in 2D or 3D space.

        Args:
            name_list (Optional[List[str]]): Names for each embedding. Defaults to None.
            legend (bool): Whether or not to show the legend. Defaults to True.
            title (Optional[str]): Title for the plot. Defaults to None.
            save_dir (Optional[str]): Directory to save the plot. If None, the plot won't be saved. Defaults to None.
            **kwargs: Other keyword arguments for customizing the plot.

        Returns:
            None: Displays or saves the plot.
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
        alpha = kwargs.get('alpha', 1)
        show_figure = kwargs.get('show_figure', True)

        if color_labels is None:
            if self.num_category_list is None:
                color_labels = get_color_labels(plot_idx.shape[0], hue = color_hue, show_labels = False)
            else:
                color_labels, main_colors = get_color_labels_for_category(self.num_category_list, min_saturation = 1, show_labels = False)

        if markers_list is None:
            markers_list = ['o', 'x', '^', 's', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'D', 'd', '.', ',', '1', '2', '3', '4', '_', '|'][:len(self.embedding_list)]

        plt.style.use("default")
        plt.rcParams["grid.color"] = "black"
        fig = plt.figure(figsize = figsize)

        if self.dim == 3:
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

        elif self.dim == 2:
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

        for i in range(len(self.embedding_list)):
            coords_i = self.embedding_list[i]
            coords_i = coords_i[plot_idx]
            if self.dim == 3:
                im = ax.scatter(
                    xs = coords_i[:, 0],
                    ys = coords_i[:, 1],
                    zs = coords_i[:, 2],
                    marker = markers_list[i],
                    color = color_labels,
                    s = marker_size,
                    alpha = 1,
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
                    alpha = 1,
                    cmap=cmap,
                )

                ax.scatter(x = [], y = [], marker = markers_list[i], color = "black", s = marker_size, alpha = 1, label = name_list[i].replace("_", " "))

        if self.category_name_list is not None:
            for i, category in enumerate(self.category_name_list):
                if self.dim == 3:
                    ax.scatter([], [], [], marker = "o", color = main_colors[i], s = marker_size, alpha = 1, label = category)

                else:
                    ax.scatter(x = [], y = [], marker = "o", color = main_colors[i], s = marker_size, alpha = 1, label = category)
        ax.set_axisbelow(True)

        if legend:
            ax.legend(fontsize = legend_size, loc = "best")
            #ax.legend(fontsize = legend_size, loc='upper left', bbox_to_anchor=(1, 1))

        if title is not None:
            plt.title(title, fontsize = title_size)

        if colorbar_label is not None:
            im.set_cmap(cmap)
            cbar = plt.colorbar(im, shrink=colorbar_shrink, ax=ax)
            cbar.set_label(colorbar_label, size=xlabel_size)
            cbar.ax.tick_params(labelsize=xlabel_size)
            cbar.mappable.set_clim(colorbar_range[0], colorbar_range[1])

        if save_dir is not None:
            plt.savefig(save_dir)

        if show_figure:
            plt.show()

        plt.clf()
        plt.close()
