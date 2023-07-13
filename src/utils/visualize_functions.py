import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import colorsys
from sklearn.decomposition import PCA 
import seaborn as sns
from typing import List
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_color_labels(n, hue = None, show_labels = True):
    """Create color labels for n objects
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

def get_color_labels_for_category(n_category_list, min_saturation, show_labels = True):
    """Create color labels for n categoris

    Args:
        n_category_list (list): list of the numbers of n concepts
        min_saturation : minimum saturation

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
    matrix, 
    title, 
    save_file_name = None,
    ticks = None, 
    category_name_list = None, 
    num_category_list = None, 
    object_labels = None, 
    **kwargs
):
    figsize = kwargs.get('figsize', (8, 6))
    title_size = kwargs.get('title_size', 60)
    
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    
    xlabel_size = kwargs.get('xlabel_size', 40)
    ylabel_size = kwargs.get('ylabel_size', 40)
    
    xticks_rotation = kwargs.get('xticks_rotation', 90)
    yticks_rotation = kwargs.get('yticks_rotation', 0)
    
    cbar_ticks_size = kwargs.get("cbar_ticks_size", 20)
    ticks_size = kwargs.get('ticks_size', 20)
    
    cmap = kwargs.get('cmap', 'cividis')
    draw_category_line  = kwargs.get('draw_category_line', False) 
    category_line_alpha = kwargs.get('category_line_alpha', 0.2)
    category_line_style = kwargs.get('category_line_style', 'dashed')
    category_line_color = kwargs.get('category_line_color', 'C2')
    
    show_figure = kwargs.get('show_figure', True)

    fig, ax = plt.subplots(figsize = figsize)
    
    if title is not None:
        ax.set_title(title, size = title_size)
        
    aximg = ax.imshow(matrix, cmap=cmap, aspect='equal')
    
    if category_name_list is not None:
        assert num_category_list is not None
        if ticks == "objects":
            plt.xticks(np.arange(sum(num_category_list)) + 0.5, labels = object_labels, rotation = xticks_rotation, size = ticks_size)
            plt.yticks(np.arange(sum(num_category_list)) + 0.5, labels = object_labels, rotation = yticks_rotation, size = ticks_size)
        
        elif ticks == "category":
            label_pos = [sum(num_category_list[:i + 1]) for i in range(len(category_name_list))]
            plt.xticks(label_pos, labels = category_name_list, rotation = xticks_rotation, size = ticks_size, fontweight = "bold")
            plt.yticks(label_pos, labels = category_name_list, rotation = yticks_rotation, size = ticks_size, fontweight = "bold")
            
            if draw_category_line:
                for pos in label_pos:
                    plt.axhline(pos, alpha = category_line_alpha, linestyle = category_line_style, color = category_line_color)
                    plt.axvline(pos, alpha = category_line_alpha, linestyle = category_line_style, color = category_line_color)
        
        else:
            plt.xticks([])
            plt.yticks([])
    else:
        if ticks == "numbers":
            plt.xticks(ticks = np.arange(len(matrix)) + 0.5, labels = np.arange(len(matrix)) + 1, size = ticks_size, rotation = xticks_rotation)
            plt.yticks(ticks = np.arange(len(matrix)) + 0.5, labels = np.arange(len(matrix)) + 1, size = ticks_size, rotation = yticks_rotation)
        elif object_labels is not None:
            plt.xticks(ticks = np.arange(len(matrix)) + 0.5, labels = object_labels, size = ticks_size, rotation = xticks_rotation)
            plt.yticks(ticks = np.arange(len(matrix)) + 0.5, labels = object_labels, size = ticks_size, rotation = yticks_rotation)
        else:
            plt.xticks([])
            plt.yticks([])
    
    
    divider = make_axes_locatable(ax) 
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cbar = fig.colorbar(aximg, cax=cax, format = "%.2e")
    
    cbar.ax.tick_params(axis='y', labelsize = cbar_ticks_size)
  
    plt.xlabel(xlabel, size = xlabel_size)
    plt.ylabel(ylabel, size = ylabel_size)
    plt.tight_layout()
    
    if save_file_name is not None:
        plt.savefig(save_file_name)
    
    if show_figure:
        plt.show()
    
    plt.clf()
    plt.close()


def plot_lower_triangular_histogram(matrix, title):
    lower_triangular = np.tril(matrix)
    lower_triangular = lower_triangular.flatten()
    plt.hist(lower_triangular)
    plt.title(title)
    plt.show()


class VisualizeEmbedding():
    def __init__(self, embedding_list : List[np.ndarray], dim, category_name_list = None, num_category_list = None, category_idx_list = None) -> None:
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

    def apply_pca_to_embedding_list(self, n_dim_pca, show_result = True):
        """apply pca to the embedding list

        Args:
            embedding_list (list): list of embeddings
            n_dim_pca (int): dimmension after pca
            show_result (bool, optional): If true, show the cumulative contibution rate. Defaults to True.

        Returns:
            embedding_list_pca: list of embeddings after pca was applied
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
    
    
    def plot_embedding(self, name_list = None, legend = True, title = None, save_dir = None, **kwargs):
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
        markers_list = kwargs.get('markers_list', None)
        marker_size = kwargs.get('marker_size', 30)
        
        
        if color_labels is None:
            if self.num_category_list is None:
                color_labels = get_color_labels(self.embedding_list[0].shape[0], hue = color_hue, show_labels = False)
            else:
                color_labels, main_colors = get_color_labels_for_category(self.num_category_list, min_saturation = 1, show_labels = False)
        
        if markers_list is None:
            markers = ['o', 'x', '^', 's', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'D', 'd', '.', ',', '1', '2', '3', '4', '_', '|'][:len(self.embedding_list)]
        
        
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
            if self.dim == 3:
                ax.scatter(xs = coords_i[:, 0], ys = coords_i[:, 1], zs = coords_i[:, 2],
                           marker = markers[i], color = color_labels, s = marker_size, alpha = 1)
                ax.scatter([], [], [], marker = markers[i], color = "black", s = marker_size, alpha = 1, label = name_list[i])
            
            else:
                ax.scatter(x = coords_i[:, 0], y = coords_i[:, 1],
                           marker = markers[i], color = color_labels, s = marker_size, alpha = 1)
                ax.scatter(x = [], y = [], marker = markers[i], color = "black", s = marker_size, alpha = 1, label = name_list[i])

        if self.category_name_list is not None:
            for i, category in enumerate(self.category_name_list):
                if self.dim == 3:
                    ax.scatter([], [], [], marker = "o", color = main_colors[i], s = marker_size, alpha = 1, label = category)

                else:
                    ax.scatter(x = [], y = [], marker = "o", color = main_colors[i], s = marker_size, alpha = 1, label = category)
        if legend:
            ax.legend(fontsize = legend_size, loc = "best")

        if title is not None:
            plt.title(title, fontsize = title_size)

        if save_dir is not None:
            plt.savefig(save_dir)
        plt.show()
