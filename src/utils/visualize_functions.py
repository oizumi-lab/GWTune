import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import colorsys
from sklearn.decomposition import PCA 
import seaborn as sns
from typing import List

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

    
def plot_embedding_2d(embedding_list, markers_list, color_label = None, group_color = None, marker_size = 30, name_list = None, category_list = None, main_colors = None, title = None, title_fontsize = 20, legend = True, legend_fontsize = 12, save_dir = None):
    """plot embedding list

    Args:
        embedding_list (list): list of embeddings
        markers_list (list): list of markers for each groups
        color_label (list): color labels for each objects
        group_color (list): color labels for each groups. Defaults to None.
        name_list (list, optional): list of names for each groups. Defaults to None.
        category_list (list, optional): list of names for each categories. Defaults to None.
        main_colors (list, optional): list of representative color labels for each categories. Defaults to None.
        title (string, optional): title of the figure. Defaults to None.
        save_dir (string, optional): directory path for saving the figure. Defaults to None.
    """
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)

    if name_list == None:
        name_list = [f"Group {i+1}" for i in range(len(embedding_list))]
        
    for i in range(len(embedding_list)):
        if color_label is not None:
            color = color_label
        else: 
            color = group_color[i]
        coords_i = embedding_list[i]
        ax.scatter(x=coords_i[:, 0], y=coords_i[:, 1],
               marker=markers_list[i], color=color, s = marker_size, alpha = 1,label = name_list[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("PC1", fontsize=14)
        ax.set_ylabel("PC2", fontsize=14)
        if legend == True:
            ax.legend(fontsize=legend_fontsize, loc= "best")
            if category_list is not None:
                for i, category in enumerate(category_list):
                    ax.scatter([], [], marker = "o", color = main_colors[i], s = 30, alpha = 1, label = category)
                    ax.legend(fontsize=legend_fontsize, loc= "best")

    if title is not  None:
        plt.title(title, fontsize = title_fontsize)
    if save_dir is not None:
        plt.savefig(save_dir)
        
    plt.show()
        

def show_heatmap(matrix, title, category_name_list = None, num_category_list = None, object_labels = None, ticks = None, ticks_size = None, xlabel = None, ylabel = None, file_name = None):
    plt.figure(figsize = (20, 20))
    ax = sns.heatmap(matrix, square = True, cbar_kws = {"shrink": .80})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize = ticks_size)
    if category_name_list is not None:
        if ticks == "objects":
            plt.xticks(np.arange(sum(num_category_list)) + 0.5, labels = object_labels, rotation = 90, size = ticks_size)
            plt.yticks(np.arange(sum(num_category_list)) + 0.5, labels = object_labels, rotation = 0, size = ticks_size)
        elif ticks == "category":
            label_pos = [sum(num_category_list[: i + 1]) for i in range(len(category_name_list))]
            plt.xticks(label_pos, labels = category_name_list, rotation = 70, size = ticks_size, fontweight = "bold")
            plt.yticks(label_pos, labels = category_name_list, rotation = 0, size = ticks_size, fontweight = "bold")
        else:
            plt.xticks([])
            plt.yticks([])
    else:    
        if ticks == "numbers":
            plt.xticks(ticks = np.arange(len(matrix)) + 0.5, labels = np.arange(len(matrix)) + 1, size = ticks_size)
            plt.yticks(ticks = np.arange(len(matrix)) + 0.5, labels = np.arange(len(matrix)) + 1, size = ticks_size, rotation = 90)
        else:
            plt.xticks([])
            plt.yticks([])
    #plt.imshow(self.sim_mat)
    plt.xlabel(xlabel, size = 40)
    plt.ylabel(ylabel, size = 40)
    plt.title(title, size = 60)
    if file_name is not None:
        plt.savefig(file_name)
    plt.show() 
    
    
class Visualize_Embedding():
    def __init__(self, embedding_list : List[np.ndarray], name_list = None, color_labels = None, category_name_list = None, category_num_list = None, category_idx_list = None) -> None:
        self.embedding_list = embedding_list
        if category_idx_list is not None:
            self.embedding_list = [np.concatenate([embedding[category_idx_list[i]] for i in range(len(category_name_list))]) for embedding in self.embedding_list]
            
        if self.embedding_list[0].shape[1] > 3:
            self.embedding_list = self.apply_pca_to_embedding_list(n_dim_pca = 3, show_result = False)
        
        self.name_list = name_list
        self.category_name_list = category_name_list
        
        if color_labels is not None:
            self.color_labels = color_labels
        else:
            if category_num_list is None:
                self.color_labels = get_color_labels(self.embedding_list[0].shape[0], show_labels = False)
            else:
                self.color_labels, self.main_colors = get_color_labels_for_category(category_num_list, min_saturation = 1, show_labels = False)
        
        markers = ['o', 'x', '^', 's', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'D', 'd', '.', ',', '1', '2', '3', '4', '_', '|']
        self.markers = markers[:len(self.embedding_list)]

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
    
    def plot_embedding(self, dim = 3, marker_size = 30, legend = True, title = None, title_fontsize = 20, legend_fontsize = 12, save_dir = None):
        """plot embedding list

        Args:
            embedding_list (list): list of embeddings
            markers_list (list): list of markers for each groups
            color_label (list): color labels for each objects
            name_list (list, optional): list of names for each groups. Defaults to None.
            category_list (list, optional): list of names for each categories. Defaults to None.
            main_colors (list, optional): list of representative color labels for each categories. Defaults to None.
            title (string, optional): title of the figure. Defaults to None.
            save_dir (string, optional): directory path for saving the figure. Defaults to None.
        """
        fig = plt.figure(figsize=(15, 15))
        if dim == 3:
            ax = fig.add_subplot(1, 1, 1, projection='3d')
        else:
            ax = fig.add_subplot(1, 1, 1)
        plt.rcParams["grid.color"] = "black"
        ax.grid(True)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.axes.get_xaxis().set_visible(True)
        ax.axes.get_yaxis().set_visible(True)
        ax.w_xaxis.gridlines.set_color('black')
        ax.w_yaxis.gridlines.set_color('black')
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.set_xlabel("PC1", fontsize = 25)
        ax.set_ylabel("PC2", fontsize = 25)
        ax.view_init(elev = 30, azim = 60)
        
        if dim == 3:
            ax.zaxis.set_ticklabels([])
            ax.zaxis.pane.fill = False
            ax.axes.get_zaxis().set_visible(True)
            ax.w_zaxis.gridlines.set_color('black')
            ax.zaxis.pane.set_edgecolor('w')
            ax.set_zlabel("PC3", fontsize = 25)
            
        for i in range(len(self.embedding_list)):
            coords_i = self.embedding_list[i]
            if dim == 3:
                ax.scatter(xs = coords_i[:, 0], ys = coords_i[:, 1], zs = coords_i[:, 2],
                       marker = self.markers[i], color = self.color_labels, s = marker_size, alpha = 1, label = self.name_list[i])
            else:
                ax.scatter(xs = coords_i[:, 0], ys = coords_i[:, 1],
                       marker = self.markers[i], color = self.color_labels, s = marker_size, alpha = 1, label = self.name_list[i])
        if self.category_name_list is not None:
            for i, category in enumerate(self.category_name_list):
                ax.scatter([], [], [], marker = "o", color = self.main_colors[i], s = 30, alpha = 1, label = category)

        if legend:
            ax.legend(fontsize=legend_fontsize, loc= "best")

        if title is not None:   
            plt.title(title, fontsize = title_fontsize)
        if save_dir is not None:
            plt.savefig(save_dir)
        plt.show()
    