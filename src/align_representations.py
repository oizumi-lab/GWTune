# %%
import copy
import glob
import itertools
import os
import re
import shutil
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import optuna
import ot
import pandas as pd
import seaborn as sns
import torch
import multiprocessing as mp
import scipy as sp

import sqlalchemy_utils
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr
from sklearn import manifold
from sqlalchemy import URL, create_engine
from sqlalchemy_utils import create_database, database_exists, drop_database
from tqdm.auto import tqdm
from sklearn.base import TransformerMixin
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from .gw_alignment import GW_Alignment
from .histogram_matching import SimpleHistogramMatching
from .utils import backend, gw_optimizer, visualize_functions, utils_functions, init_matrix


# %%
class OptimizationConfig:
    """
    This is an instance for sharing the parameters to compute GWOT with the instance PairwiseAnalysis.

    Please check the tutoial.ipynb for detailed info.
    """
    def __init__(
        self,
        gw_type: str = "entropic_gromov_wasserstein",
        eps_list: List[float] = [1.0, 10.0],
        eps_log: bool = True,
        num_trial: int = 4,
        sinkhorn_method: str = "sinkhorn",
        device: str = "cpu",
        to_types: str = "numpy",
        data_type: str = "double",
        n_jobs: int = 1,
        multi_gpu: Union[bool, List[int]] = False,
        storage: Optional[str] = None,
        db_params: Optional[Dict[str, Union[str, int]]] = {
            "drivername": "mysql",
            "username": "root",
            "password": "",
            "host": "localhost",
            "port": 3306,
        },
        init_mat_plan: str = "random",
        user_define_init_mat_list: Union[List, None] = None,
        n_iter: int = 1,
        max_iter: int = 200,
        tol: float = 1e-9,
        numItermax: int = 1000,
        sampler_name: str = "tpe",
        pruner_name: str = "hyperband",
        pruner_params: Dict[str, Union[int, float]] = {
            "n_startup_trials": 1,
            "n_warmup_steps": 2,
            "min_resource": 2,
            "reduction_factor": 3,
        },
        show_progress_bar: bool = True,
    ) -> None:
        """Initialization of the instance.

        Args:
            gw_type (str, optional):
                Type of Gromov-Wasserstein alignment. Options are "entropic_gromov_wasserstein",
                "entropic_semirelaxed_gromov_wasserstein" and "entropic_partial_gromov_wasserstein".
                Defaults to "entropic_gromov_wasserstein".
            eps_list (List[float], optional):
                List of epsilon values (regularization term). Defaults to [1., 10.].
            eps_log (bool, optional):
                Indicates if the epsilon values are sampled from log space. Defaults to True.
            num_trial (int, optional):
                Number of trials for the optimization. Defaults to 4.
            sinkhorn_method (str, optional):
                Method used for Sinkhorn algorithm. Options are "sinkhorn", "sinkhorn_log", "greenkhorn",
                "sinkhorn_stabilized", or "sinkhorn_epsilon_scaling". Defaults to 'sinkhorn'.
            device (str, optional):
                The device to be used for computation, either "cpu" or "cuda". Defaults to "cpu".
            to_types (str, optional):
                Specifies the type of data structure to be used, either "torch" or "numpy". Defaults to "numpy".
            data_type (str, optional):
                Specifies the type of data to be used in computation. Defaults to "double".
            n_jobs (int, optional):
                Number of jobs to run in parallel. Defaults to 1.
            multi_gpu (Union[bool, List[int]], optional):
                Indicates if multiple GPUs are used for computation. Defaults to False.
            storage (str, optional):
                URL for the database storage. Defaults to None.
            db_params (dict, optional):
                Parameters for creating the database URL.
                Defaults to {"drivername": "mysql", "username": "root", "password": "", "host": "localhost", "port": 3306}.
            init_mat_plan (str, optional):
                The method to initialize transportation plan. Defaults to "random".
            n_iter (int, optional):
                Number of initial plans evaluated in single optimization. Defaults to 1.
            max_iter (int, optional):
                Maximum number of iterations for entropic Gromov-Wasserstein alignment by POT. Defaults to 200.
            numItermax (int):
                Maximum number of iterations for sinkhorn algorithm by POT. Defaults to 1000.
            tol (float, optional):
                Tolerance for convergence while computing OT matrix in sinkhorn algorithm by POT. Defaults to 1e-9.
            sampler_name (str, optional):
                Name of the sampler used in optimization. Options are "random", "grid", and "tpe". Defaults to "tpe".
            pruner_name (str, optional):
                Name of the pruner used in optimization. Options are "hyperband", "median", and "nop". Defaults to "hyperband".
            pruner_params (dict, optional):
                Additional parameters for the pruner. See Optuna's pruner page for more details.
                Defaults to {"n_startup_trials": 1, "n_warmup_steps": 2, "min_resource": 2, "reduction_factor": 3}.
            show_progress_bar (bool, optional):
                Indicates if the progress bar is shown during optimization. Defaults to True.
        """

        self.gw_type = gw_type
        self.eps_list = eps_list
        self.eps_log = eps_log
        self.num_trial = num_trial
        self.sinkhorn_method = sinkhorn_method

        self.to_types = to_types
        self.data_type = data_type
        self.device = device

        self.n_jobs = n_jobs
        self.multi_gpu = multi_gpu

        self.storage = storage
        self.db_params = db_params
        assert (
            storage is not None or db_params is not None
        ), "storage or db_params must be provided."

        self.init_mat_plan = init_mat_plan
        self.n_iter = n_iter
        self.max_iter = max_iter
        self.numItermax = numItermax
        
        self.tol = tol

        self.sampler_name = sampler_name
        self.user_define_init_mat_list = user_define_init_mat_list

        self.pruner_name = pruner_name
        self.pruner_params = pruner_params
        self.show_progress = show_progress_bar

class VisualizationConfig:
    """This is an instance for sharing the parameters to make the figures of GWOT with the instance PairwiseAnalysis.

    Please check the tutoial.ipynb for detailed info.
    """

    def __init__(
        self,
        show_figure: bool = True,
        fig_ext:str='png',
        bins: int = 100,
        dpi:int = 300,
        font:str='DejaVu Sans',#'Noto Sans CJK JP'
        figsize: Tuple[int, int] = (8, 6),
        cbar_label_size: int = 15,
        cbar_ticks_size: int = 10,
        cbar_format: Optional[str]=None,
        cbar_label: Optional[str]=None,
        cbar_range: Optional[List[float]] = None,
        xticks_size: int = 10,
        yticks_size: int = 10,
        xticks_rotation: int = 0,
        yticks_rotation: int = 0,
        elev:int = 30,
        azim:int = 60,
        tick_format: str = '%.2f',
        title_size: int = 20,
        legend_size: int = 5,
        xlabel: Optional[str] = None,
        xlabel_size: int = 15,
        ylabel: Optional[str] = None,
        ylabel_size: int = 15,
        zlabel: Optional[str] = None,
        zlabel_size: int = 15,
        color_labels: Optional[List[str]] = None,
        color_label_width = None,
        color_hue: Optional[str] = None,
        colorbar_label: Optional[str] = None,
        colorbar_range: List[float] = [0., 1.],
        colorbar_shrink: float = 1.,
        markers_list: Optional[List[str]] = None,
        marker_size: int = 30,
        alpha: int = 1,
        grid_alpha :float = 1.0,
        color: str = 'C0',
        cmap: str = 'cividis',
        ticks: Optional[str] = None,
        ot_object_tick: bool = False,
        ot_category_tick: bool = False,
        draw_category_line: bool = False,
        category_line_color: str = 'C2',
        category_line_alpha: float = 0.2,
        category_line_style: str = 'dashed',
        plot_eps_log: bool = False,
        lim_eps: Optional[float] = None,
        lim_gwd: Optional[float] = None,
        lim_acc: Optional[float] = None,
        edgecolor:Optional[str] = None,
        linewidth:Optional[int] = None,
    ) -> None:
        """Initializes the VisualizationConfig class with specified visualization parameters.

        Args:
            show_figure (bool, optional):
                Whether to display the figure. Defaults to True.
            fig_ext (str, optional):
                The extension of the figure. Defaults to 'png'.
            fonmt (str, optional):
                The font of the figure. Defaults to 'DejaVu Sans'.
            figsize (Tuple[int, int], optional):
                Size of the figure. Defaults to (8, 6).
            cbar_label_size (int, optional):
                Size of the colorbar label. Defaults to 15.
            cbar_ticks_size (int, optional):
                Size of the colorbar ticks. Defaults to 10.
            cbar_format (Optional[str]):
                Format of the colorbar. Defaults to None.
            cbar_label (Optional[str]):
                Title of the colorbar. Defaults to None.
            xticks_size (int, optional):
                Size of the xticks. Defaults to 10.
            yticks_size (int, optional):
                Size of the yticks. Defaults to 10.
            xticks_rotation (int, optional):
                Rotation angle of the xticks. Defaults to 0.
            yticks_rotation (int, optional):
                Rotation angle of the yticks. Defaults to 0.
            elev (int, optional):
                Elevation angle of the 3D plot. Defaults to 30.
            azim (int, optional):
                Azimuthal angle of the 3D plot. Defaults to 60.
            tick_format (Optional[str]):
                Format of the ticks. Defaults to '%.2f'.
            title_size (int, optional):
                Size of the title. Defaults to 20.
            legend_size (int, optional):
                Size of the legend. Defaults to 5.
            xlabel (str, optional):
                Label of the x-axis. Defaults to None.
            xlabel_size (int, optional):
                Size of the x-axis label. Defaults to 15.
            ylabel (str, optional):
                Label of the y-axis. Defaults to None.
            ylabel_size (int, optional):
                Size of the y-axis label. Defaults to 15.
            zlabel (str, optional):
                Label of the z-axis. Defaults to None.
            zlabel_size (int, optional):
                Size of the z-axis label. Defaults to 15.
            color_labels (List[str], optional):
                Labels of the color. Defaults to None.
            color_label_width (int, optional):
                Width of the color label. Defaults to None.
            color_hue (str, optional):
                Hue of the color. Defaults to None.
            colorbar_label (str, optional):
                Label of the colorbar. Defaults to None.
            colorbar_range (list, optional):
                Range of the colorbar. Defaults to [0, 1].
            colorbar_shrink (float, optional):
                Shrink of the colorbar. Defaults to 1.
            markers_list (List[str], optional):
                List of markers. Defaults to None.
            marker_size (int, optional):
                Size of the marker. Defaults to 30.
            alpha (int, optional):
                Alpha of the marker. Defaults to 1.
            grid_alpha (float, optional):
                Alpha of the grid. Defaults to 1.0.
            color (str, optional):
                Color for plots. Defaults to 'C0'.
            cmap (str, optional):
                Colormap of the figure. Defaults to 'cividis'.
            ot_object_tick (bool, optional):
                Whether to tick object. Defaults to False.
            ot_category_tick (bool, optional):
                Whether to tick category. Defaults to False.
            draw_category_line (bool, optional):
                Whether to draw category lines. Defaults to False.
            category_line_color (str, optional):
                Color for category lines. Defaults to 'C2'.
            category_line_alpha (float, optional):
                Alpha for category lines. Defaults to 0.2.
            category_line_style (str, optional):
                Style for category lines. Defaults to 'dashed'.
            plot_eps_log (bool, optional):
                Whether to plot in logarithmic scale for epsilon. Defaults to False.
            lim_eps (float, optional):
                Limits for epsilon. Defaults to None.
            lim_gwd (float, optional):
                Limits for GWD. Defaults to None.
            lim_acc (float, optional):
                Limits for accuracy. Defaults to None.
            edgecolor (Optional[str], optional):
                Color of the edge. Defaults to None.
            linewidth (Optional[int], optional):
                Width of the line for the edge of plot. Defaults to None.
        """

        self.visualization_params = {
            'show_figure':show_figure,
            'fig_ext':fig_ext,
            'bins':bins,
            'dpi':dpi,
            'font':font,
            'figsize': figsize,
            'cbar_label_size': cbar_label_size,
            'cbar_ticks_size': cbar_ticks_size,
            'cbar_format':cbar_format,
            'cbar_label':cbar_label,
            'cbar_range': cbar_range,
            'xticks_size': xticks_size,
            'yticks_size': yticks_size,
            'xticks_rotation': xticks_rotation,
            'yticks_rotation': yticks_rotation,
            'elev': elev,
            'azim': azim,
            'tick_format': tick_format,
            'title_size': title_size,
            'legend_size': legend_size,
            'xlabel': xlabel,
            'xlabel_size': xlabel_size,
            'ylabel': ylabel,
            'ylabel_size': ylabel_size,
            'zlabel': zlabel,
            'zlabel_size': zlabel_size,
            'color_labels': color_labels,
            'color_label_width': color_label_width,
            'color_hue': color_hue,
            'colorbar_label': colorbar_label,
            'colorbar_range': colorbar_range,
            'colorbar_shrink': colorbar_shrink,
            'alpha': alpha,
            'grid_alpha': grid_alpha,
            'markers_list': markers_list,
            'marker_size': marker_size,
            'color':color,
            'cmap':cmap,
            'ticks':ticks,
            'ot_object_tick': ot_object_tick,
            'ot_category_tick': ot_category_tick,
            'draw_category_line': draw_category_line,
            'category_line_color': category_line_color,
            'category_line_alpha': category_line_alpha,
            'category_line_style': category_line_style,
            'plot_eps_log':plot_eps_log,
            'lim_eps':lim_eps,
            'lim_gwd':lim_gwd,
            'lim_acc':lim_acc,
            'edgecolor':edgecolor,
            'linewidth':linewidth,
        }

    def __call__(self) -> Dict[str, Any]:
        """Returns the visualization parameters.

        Returns:
            Dict[str, Any]: Dictionary containing the visualization parameters.
        """
        return self.visualization_params

    def set_params(self, **kwargs) -> None:
        """Allows updating the visualization parameters.

        Args:
            **kwargs: keyword arguments representing the parameters to be updated.
        """

        for key, item in kwargs.items():
            self.visualization_params[key] = item


class Representation:
    """A class that has information of a representation, such as embeddings and similarity matrices.

    The class provides methods and attributes to manage and visualize different data representations,
    such as embeddings and similarity matrices. It supports operations like computing similarity matrices from embeddings,
    estimating embeddings from similarity matrices using Multi-Dimensional Scaling (MDS), and various visualization techniques
    for the embeddings and similarity matrices. Additionally, it incorporates functionality to handle and visualize category-wise
    information if provided.

    Attributes:
        name (str):
            The name of the representation.
        metric (str, optional):
            The metric for computing distances between embeddings. Defaults to "cosine".
        sim_mat (np.ndarray, optional):
            Representational dissimilarity matrix. Defaults to None.
        embedding (np.ndarray, optional):
            The array of N-dimensional embeddings of all stimuli. Defaults to None.
        get_embedding (bool, optional):
            If True, the embeddings are automatically computed from the sim_mat using MDS method. Defaults to True.
        MDS_dim (int, optional):
            The dimension of the embeddings computed automatically from the sim_mat using MDS. Defaults to 3.
        object_labels (List[str], optional):
            The labels for each stimulus or points. Defaults to None.
        category_name_list (List[str], optional):
            If there is coarse category labels, you can select categories to be used in the unsupervised alignment. Defaults to None.
        num_category_list (List[int], optional):
            The list of numbers of stimuli each coarse category contains. Defaults to None.
        category_idx_list (List[int], optional):
            The list of indices that represents which coarse category each stimulus belongs to. Defaults to None.
        func_for_sort_sim_mat (Callable, optional):
            A function to rearrange the matrix so that stimuli belonging to the same coarse category are arranged adjacent to each other.
            Defaults to None.
        save_conditional_rdm_path (str, optional):
            The path to save the conditional similarity matrix. Defaults to None.
            If None, the conditional similarity matrix is not created. 
            The conditional similarity matrix is saved in the "save_conditional_rdm_path/metric" for each metric.
    """

    def __init__(
        self,
        name: str,
        metric: str = "cosine",
        sim_mat: Optional[np.ndarray] = None,
        embedding: Optional[np.ndarray] = None,
        get_embedding: bool = False,
        MDS_dim: int = 3,
        object_labels: Optional[List[str]] = None,
        category_name_list: Optional[List[str]] = None,
        num_category_list: Optional[List[int]] = None,
        category_idx_list: Optional[List[int]] = None,
        func_for_sort_sim_mat: Optional[Callable] = None,
        save_conditional_rdm_path: Optional[str] = None,
    ) -> None:
        
        self.name = name
        self.metric = metric

        # parameters for label information (e.g. pictures of dog, cat,...) in the dataset for the representation matrix.
        self.object_labels = copy.deepcopy(object_labels)
        self.category_name_list = copy.deepcopy(category_name_list)
        self.category_idx_list = copy.deepcopy(category_idx_list)
        self.num_category_list = copy.deepcopy(num_category_list)

        # define the function to sort the representation matrix by the label parameters above (Default is None).]
        # Users can define it by themselves.
        self.func_for_sort_sim_mat = func_for_sort_sim_mat
        
        # save the conditional similarity matrix. If None, the conditional similarity matrix is not created. 
        # The conditional similarity matrix is saved in the "save_conditional_rdm_path/metric" for each metric.
        self.save_conditional_rdm_path = save_conditional_rdm_path
        
        # compute the dissimilarity matrix from embedding if sim_mat is None,
        # or estimate embedding from the dissimilarity matrix using MDS if embedding is None.
        assert (sim_mat is not None) or (embedding is not None), "sim_mat and embedding are None."
        
        self.back_end = backend.Backend(device='cpu', to_types='numpy', data_type='double')        

        if sim_mat is None:
            self.embedding = self.back_end(embedding)
            self.sim_mat = self._get_sim_mat()
        else:
            self.sim_mat = sim_mat

        if embedding is None:
            self.sim_mat = self.back_end(sim_mat)
            
            if get_embedding:
                self.embedding = self._get_embedding(dim=MDS_dim)
            
        else:
            self.embedding = embedding

        if self.category_idx_list is not None:
            self.sorted_sim_mat = self.func_for_sort_sim_mat(
                self.sim_mat, 
                category_idx_list=self.category_idx_list
            )
        else:
            self.sorted_sim_mat = None

    def _get_embedding(self, dim: int) -> np.ndarray:
        """Estimate embeddings from the dissimilarity matrix using the MDS method.

        Args:
            dim (int): The dimension of the embeddings to be computed.

        Returns:
            np.ndarray: The computed embeddings.
        """
        MDS_embedding = manifold.MDS(
            n_components=dim, 
            dissimilarity="precomputed", 
            random_state=0, 
            normalized_stress="auto",
        )
        embedding = MDS_embedding.fit_transform(self.sim_mat)
        return embedding

    def _conditional_sim_mat(self, sim_matrix: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            sim_mat (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        save_path = os.path.join(self.save_conditional_rdm_path, self.metric)
        os.makedirs(save_path, exist_ok=True)
        
        tar_path = os.path.join(save_path, f"{self.name}_{self.metric}.npy")
        
        if not os.path.exists(tar_path):
            print(f"Conditional similarity matrix is computed for {self.name} with {self.metric}.")
            n_objects = sim_matrix.shape[0]
            indices = np.array(np.triu_indices(n_objects, k=1))

            n_indices = indices.shape[1]
            batch_size = min(n_indices, 10_000)
            n_batches = (n_indices + batch_size - 1) // batch_size

            rsm = np.zeros_like(sim_matrix)
            ooo_accuracy = 0.0

            pbar = tqdm(total=n_batches)

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_indices)
                batch_indices = indices[:, start_idx:end_idx]
                i, j = batch_indices
                s_ij = sim_matrix[i, j]
                s_ik = sim_matrix[i, :]
                s_jk = sim_matrix[j, :]

                n = end_idx - start_idx
                n_range = np.arange(n)

                s_ik[n_range, i] = 0
                s_ik[n_range, j] = 0
                s_jk[n_range, j] = 0
                s_jk[n_range, i] = 0

                s_ij = np.expand_dims(s_ij, 1)
                s_ij = np.repeat(s_ij, s_jk.shape[1], axis=1)

                rr = sp.special.softmax([s_ij, s_jk, s_ik], axis=0)
                softmax_ij = rr[0]
                
                proba_sum = softmax_ij.sum(1) - 2
                mean_proba = proba_sum / (n_objects - 2)
                rsm[i, j] = mean_proba
                ooo_accuracy += mean_proba.mean()

                pbar.set_description(f"Batch {batch_idx+1}/{n_batches}")
                pbar.update(1)
                
            rsm = rsm + rsm.T 
            rsm[range(len(sim_matrix)), range(len(sim_matrix))] = 1
            ooo_accuracy = ooo_accuracy.item() / n_batches
            
            sim_mat = 1 - rsm
            
            np.save(tar_path, sim_mat)
            
        else:
            sim_mat = np.load(tar_path)
            
        return sim_mat
    
    def _get_sim_mat(self) -> np.ndarray:
        """Compute the dissimilarity matrix based on the given metric.

        Returns:
            np.ndarray: The computed dissimilarity matrix.
        """
        
        if self.metric == "dot":
            sim_matrix = np.dot(self.embedding, self.embedding.T)
        
        elif self.metric == "L2_normalized_euclidean":
            self.embedding = self.embedding / np.linalg.norm(self.embedding, axis=1, keepdims=True)
            sim_matrix = distance.cdist(self.embedding, self.embedding, metric="euclidean")
        
        elif self.metric == "mahalanobis":
            cov_matrix = np.cov(self.embedding.T)
        
            inv_cov_matrix = np.linalg.pinv(cov_matrix)
            
            diff_mat = self.embedding[:, np.newaxis, :] - self.embedding[np.newaxis, :, :]
             
            mid_values = np.dot(diff_mat, inv_cov_matrix) * diff_mat

            sim_matrix = np.sqrt(np.sum(mid_values, axis = 2))
            
        else:
            sim_matrix = distance.cdist(self.embedding, self.embedding, metric=self.metric)
        
        if self.save_conditional_rdm_path is not None:
            assert isinstance(self.save_conditional_rdm_path, str) == True, "The save_conditional_rdm_path must be a string for the save path." 
            sim_mat = self._conditional_sim_mat(sim_matrix)
        
        else:
            sim_mat = sim_matrix 
        
        return sim_mat

    def plot_sim_mat(
        self,
        fig_dir:Optional[str]=None,
        return_sorted:bool=False,
        visualization_config: VisualizationConfig = VisualizationConfig(),
    ):
        if fig_dir is not None:
            os.makedirs(fig_dir, exist_ok=True)

        if not return_sorted:
            visualize_functions.show_heatmap(
                self.sim_mat,
                title=self.name,
                category_name_list=None,
                num_category_list=None,
                x_object_labels=self.object_labels,
                y_object_labels=self.object_labels,
                fig_name=f"RDM {self.name}",
                fig_dir=fig_dir,
                **visualization_config(),
            )

        else:
            assert self.category_idx_list is not None, "No label info to sort the 'sim_mat'."
            visualize_functions.show_heatmap(
                self.sorted_sim_mat,
                title=self.name,
                category_name_list=self.category_name_list,
                num_category_list=self.num_category_list,
                x_object_labels=self.object_labels,
                y_object_labels=self.object_labels,
                fig_name=f"RDM {self.name} (sorted)",
                fig_dir=fig_dir,
                **visualization_config(),
            )

    def show_sim_mat_distribution(
        self,
        visualization_config: VisualizationConfig = VisualizationConfig(),
        **kwargs,
    ) -> None:
        """
        Show the distribution of the values of elements of the dissimilarity matrix.
        """
        params = visualization_config()
        figsize = params.get('figsize', (4, 3))
        title_size = params.get("title_size", 20)
        color = params.get("color", "C0")
        bins = params.get("bins", 100)
        show_figure = params.get("show_figure", True)
        
        lower_triangular = np.tril(self.sim_mat)
        lower_triangular = lower_triangular.flatten()

        plt.figure(figsize=figsize)
        plt.hist(lower_triangular, bins=bins, color=color)
        plt.title(f"Distribution of RDM ({self.name})", fontsize=title_size)
        plt.xlabel("RDM value")
        plt.ylabel("Count")
        plt.grid(True)
        
        if show_figure:
            plt.show()
        
        plt.clf()
        plt.close()

    def show_embedding(
        self,
        dim: int = 3,
        visualization_config: VisualizationConfig = VisualizationConfig(),
        category_name_list: Optional[List[str]] = None,
        num_category_list: Optional[List[int]] = None,
        category_idx_list: Optional[List[int]] = None,
        title: Optional[str] = None,
        legend: bool = True,
        fig_dir: Optional[str] = None,
        fig_name: str ="Aligned_embedding.png"
    ) -> None:
        """Show the embeddings.

        Depending on the provided "dim", this function visualizes the embeddings in 2D or 3D space.
        If the original dimensionality of the embeddings exceeds the specified "dim", a dimensionality
        reduction method (e.g., PCA) is applied.

        Args:
            dim (int, optional):
                The dimension of the embedding space for visualization. If the original dimensions of the embeddings are
                higher than "dim", the dimension reduction method (PCA) is applied. Defaults to 3.
            visualization_config (VisualizationConfig, optional):
                Configuration for visualization details. Defaults to VisualizationConfig().
            category_name_list (Optional[List[str]:, optional):
                Select the coarse category labels to be visualized in the embedding space. Defaults to None.
            num_category_list (Optional[List[int]], optional):
                List of numbers of stimuli each coarse category contains. Defaults to None.
            category_idx_list (Optional[List[int]], optional):
                List of category indices. Defaults to None.
            title (Optional[str], optional):
                The title of the figure. Defaults to None.
            legend (bool, optional):
                If True, the legend is shown. Defaults to True.
            fig_dir (Optional[str], optional):
                The directory for saving the figure. Defaults to None.
            fig_name (str, optional):
                The name of the figure. Defaults to "Aligned_embedding.png".
        """

        if category_idx_list is None:
            if self.category_idx_list is not None:
                category_name_list = self.category_name_list
                num_category_list = self.num_category_list
                category_idx_list = self.category_idx_list
            
        if self.embedding.shape[1] > dim:
            embedding_list, _ = utils_functions.obtain_embedding(
                embedding_list=[self.embedding],
                dim=dim,
                emb_name="PCA",
            )
        else:
            embedding_list = [self.embedding]
        
        if fig_dir is not None:
            os.makedirs(fig_dir, exist_ok=True)

        visualize_functions.plot_embedding(
            embedding_list=embedding_list,
            dim=dim,
            name_list=[self.name], 
            category_name_list=category_name_list,
            num_category_list=num_category_list,
            category_idx_list=category_idx_list,
            title=title, 
            has_legend=legend,
            fig_name=fig_name, 
            fig_dir=fig_dir, 
            **visualization_config()
        )


class PairwiseAnalysis:
    """A class object that has methods conducting gw-alignment and corresponding results.

    This class provides functionalities to handle the alignment and comparison between a pair of Representations,
    typically referred to as source and target.

    Attributes:
        config (OptimizationConfig):
            Parameters to compute the GWOT.
        source (Representation):
            Instance of the source representation.
        target (Representation):
            Instance of the target representation.
        data_name (str):
            Name of the data to be analyzed. For example, 'color'.
        pair_name (str):
            Name of this instance. Derived from source and target if not provided.
        instance_name (str):
            Name of this instance. Derived from data_name and pair_name if not provided.
        study_name (str):
            Name of the optimization study.
        results_dir (str):
            Path to save the result data
        save_path (str):
            Complete path to save the result data, generated for each init_mat_plan.
        figure_path (str):
            Path to save figures.
        data_path (str):
            Path to save optimization results.
        storage (str):
            URL for the database storage.
        OT (np.ndarray):
            GWOT.
    """
    def __init__(
        self,
        data_name: str,
        results_dir: str,
        config: OptimizationConfig,
        source: Representation,
        target: Representation,
        pair_name: Optional[str] = None,
        instance_name: Optional[str] = None,
    ) -> None:

        # information of the representations and optimization
        self.config = config
        self.source = source
        self.target = target

        # information of the data
        self.data_name = data_name  # name of align representations
        
        # name of pair
        if pair_name is None:
            self.pair_name = f"{source.name}_vs_{target.name}"
        else:
            self.pair_name = pair_name
        
        # name of align representations, the same as the name of the folder for computation result data.
        if instance_name is None:
            self.instance_name = self.data_name + "_" + self.pair_name
        else: 
            self.instance_name = instance_name
        
        # name of study name, the same as the name of database for computation result data.
        self.study_name = self.instance_name + "_" + self.config.init_mat_plan

        # path setting
        self.results_dir = results_dir
        self.save_path = os.path.join(results_dir, self.config.init_mat_plan)
        self.data_path = os.path.join(self.save_path, "data")
        
        # Generate the URL for the database. Syntax differs for SQLite and others.
        self.storage = self.config.storage
        if self.storage is None:
            if self.config.db_params["drivername"] == "sqlite":
                self.storage = f"sqlite:///{self.save_path}/{self.study_name}.db"

            else:
                self.storage = URL.create(
                    database=self.study_name,
                    **self.config.db_params,
                ).render_as_string(hide_password=False)
        
        self.__OT = None
    
    @property
    def OT(self):
        try:
            if self.__OT is None:
                self.__OT = self._run_entropic_gwot(compute_OT=False)
        except KeyError as e:
            raise ValueError(f'OT for {self.pair_name} has not been computed yet.')
        return self.__OT
    
    @OT.setter
    def OT(self, value):
        self.__OT = value

    @property
    def sorted_OT(self):
        assert (self.source.sorted_sim_mat is not None), "No label info to sort the 'sim_mat'."
        sorted_OT = self.source.func_for_sort_sim_mat(
            self.OT, 
            category_idx_list=self.source.category_idx_list
        )
        return sorted_OT

    @property
    def study(self):
        try:
            study = self._run_optimization(compute_OT=False)
        except KeyError as e:
            raise ValueError(f'study (optuna result) for {self.pair_name} has not been computed yet.')
        return study
    
    # RSA methods
    def rsa(self, metric: str = "pearson") -> float:
        """Conventional representation similarity analysis (RSA).

        Args:
            metric (str, optional):
                spearman or pearson. Defaults to "pearson".

        Returns:
            corr :
                RSA value
        """
        a = self.source.sim_mat
        b = self.target.sim_mat

        upper_tri_a = a[np.triu_indices(a.shape[0], k=1)]
        upper_tri_b = b[np.triu_indices(b.shape[0], k=1)]

        if metric == "spearman":
            corr, _ = spearmanr(upper_tri_a, upper_tri_b)
        elif metric == "pearson":
            corr, _ = pearsonr(upper_tri_a, upper_tri_b)

        return corr

    def rsa_for_each_category(self, metric: str = "pearson") -> List[float]:
        """_summary_

        Args:
            metric (str, optional): _description_. Defaults to "pearson".

        Returns:
            List[float]: _description_
        """
        assert self.source.category_idx_list is not None, "No label info to sort the 'category_idx_list'."
        
        category_match = []
        cat_boundary = np.cumsum(self.source.num_category_list)
        
        for idx, block in enumerate(cat_boundary):
            if idx == 0:
                prev_boundary = 0
            else:
                prev_boundary = cat_boundary[idx-1]
                
            category_data_a = self.source.sim_mat[prev_boundary:block, prev_boundary:block]
            category_data_b = self.target.sim_mat[prev_boundary:block, prev_boundary:block]
            
            upper_tri_a = category_data_a[np.triu_indices(category_data_a.shape[0], k=1)]
            upper_tri_b = category_data_b[np.triu_indices(category_data_b.shape[0], k=1)]
            
            if metric == "spearman":
                corr, _ = spearmanr(upper_tri_a, upper_tri_b)
            elif metric == "pearson":
                corr, _ = pearsonr(upper_tri_a, upper_tri_b)
            
            category_match.append(corr)
       
        index_list = [f"{name} ({self.source.num_category_list[i]})" for i, name in enumerate(self.source.category_name_list)]
        
        df_cat_match = pd.DataFrame({"category_rsa": category_match}, index=index_list)

        return df_cat_match
    
    def show_both_sim_mats(self, src = None, tar = None):
        """
        visualize the two sim_mat.
        """
        if src is None:
            a = self.source.sim_mat
        else:
            a = src
        
        if tar is None:
            b = self.target.sim_mat
        else:
            b = tar

        plt.figure()
        plt.subplot(121)

        plt.title("source : " + self.source.name)
        plt.imshow(a, cmap=plt.cm.jet)
        plt.colorbar(orientation="horizontal")

        plt.subplot(122)
        plt.title("target : " + self.target.name)
        plt.imshow(b, cmap=plt.cm.jet)
        plt.colorbar(orientation="horizontal")

        plt.tight_layout()
        plt.show()

        a_hist, a_bin = np.histogram(a, bins=100)
        b_hist, b_bin = np.histogram(b, bins=100)

        plt.figure()
        plt.title("histogram, source : " + self.source.name + ", target : " + self.target.name)
        plt.hist(a_bin[:-1], a_bin, weights=a_hist, label=self.source.name, alpha=0.5)
        plt.hist(b_bin[:-1], b_bin, weights=b_hist, label=self.target.name, alpha=0.5)
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()

    def match_sim_mat_distribution(
        self,
        return_data: bool = False,
        method: str = "target",
        plot: bool = False,
    ) -> np.ndarray:
        """Performs simple histogram matching between two matrices.

        Args:
            return_data (bool, optional): 
                If True, this method will return the matched result.
                If False, self.target.sim_mat will be re-written. Defaults to False.

            method (str) : 
                change the sim_mat of which representations, source or target. Defaults to "target".
            
            plot (bool):
                If True, the RDM and histograms of the source and target will be plotted. Defaults to False.
            
            bins (int):
                The number of bins for the histogram. Defaults to 100.

        Returns:
            new target: matched results of sim_mat.
        """
        a = self.source.sim_mat
        b = self.target.sim_mat
        
        if plot:
            print("Before matching")
            self.show_both_sim_mats(a, b)

        matching = SimpleHistogramMatching(a, b)

        if method == "source":
            unchange = b
            new_data = matching.simple_histogram_matching(method=method)
        elif method == "target":
            unchange = a
            new_data = matching.simple_histogram_matching(method=method)
        else:
            raise ValueError("method must be either 'source' or 'target'.")
        
        if plot:
            if method == "source":
                print("source is changed after matching")
                self.show_both_sim_mats(new_data, unchange)
                
            else:
                print("target is changed after matching")
                self.show_both_sim_mats(unchange, new_data)

        if return_data:
            return new_data
        else:
            if method == "source":
                self.source.sim_mat = new_data
            else:
                self.target.sim_mat = new_data

    def delete_previous_results(
        self,
        delete_database: bool = True,
        delete_directory: bool = True,
    ) -> None:
        """Delete the previous computed results of GWOT.

        Args:
            delete_database (bool, optional):
                If True, the database will be deleted.
                If False, only the study will be deleted. Defaults to True.
            delete_directory (bool, optional):
                If True, the directory `self.save_path` will be deleted.
                If False, only the files in the directory will be deleted. Defaults to True.
        """

        # delete study or storage
        if sqlalchemy_utils.database_exists(self.storage):
            if delete_database:
                sqlalchemy_utils.drop_database(self.storage)
            else:
                try:
                    optuna.delete_study(study_name=self.study_name, storage=self.storage)
                except KeyError:
                    print(f"study {self.study_name} doesn't exist in {self.storage}")

        # delete directosry
        if delete_directory:
            shutil.rmtree(self.results_dir)
        else:
            if os.path.exists(self.data_path):
                for root, _, files in os.walk(self.data_path, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))

    #GWOT methods
    def run_entropic_gwot(
        self,
        compute_OT: bool = False,
        save_dataframe: bool = False,
        target_device: Optional[str] = None,
        sampler_seed: int = 42,
        fix_random_init_seed: bool = False,
        *,
        first_random_init_seed: Optional[int] = None,
        iter_from_ComputeGWOT: Optional[int] = None,
        queue: Optional[mp.Queue] = None,
    ):
        """
        Run the entropic GWOT.
        
        Args:
            compute_OT (bool, optional):
                If True, GWOT will be computed. Defaults to False.
            save_dataframe (bool, optional):
                If True, the dataframe will be saved. Defaults to False.
            target_device (Optional[str], optional):
                The device to be used for computation. Defaults to None.
            sampler_seed (int, optional):
                Seed for the sampler. Defaults to 42.
            fix_random_init_seed (bool, optional):
                If True, the random seed will be fixed. Defaults to False.
            first_random_init_seed (Optional[int], optional):
                The first random seed for the initialization. Defaults to None.
            iter_from_ComputeGWOT (Optional[int], optional):
                The iteration number of the PairwiseAnalysis instance. Defaults to None.
            queue (Optional[mp.Queue], optional):
                The queue for multiprocessing with multi GPU. Defaults to None.

        Returns:
            ot (Union[np.ndarray, torch.Tensor]): GWOT.
        """
        
        for p in [self.results_dir, self.save_path, self.data_path]:
            os.makedirs(p, exist_ok=True)

        # device setting
        if self.config.to_types == "torch" and self.config.device == "cuda" and self.config.multi_gpu != False:
            if queue is None:
                if target_device is None:
                    new_device = self.config.device
                else:
                    new_device = target_device
            
            elif not queue.empty():
                device_idx = queue.get()
                new_device = "cuda:" + str(device_idx)
                print(f"cuda:{device_idx} is computing another pair, {self.instance_name}...")
            else:
                if self.config.multi_gpu == True:
                    device_idx = iter_from_ComputeGWOT % torch.cuda.device_count()
                    new_device = "cuda:" + str(device_idx)
                    print(f"cuda:{device_idx} is computing {self.instance_name}...")
                
                elif isinstance(self.config.multi_gpu, list):
                    _idx = iter_from_ComputeGWOT % len(self.config.multi_gpu)
                    device_idx = self.config.multi_gpu[_idx]
                    new_device = "cuda:" + str(device_idx)
                    print(f"multi_gpu is list, cuda:{device_idx} is computing {self.instance_name}...")
        
        else:
            new_device = target_device

        # run gwot
        ot = self._run_entropic_gwot(
            compute_OT,
            target_device=new_device,
            sampler_seed=sampler_seed,
            save_dataframe=save_dataframe,
            fix_random_init_seed=fix_random_init_seed,
            first_random_init_seed=first_random_init_seed,
        )
        
        if self.config.multi_gpu != False and self.config.to_types == "torch":
            queue.put(device_idx)

        return ot

    def _run_entropic_gwot(
        self,
        compute_OT: bool,
        target_device: Optional[str] = None,
        sampler_seed: int = 42,
        save_dataframe: bool = False,
        fix_random_init_seed: bool = False,
        first_random_init_seed: Optional[int] = None,
    ) -> Tuple[Union[np.ndarray, torch.Tensor]]:
        """Computes or loads the entropic Gromov-Wasserstein Optimal Transport (GWOT).

        This method either computes the GW alignment or loads it from a saved path,
        depending on the provided arguments.

        Args:
            compute_OT (bool):
                If True, GWOT will be computed.
            target_device (Optional[str], optional):
                The device to be used for computation. Defaults to None.
            sampler_seed (int, optional):
                Seed for the sampler. Defaults to 42.
            save_dataframe (bool, optional):
                If True, the dataframe will be saved. Defaults to False.
            fix_random_init_seed (bool, optional):
                If True, the random seed will be fixed. Defaults to False.
            first_random_init_seed (Optional[int], optional):
                The first random seed for the initialization. Defaults to None.

        Returns:
            OT (Union[np.ndarray, torch.Tensor]): GWOT.
        """

        study = self._run_optimization(
            compute_OT=compute_OT,
            target_device=target_device,
            sampler_seed=sampler_seed,
            fix_random_init_seed=fix_random_init_seed,
            first_random_init_seed=first_random_init_seed,
        )

        if save_dataframe:
            df_trial = study.trials_dataframe()
            df_trial.to_csv(self.save_path + "/" + self.instance_name + ".csv")

        best_trial = study.best_trial
        ot_path = glob.glob(self.data_path + f"/gw_{best_trial.number}.*")[0]

        if ".npy" in ot_path:
            OT = np.load(ot_path)

        elif ".pt" in ot_path:
            OT = torch.load(ot_path, weights_only=False).to("cpu").numpy()

        return OT

    def _run_optimization(
        self,
        compute_OT: bool = False,
        target_device: Optional[str] = None,
        sampler_seed: int = 42,
        fix_random_init_seed: bool = False,
        first_random_init_seed: Optional[int] = None,
        *,
        n_jobs_for_pairwise_analysis: int = 1,
    ) -> optuna.study.Study:
        """Run or load an optimization study for Gromov-Wasserstein Optimal Transport (GWOT).

        Args:
            compute_OT (bool, optional):
                If True, runs the optimization study to compute GWOT. If False, loads an existing study.
                Defaults to False.
            target_device (Optional[str], optional):
                The device to compute GWOT. If not provided, defaults to the device specified in the configuration.
            sampler_seed (int, optional):
                Seed for the sampler. Defaults to 42.
            n_jobs_for_pairwise_analysis (int, optional):
                Number of jobs to run originally implemented by Optuna. Defaults to 1.
                Changing this value may cause unexpected behavior with TPE sampler.

        Returns:
            study (optuna.study.Study):
                The result of the optimization study, typically an instance of a study object.
        """

        # generate instance optimize gw_alignment
        opt = gw_optimizer.load_optimizer(
            save_path=self.save_path,
            filename=self.instance_name,
            storage=self.storage,
            init_mat_plan=self.config.init_mat_plan,
            n_iter=self.config.n_iter,
            num_trial=self.config.num_trial,
            n_jobs=n_jobs_for_pairwise_analysis,
            method="optuna",
            sampler_name=self.config.sampler_name,
            pruner_name=self.config.pruner_name,
            pruner_params=self.config.pruner_params,
        )

        if compute_OT:
            
            if target_device == None:
                target_device = self.config.device
            
            # generate instance solves gw_alignment
            gw = GW_Alignment(
                self.source.sim_mat,
                self.target.sim_mat,
                self.data_path,
                storage=self.storage,
                study_name=self.study_name,
                max_iter=self.config.max_iter,
                numItermax=self.config.numItermax,
                n_iter=self.config.n_iter,
                fix_random_init_seed=None if fix_random_init_seed is False else int(self.config.num_trial * self.config.n_iter),
                device=target_device,
                gw_type=self.config.gw_type,
                to_types=self.config.to_types,
                data_type=self.config.data_type,
                sinkhorn_method=self.config.sinkhorn_method,
                instance_name=self.instance_name,
                first_random_init_seed=first_random_init_seed,
                tol=self.config.tol,
                show_progress=self.config.show_progress,
            )

            # setting for optimization
            if self.config.init_mat_plan == "user_define":
                gw.main_compute.init_mat_builder.set_user_define_init_mat_list(
                    self.config.user_define_init_mat_list
                )

            if self.config.sampler_name == "grid":
                # used only in grid search sampler below the two lines
                if self.config.eps_log:
                    eps_space = np.logspace(
                        np.log10(self.config.eps_list[0]), 
                        np.log10(self.config.eps_list[1]), 
                        self.config.num_trial,
                    )
                else:
                    eps_space = np.linspace(
                        self.config.eps_list[0],
                        self.config.eps_list[1], 
                        self.config.num_trial,
                    )
                
                search_space = {"eps": eps_space}
            else:
                search_space = None

            # 2. run optimzation
            if self.config.init_mat_plan == "random":
                print(f"fix init mat random seed: {fix_random_init_seed}")
                
            study = opt.run_study(
                gw,
                seed=sampler_seed,
                init_mat_plan=self.config.init_mat_plan,
                eps_list=self.config.eps_list,
                eps_log=self.config.eps_log,
                search_space=search_space,
            )

        else:
            study = opt.load_study(compute_OT=compute_OT)

        return study

    # evaluation methods
    def calc_matching_rate(
        self,
        top_k_list: List[int],
        eval_type: str = "ot_plan",
        category_mat: Optional[np.ndarray] = None,
        metric: str = "cosine",
        barycenter: bool = False,
        supervised: bool = False,
    ) -> pd.DataFrame:
        """Evaluation of the accuracy of the unsupervised alignment

        There are two ways to evaluate the accuracy.
        1. Calculate the accuracy based on the OT plan.
        For using this method, please set the parameter `eval_type = "ot_plan"` in "calc_accuracy()".

        2. Calculate the matching rate based on the k-nearest neighbors of the embeddings.
        For using this method, please set the parameter `eval_type = "k_nearest"` in "calc_accuracy()".

        For both cases, the accuracy evaluation criterion can be adjusted by setting `top_k_list`.

        Args:
            top_k_list (List[int]):
                the list of k for the accuracy evaluation.

            eval_type (str, optional):
                two ways to evaluate the accuracy as above. Defaults to "ot_plan".

            category_mat (Optional[np.ndarray], optional):
                This will be used for the category info. Defaults to None.
                
            metric (str, optional):
                Please set the metric that can be used in "scipy.spatical.distance.cdist()". Defaults to "cosine".

            barycenter (bool, optional):
                Indicates if the accuracy should be evaluated with respect to a barycenter representation.
                Defaults to False.

            supervised (bool, optional):
                define the accuracy based on a diagnoal matrix. Defaults to False.

        Returns:
            df : dataframe which has accuracies for top_k.
        """

        df = pd.DataFrame()
        df["top_n"] = top_k_list

        if supervised:
            OT = np.diag([1 / len(self.target.sim_mat)] * len(self.target.sim_mat))
        else:
            OT = self.OT

        acc_list = list()
        
        for k in top_k_list:
            if eval_type == "k_nearest":
                if not barycenter:
                    new_embedding_source = self.procrustes(self.target.embedding, self.source.embedding, OT)
                else:
                    new_embedding_source = self.source.embedding

                # Compute distances between each points
                dist_mat = distance.cdist(self.target.embedding, new_embedding_source, metric)

                acc = self._calc_matching_rate_with_eval_mat(dist_mat, k=k, eval_mat=None, order="minimum")

            elif eval_type == "ot_plan":
                acc = self._calc_matching_rate_with_eval_mat(OT, k=k, eval_mat=None, order="maximum")
            
            elif eval_type == "category":
                eval_mat = np.dot(category_mat.values, category_mat.values.T)
                acc = self._calc_matching_rate_with_eval_mat(OT, k=k, eval_mat=eval_mat, order="maximum")

            acc_list.append(acc)

        df[self.pair_name] = acc_list

        return df

    def _calc_matching_rate_with_eval_mat(self, matrix, k, eval_mat=None, order="maximum"):
        """
        Calculate the matching rate based on the k-nearest neighbors of the embeddings.
        
        Args:
            matrix (np.ndarray):
                The matrix to be evaluated.
            k (int):
                The number of nearest neighbors to consider.
            eval_mat (Optional[np.ndarray], optional):
                The evaluation matrix. Defaults to None.
            order (str, optional):
                The order to consider. Defaults to "maximum".
        
        return:
            matching_rate (float): The matching rate.
        
        """
        if eval_mat is None:
            eval_mat = np.eye(matrix.shape[0])
        else:
            assert matrix.shape == eval_mat.shape
        
        # Get the top k values for each row
        if order == "maximum":
            topk_values = np.argpartition(matrix, -k)[:, -k:]
        elif order == "minimum":
            topk_values = np.argpartition(matrix, k - 1)[:, :k]
        else:
            raise ValueError("Invalid order parameter. Must be 'maximum' or 'minimum'.")
        
        count = 0
        for i, row in enumerate(topk_values):
            indices = np.argwhere(eval_mat[i, :] == 1) # indices of grand truth
            
            # Count the number of indices for each row where the indices of matching values are included in the top k values
            count += np.isin(indices, row).any()

        # Calculate the accuracy as the proportion of counts to the total number of rows
        matching_rate = count / matrix.shape[0]
        matching_rate *= 100
        
        return matching_rate 

    def compare_for_each_category(
        self,
        category_mat: Optional[pd.DataFrame] = None,
        eval_type: str = "category",
    ):    
        """
        Compare the accuracy for each category.

        Args:
            category_mat (Optional[pd.DataFrame], optional):
                The category matrix. Defaults to None.
            eval_type (str, optional):
                The type of evaluation. Defaults to "category

        Returns:
            count_df (pd.DataFrame):
                The dataframe containing the accuracy for each category.
        """
        
        assert self.source.category_idx_list is not None, "No label info to sort the 'category_idx_list'."
        assert self.source.category_name_list is not None, "No label info to sort the 'category_name_list'."
        assert self.source.num_category_list is not None, "No label info to sort the 'num_category_list'."
        
        k = 1
        matrix = self.OT
        
        topk_values = np.partition(matrix, -k)[:, -k:]
        
        if eval_type == "category":
            eval_mat = np.dot(category_mat.values, category_mat.values.T)
        
        else:
            eval_mat = np.eye(matrix.shape[0])
        
        assert matrix.shape == eval_mat.shape
        
        df = pd.DataFrame()
        for i, row in enumerate(matrix):
            indices = np.argwhere(eval_mat[i, :] == 1) # indices of grand truth
            matching_values = row[indices] # and there values on a row
            
            count = np.isin(matching_values, topk_values[i]).any()
        
            class_info = category_mat.iloc[i][category_mat.iloc[i] != 0].index
            assert len(class_info) <= 1, "There are multiple classes in the category matrix."
    
            df_min = pd.DataFrame({'category':str(class_info[0]), 'count':count}, index=[i]) 
            df = pd.concat([df, df_min])
            

        count_df = df.groupby('category').sum()    
        count_df = count_df.reindex(index=self.source.category_name_list)
        
        index_list = [f"{name} ({self.source.num_category_list[i]})" for i, name in enumerate(self.source.category_name_list)]
        
        count_df["count"] = count_df["count"] / np.array(self.source.num_category_list) * 100
        
        count_df.index = index_list
        
        return count_df
       
    def procrustes(
        self,
        embedding_target: np.ndarray,
        embedding_source: np.ndarray,
        OT: np.ndarray,
    ) -> np.ndarray:
        """Function that brings embedding_source closest to embedding_target by orthogonal matrix

        Args:
            embedding_target (np.ndarray):
                Target embeddings with shape (n_target, m).

            embedding_source (np.ndarray):
                Source embeddings with shape (n_source, m).

            OT (np.ndarray):
                Transportation matrix from source to target with shape (n_source, n_target).


        Returns:
            new_embedding_source (np.ndarray):
                Transformed source embeddings with shape (n_source, m).
        """
        
        if embedding_target.shape[1] == embedding_source.shape[1]:
            U, S, Vt = np.linalg.svd(
                np.matmul(embedding_source.T, np.matmul(OT, embedding_target))
            )

            Q = np.matmul(U, Vt)
            new_embedding_source = np.matmul(embedding_source, Q)
        
        else:
            print("embedding_target and embedding_source have different dimensions.")
            new_embedding_source = None

        return new_embedding_source

    # barycenter methods
    def wasserstein_alignment(self, metric):
        a = ot.unif(len(self.source.embedding))
        b = ot.unif(len(self.target.embedding))

        M = distance.cdist(self.source.embedding, self.target.embedding, metric=metric)

        self.OT, log = ot.emd(a, b, M, log=True)

        return log["cost"]

    def get_new_source_embedding(self):
        return self.procrustes(self.target.embedding, self.source.embedding, self.OT)

    # plot methods
    def plot_OT(
        self,
        return_sorted:bool = False,
        fig_dir: Optional[str] = None,
        visualization_config: VisualizationConfig = VisualizationConfig(),
    ) -> matplotlib.axes.Axes:
        """
        Visualize the OT for a single PairwiseAnalysis object.

        Args:
            return_sorted (bool, optional):
                format of sim_mat to visualize. Defaults to False.
            fig_dir (Optional[str], optional):
                Directory to save the heatmap. If None, the heatmap won't be saved.
            visualization_config (VisualizationConfig, optional):
                Configuration for visualization details. Defaults to VisualizationConfig().
        """

        if fig_dir is not None:
            os.makedirs(fig_dir, exist_ok=True)
        
        title = f"$\Gamma$ ({self.pair_name.replace('_', ' ')})"

        if not return_sorted:
            visualize_functions.show_heatmap(
                self.OT,
                title=title,
                category_name_list=None,
                num_category_list=None,
                x_object_labels=self.source.object_labels,
                y_object_labels=self.target.object_labels,
                fig_name=self.instance_name,
                fig_dir=fig_dir,
                **visualization_config(),
            )

        else:
            visualize_functions.show_heatmap(
                self.sorted_OT,
                title=title,
                category_name_list=self.source.category_name_list,
                num_category_list=self.source.num_category_list,
                x_object_labels=self.source.object_labels,
                y_object_labels=self.target.object_labels,
                fig_name=self.instance_name,
                fig_dir=fig_dir,
                **visualization_config(),
            )

    def plot_optimization_log(
        self, 
        fig_dir: Optional[str] = None, 
        visualization_config: VisualizationConfig = VisualizationConfig(),
    ):
        """
        Display a heatmap of the given matrix with various customization options.

        Args:
            fig_dir (Optional[str], optional):
                Directory to save the heatmap. If None, the heatmap won't be saved.
        """
        # get trial history
        df_trial = self.study.trials_dataframe()
        
        params = visualization_config()
        lim_acc = params.get("lim_acc", None)
        lim_eps = params.get("lim_eps", None)
        lim_gwd = params.get("lim_gwd", None)
        
        if lim_gwd is not None:
            df_trial = df_trial[(df_trial["value"] > lim_gwd[0]) & (df_trial["value"] < lim_gwd[1])]
        
        if lim_eps is not None:
            df_trial = df_trial[(df_trial["params_eps"] > lim_eps[0]) & (df_trial["params_eps"] < lim_eps[1])]
            
        if lim_acc is not None:
            df_trial = df_trial[(df_trial["user_attrs_best_acc"] > lim_acc[0]) & (df_trial["user_attrs_best_acc"] < lim_acc[1])]
            
        if fig_dir is not None:
            os.makedirs(fig_dir, exist_ok=True)
        
        visualize_functions.plot_optimization_log(
            df_trial,
            self.pair_name.replace('_', ' '),
            self.config.eps_list,
            fig_dir, 
            **visualization_config(),
        )

    def _run_gwot_no_entropy(
        self,
        sim_mat1:np.ndarray,
        sim_mat2:np.ndarray,
        p:np.ndarray,
        q:np.ndarray,
        ot_mat:np.ndarray,
        back_end:backend.Backend,
        max_iter:int=10000,
    ):
        """
        run the GWOT without entropy regularization.
        args:
            sim_mat1 : the similarity matrix of the source representation.
            sim_mat2 : the similarity matrix of the target representation.
            p : the probability distribution of the source representation.
            q : the probability distribution of the target representation.
            ot_mat : the initial OT matrix.
            back_end : the backend object to run the GWOT.
            max_iter : the number of iterations to run
        """
 
        sim_mat1, sim_mat2, p, q, ot_mat = back_end(sim_mat1, sim_mat2, p, q, ot_mat)
        
        new_ot_mat, log0 = ot.gromov.gromov_wasserstein(
            sim_mat1,
            sim_mat2,
            p,
            q,
            loss_fun = 'square_loss',
            symmetric = None,
            log=True,
            armijo = False,
            G0 = ot_mat,
            max_iter = max_iter,
            tol_rel = 1e-9,
            tol_abs = 1e-9,
            verbose=False,
        )

        log0["ot"] = new_ot_mat

        return log0

    def run_gwot_no_entropy(
        self,
        num_seed: int = 1,
        max_iter: int = 10000,
    ):
        """
        test function of the GWOT without entropy regularization by starting random init matrix.
        args:
            num_seed : the number of seeds to test.
            max_iter : the number of iterations to run the GWOT.
        """
        
        back_end = backend.Backend("cpu", "numpy", "double")
        
        p = ot.unif(len(self.source.sim_mat))
        q = ot.unif(len(self.target.sim_mat))

        init_mat_builder = init_matrix.InitMatrix(len(self.source.sim_mat), len(self.target.sim_mat), p, q)
        seeds = np.arange(num_seed)
        init_mat_list = [init_mat_builder.make_initial_T("random", seed, tol=1e-8) for seed in seeds]

        top1_acc_list = []
        top5_acc_list = []
        gwd_list = []
        computed_seed = []
        
        save_path = os.path.join(self.results_dir, "gwot_no_entropy", "OT")
        os.makedirs(save_path, exist_ok=True)
        
       
        for idx, init_mat in tqdm(enumerate(init_mat_list)):
            save_ot_path = save_path + f"/gw_{idx}.npy"
            
            if os.path.exists(save_ot_path):
                continue
            else:
                log0 = self._run_gwot_no_entropy(
                    self.source.sim_mat, 
                    self.target.sim_mat, 
                    p, 
                    q, 
                    init_mat, 
                    back_end, 
                    max_iter,
                )
                
                gw = log0["ot"]
                gw_loss = log0["gw_dist"]
            
                back_end.save_computed_results(gw, save_path, idx)
                
                top1_acc = self._calc_matching_rate_with_eval_mat(gw, k=1, eval_mat=None, order="maximum")
                top5_acc = self._calc_matching_rate_with_eval_mat(gw, k=5, eval_mat=None, order="maximum")
                
                computed_seed.append(seeds[idx])
                top1_acc_list.append(top1_acc)
                top5_acc_list.append(top5_acc)
                gwd_list.append(gw_loss)
        
        if len(top1_acc_list) == 0:
            print(f"all {num_seed} seeds are already computed.", end="\r")
            
        else:
            new_df = pd.DataFrame({"seed": computed_seed, "top1": top1_acc_list, "top5": top5_acc_list, "gwd": gwd_list}, index=computed_seed)
        
        
        if not os.path.exists(os.path.dirname(save_path) + "/result.csv"):
            new_df.to_csv(os.path.dirname(save_path) + "/result.csv")
            return new_df
            
        else:
            df = pd.read_csv(os.path.dirname(save_path) + "/result.csv", index_col=0)
            if len(top1_acc_list) != 0:
                df = pd.concat([df, new_df])
                df.to_csv(os.path.dirname(save_path) + "/result.csv")
            
            else:
                if len(df) > num_seed:
                    df = df.iloc[:num_seed]
                    
            return df


class AlignRepresentations:
    """This object has methods for conducting N groups level analysis and corresponding results.

    This class has information of all pairs of representations. The class provides functionality
    to conduct group-level analysis across various representations and compute pairwise alignments between them.

    Attributes:
        config (OptimizationConfig):
            all the essential parameters for GWOT.
        pairwise_method (str):
            The method to compute pairwise alignment. You can choose "combination" or "permutation". Defaults to "combination".
        representations_list (List[Representation]):
            List of Representation. used in the "combination" method.
        source_list (List[Representation]):
            List of source Representation. used in the "permutation" method.
        target_list (List[Representation]):
            List of target Representation. used in the "permutation" method.
        histogram_matching (bool):
            This will adjust the histogram of target to that of source. Defaults to False.
        main_results_dir (Optional[str]):
            The main folder directory to save the results. Defaults to None.
        data_name (str):
            The name of the folder to save the result for each pair. Defaults to "NoDefined".
        pairs_computed (Optional[List[str]]):
            List of pairs that have been computed. Defaults to None.
        specific_eps_list (Optional[dict]):
            Dictionary to set specific eps_list for each representation. Defaults to None.
        
        RSA_corr (dict):
            Dictionary to store RSA correlation values. (Set internally)
        name_list (List[str]):
            List of names from the provided representations.
        pairs_index_list (List[Tuple[int, int]]):
            All possible pairwise combinations of representations.
    """

    def __init__(
        self,
        config: OptimizationConfig,
        pairwise_method:str = "combination",
        representations_list: List[Representation]=None,
        source_list: List[Representation] = None,
        target_list: List[Representation] = None,
        histogram_matching: bool = False,
        main_results_dir: Optional[str] = None,
        data_name: str = "NoDefined",
        pairs_computed: Optional[List[str]] = None,
        specific_eps_list: Optional[dict] = None,
    ) -> None:
        self.config = config
        self.data_name = data_name
        self.histogram_matching = histogram_matching
        
        print(f"data_name : {self.data_name}")

        self.main_results_dir = main_results_dir
        self.pairwise_method = pairwise_method
        
        self.main_figure_dir = os.path.join(self.main_results_dir, "Figure", self.config.init_mat_plan)
        
        self.RSA_corr = dict()
        
        if pairwise_method == "combination":
            assert representations_list is not None, "Please set representations_list."
            assert source_list is None and target_list is None, "Please set source_list and target_list to None."
            print(f"pairwise_method : {pairwise_method}")
            self.representations_list = representations_list
            self.name_list = [rep.name for rep in self.representations_list]
            self.pairs_index_list = list(itertools.combinations(range(len(self.representations_list)), 2))
        
        elif pairwise_method == "permutation":
            assert representations_list is None, "Please set representations_list to None."
            assert source_list is not None and target_list is not None, "Please set source_list and target_list."
            print(f"pairwise_method : {pairwise_method}")    
            self.source_list = source_list
            self.target_list = target_list
            self.name_list = [rep.name for rep in self.source_list + self.target_list]
            self.pairs_index_list = list(itertools.product(range(len(self.source_list)), range(len(self.target_list))))
        
        else:
            raise ValueError("pairwise_method must be 'combination' or 'permutation'.")
    
        self.set_specific_eps_list(specific_eps_list)
        self.set_pairs_computed(pairs_computed)
    
    # data for gwot
    def _make_pairwise(
        self,
        source: Representation,
        target: Representation,
    ) -> PairwiseAnalysis:
        """Create a PairwiseAnalysis object from two representations.

        pair_name will be automatically generated as "source_name_vs_target_name".
        pairwise_name will be automatically generated as "data_name_pair_name".
        directory of PairwiseAnalysis will be automatically generated as "main_results_dir/pairwise_name".

        Args:
            source (Representation): source representation
            target (Representation): target representation

        Returns:
            pairwise (PairwiseAnalysis): PairwiseAnalysis object.
        """

        # set information for the pairwise
        assert source.name != target.name, "source and target must be different."
        
        pair_name = f"{source.name}_vs_{target.name}"
        pairwise_name = self.data_name + "_" + pair_name
        pair_results_dir = self.main_results_dir + "/" + pairwise_name

        config_copy = copy.deepcopy(self.config)
        
        if self.specific_eps_list is not None:
            if source.name in self.specific_eps_list.keys():
                config_copy.eps_list = self.specific_eps_list[source.name]
            
            if target.name in self.specific_eps_list.keys():
                config_copy.eps_list = self.specific_eps_list[target.name]

            elif pair_name in self.specific_eps_list.keys():
                config_copy.eps_list = self.specific_eps_list[pair_name]

        # create PairwiseAnalysis object
        pairwise = PairwiseAnalysis(
            results_dir=pair_results_dir,
            config=config_copy,
            source=source,
            target=target,
            data_name=self.data_name,
            pair_name=pair_name,
            instance_name=pairwise_name,
        )
        
        print("pair:", pairwise.pair_name, "eps_list:", pairwise.config.eps_list)
        
        if self.histogram_matching:
            pairwise.match_sim_mat_distribution(return_data=False)
                    
        return pairwise
    
    def _get_pairwise_list(self, pair_list):
        pairwise_list = []
        
        if self.pairwise_method == "combination":
            for pair in pair_list:
                source = self.representations_list[pair[0]]
                target = self.representations_list[pair[1]]
                pairwise = self._make_pairwise(source, target)
                pairwise_list.append(pairwise)
        
        elif self.pairwise_method == "permutation":
            for source_idx, target_idx in pair_list:
                source = self.source_list[source_idx]
                target = self.target_list[target_idx]
                pairwise = self._make_pairwise(source, target)
                pairwise_list.append(pairwise)
            
        return pairwise_list    

    def set_pairs_computed(self, pairs_computed: Optional[List[str]]) -> None:
        """User can only re-run the optimization for specific pairs by using `set_pair_computed`.

        Args:
            pairs_computed (Optional[List[str]]):
                List of specific representation pairs to compute.  If not provided, optimization will be
                run for all pairs. Example values in the list might be: ["Group1", "Group2_vs_Group4"]

        Examples:
            >>> pairs_computed = ["Group1", "Group2_vs_Group4"]
            >>> align_representation.set_pair_computed(pairs_computed)
        """
        self.pairs_computed = pairs_computed
        if self.pairs_computed is not None:
            print("The pairs to compute was selected by pairs_computed...")
            self.specific_pair_list = self._specific_pair_list(pairs_computed)
            self.pairwise_list = self._get_pairwise_list(self.specific_pair_list)

        else:
            print("All the pairs in the list below will be computed. ")
            self.pairwise_list = self._get_pairwise_list(self.pairs_index_list)

    def set_specific_eps_list(
        self,
        specific_eps_list: Optional[Dict[str, List[float]]],
        specific_only: bool = False,
    ) -> None:
        """
        Also, user can re-define the epsilon range for some pairs by using `set_specific_eps_list`.

        The rest of them will be computed with `config.eps_list`.
        If `specific_only` is True (default is False), only these pairs will be computed and the rest of them were skipped.

        Args:
            specific_eps_list (Optional[Dict[str, List[float]]]):
                A dictionary specifying custom epsilon values for particular representation pairs.
                Key is the representation pair name and value is a list of epsilon values.
            specific_only (bool):
                If True, only pairs specified in `specific_eps_list` will be computed. Defaults to False.

        Examples:
            >>> specific_eps_list = {'Group1': [0.02, 0.1], "Group2_vs_Group4":[0.1, 0.2]}
            >>> align_representation.set_specific_eps_list(specific_eps_list, specific_only=False)
        """

        self.specific_eps_list = specific_eps_list

        if specific_eps_list is not None:
            assert isinstance(self.specific_eps_list, dict), "specific_eps_list needs to be dict."
            
            print("The range of epsilon for some pairs in the list below was changed ...")

            self.specific_pair_list = self._specific_pair_list(specific_eps_list)

            if specific_only:
                self.pairwise_list = self._get_pairwise_list(self.specific_pair_list)
            else:
                self.pairwise_list = self._get_pairwise_list(self.pairs_index_list)

    def _specific_pair_list(self, pair_list):
        if isinstance(pair_list, dict):
            key_loop = pair_list.keys()
        elif isinstance(pair_list, list):
            key_loop = pair_list
        
        if self.pairwise_method == "permutation":
            source_name_list = [source.name for source in self.source_list]
            target_name_list = [target.name for target in self.target_list]
            
        specific_pair_list = []
        for key in key_loop:
            if "_vs_" in key:
                source_name, target_name = key.split("_vs_")

                if self.pairwise_method == "combination":
                    source_idx = self.name_list.index(source_name)
                    target_idx = self.name_list.index(target_name)
                
                elif self.pairwise_method == "permutation":
                    source_idx = source_name_list.index(source_name)
                    target_idx = target_name_list.index(target_name)

                rep_list = [(source_idx, target_idx)]

            else:
                
                if self.pairwise_method == "combination":
                    rep_idx = self.name_list.index(key)
                    rep_list = [nn for nn in self.pairs_index_list if rep_idx in nn]
                
                elif self.pairwise_method == "permutation":
                    if key in source_name_list:
                        source_idx = source_name_list.index(key)
                        rep_list = [(source_idx, idx) for idx in range(len(target_name_list))]
                    
                    elif key in target_name_list:
                        target_idx = target_name_list.index(key)
                        rep_list = [(idx, target_idx) for idx in range(len(source_name_list))]
                    
                    else:
                        raise ValueError("the specific group must be in source_name_list or target_name_list.")

            specific_pair_list.extend(rep_list)

        return specific_pair_list
    
    def show_sim_mat(
        self,
        sim_mat_format: str = "default",
        visualization_config: VisualizationConfig = VisualizationConfig(),
        visualization_config_hist: VisualizationConfig = VisualizationConfig(),
        fig_dir: Optional[str] = None,
        show_distribution: bool = False,
    ) -> None:
        """Show the dissimilarity matrix of the representation.

        Args:
            sim_mat_format (str, optional):
                "default" or "sorted". If "sorted" is selected, the rearranged matrix is shown. Defaults to "default".
            visualization_config (VisualizationConfig, optional):
                container of parameters used for figure. Defaults to VisualizationConfig().
            visualization_config_hist (VisualizationConfig, optional):
                container of parameters used for histogram figure. Defaults to VisualizationConfig().
            fig_dir (Optional[str], optional):
                The directory for saving the figure. Defaults to None.
            show_distribution (bool, optional):
                show the histogram figures. Defaults to False.
        """

        if self.pairwise_method == "combination":
            rep_list = self.representations_list
        elif self.pairwise_method == "permutation":
            rep_list = self.source_list + self.target_list
        
        if fig_dir is None:
            fig_dir = os.path.join(self.main_figure_dir, "individual_sim_mat")
            os.makedirs(fig_dir, exist_ok=True)
        
        if sim_mat_format == "default":
            return_sorted = False
        elif sim_mat_format == "sorted":
            return_sorted = True
        else:
            raise ValueError("sim_mat_format must be 'default' or 'sorted'.")

        for rep in rep_list:
            rep.plot_sim_mat(
                fig_dir=fig_dir,
                return_sorted=return_sorted,
                visualization_config=visualization_config,
            )
            
            if show_distribution:
                rep.show_sim_mat_distribution(visualization_config = visualization_config_hist)
    
    # RSA methods
    def RSA_get_corr(self, metric: str = "pearson", return_data = False) -> None:
        """Conventional representation similarity analysis (RSA).

        Args:
            metric (str, optional):
                spearman or pearson. Defaults to "pearson".
        """
        for pairwise in self.pairwise_list:
            corr = pairwise.rsa(metric=metric)
            self.RSA_corr[pairwise.pair_name] = corr
            print(f"Correlation {pairwise.pair_name.replace('_', ' ')} : {corr}")
        
        if return_data:
            return self.RSA_corr
    
    # entropic GWOT
    def gw_alignment(
        self,
        compute_OT: bool = False,
        delete_results: bool = False,
        return_data: bool = False,
        return_figure: bool = False,
        OT_format: str = "default",
        visualization_config: VisualizationConfig = VisualizationConfig(),
        show_log: bool = False,
        fig_dir: Optional[str] = None,
        save_dataframe: bool = False,
        change_sampler_seed: bool = False,
        sampler_seed: int = 42,
        fix_random_init_seed: bool = False,
        first_random_init_seed: Optional[int] = None,
        parallel_method :str = "multiprocess",
        delete_confirmation: bool = True,
    ) -> Optional[List[np.ndarray]]:
        """
        compute GWOT for each pair.

        Args:
            compute_OT (bool, optional):
                If True, the GWOT will be computed.
                If False, the saved results will be loaded. Defaults to False.

            delete_results (bool, optional):
                If True, all the saved results will be deleted. Defaults to False.

            return_data (bool, optional):
                return the computed OT. Defaults to False.

            return_figure (bool, optional):
                make the result figures or not. Defaults to True.

            OT_format (str, optional):
                format of sim_mat to visualize.
                Options are "default", "sorted", and "both". Defaults to "default".

            visualization_config (VisualizationConfig, optional):
                container of parameters used for figure. Defaults to VisualizationConfig().

            show_log (bool, optional):
                If True, the evaluation figure of GWOT will be made. Defaults to False.

            fig_dir (Optional[str], optional):
                you can define the path to which you save the figures (.png).
                If None, the figures will be saved in the same subfolder in "results_dir".
                Defaults to None.

            save_dataframe (bool, optional):
                If True, you can save all the computed data stored in SQlite or PyMySQL in csv
                format (pandas.DataFrame) in the result folder.  Defaults to False.

            change_sampler_seed (bool, optional):
                If True, the random seed will be different for each pair. Defaults to False.

            fix_sampler_seed (int, optional):
                The random seed value for optuna sampler. Defaults to 42.

            parallel_method (str, optional):
                parallel method to compute GWOT. Defaults to "multithread".
                perhaps, multithread may be better to compute fast, because of Optuna's regulations.

        Returns:
            OT_list (Optional[List[np.ndarray]]):
                Returns a list of computed Optimal Transport matrices if `return_data` is True. Otherwise, returns None.
        """

        # check the n_jobs and sampler_seed
        utils_functions.check_parameters(self.config.n_jobs, sampler_seed)
        
        if delete_results:
            self.drop_gw_alignment_files(drop_all=True, delete_database=True, delete_directory=True, delete_confirmation=delete_confirmation)

        # compute the entropic GWOT in parallel.
        if compute_OT:
            self._entropic_gwot(
                parallel_method,
                sampler_seed,
                change_sampler_seed,
                save_dataframe,
                fix_random_init_seed,
                first_random_init_seed,
            )
            
        if return_figure:
            self.show_OT(OT_format, fig_dir, visualization_config)
        
        if show_log:
            self.show_optimization_log(fig_dir, visualization_config)
        
        if return_data:
            if OT_format == "sorted":
                OT_list = [pairwise.sorted_OT for pairwise in self.pairwise_list]
            elif OT_format == "default":
                OT_list = [pairwise.OT for pairwise in self.pairwise_list]
            else:
                raise ValueError("OT_format must be 'default' or 'sorted'.")
            return OT_list
   
    def _entropic_gwot(
        self,
        parallel_method: str,
        sampler_seed: int,
        change_sampler_seed: bool,
        save_dataframe :bool,
        fix_random_init_seed :bool,
        first_random_init_seed :int,
    ):
        """
        compute the entropic GWOT in parallel.
        
        Args:
            parallel_method (str):
                parallel method to compute GWOT.
                perhaps, multiprocess may be better to compute fast, because of Optuna's regulations.
            sampler_seed (int):
                the random seed value for optuna sampler.
            change_sampler_seed (bool):
                If True, the random seed will be different for each pair.
            save_dataframe (bool):
                If True, you can save all the computed data stored in SQlite or PyMySQL in csv
                format (pandas.DataFrame) in the result folder.
            fix_random_init_seed (bool):
                If True, the random seed will be fixed for each pair.
            first_random_init_seed (int):
                the random seed value for the first initial matrix.
            
        """
        if parallel_method == "multiprocess":
            pool = ProcessPoolExecutor(self.config.n_jobs)
        
        elif parallel_method == "multithread":
            pool = ThreadPoolExecutor(self.config.n_jobs)
        
        else:
            raise ValueError("parallel_method must be 'multiprocess' or 'multithread'.")
        
        with mp.Manager() as manager:
            queue = manager.Queue()
        
            with pool:
                processes = []
                for idx, pairwise in enumerate(self.pairwise_list):

                    if change_sampler_seed:
                        sampler_seed = sampler_seed + idx
                    else:
                        sampler_seed = sampler_seed

                    future = pool.submit(
                        pairwise.run_entropic_gwot,
                        compute_OT=True,
                        save_dataframe=save_dataframe,
                        target_device=self.config.device,
                        sampler_seed=sampler_seed,
                        fix_random_init_seed=fix_random_init_seed,
                        first_random_init_seed=first_random_init_seed,
                        iter_from_ComputeGWOT=idx,
                        queue=queue,
                    )

                    processes.append(future)

                for future in as_completed(processes):
                    future.result()

    def drop_gw_alignment_files(
        self,
        rep_or_pair_name_list: Optional[List[str]] = None,
        drop_all: bool = False,
        delete_database: bool = False,
        delete_directory: bool = False,
        delete_confirmation: bool = False,
    ) -> None:
        """Delete the specified database and directory with the given filename and sampler name.

        Args:
            rep_or_pair_name_list (Optional[List[str]], optional):
                List of representation or pairwise names to delete. (e.g. rep_or_pair_name_list = ["Group1", "Group2_vs_Group4"])
                If `drop_all` is set to True, this parameter is ignored. Defaults to None.
            drop_all (bool, optional):
                If set to True, all results will be deleted regardless of the `drop_filenames` parameter.
                Defaults to False.
            delete_database (bool, optional):
                If set to True, the database will be deleted. Defaults to False.
            delete_directory (bool, optional):
                If set to True, the directory will be deleted. Defaults to False.

        Raises:
            ValueError: If neither `drop_filenames` is specified nor `drop_all` is set to True.
        """
        
        all_drop_list = [pairwise.study_name for pairwise in self.pairwise_list]
        
        if drop_all:
            drop_list = all_drop_list
            
            if delete_confirmation:
                confirm = input(f"Are you sure you want to delete all the results of following list in {self.main_results_dir}?:\n{all_drop_list}\n(yes/no) : ")
                
                if confirm == "yes":
                    pass
                else:
                    return None
        
        else:
            assert rep_or_pair_name_list is not None, "Specify the results name in drop_filenames or set drop_all=True"
            
            drop_idx_list = []
            for name in rep_or_pair_name_list:
                drop_idx = [idx for idx, pairname in enumerate(all_drop_list) if name in pairname]
                drop_idx_list.extend(drop_idx)
            
            drop_list = [all_drop_list[n] for n in set(drop_idx_list)]    

        # delete files for each pairwise
        for pairwise in self.pairwise_list:
            if pairwise.study_name not in drop_list:
                continue        
            if not sqlalchemy_utils.database_exists(pairwise.storage):
                continue
            
            self._delete_figure(pairwise)
            pairwise.delete_previous_results(
                delete_database=delete_database, 
                delete_directory=delete_directory
            )
    
    def _delete_figure(self, pairwise:PairwiseAnalysis):        
        for root, _, files in os.walk(self.main_figure_dir, topdown=False):
            for file in files:
                if pairwise.pair_name in file:
                    os.remove(os.path.join(root, file))
    
    def show_OT(
        self, 
        OT_format: str = "default", 
        fig_dir: Optional[str] = None, 
        visualization_config: VisualizationConfig = VisualizationConfig()
    ) -> None:
        """
        Show the OT for each pair.

        Args:
            OT_format (str, optional): 
                format of sim_mat to visualize. Options are "default" or "sorted". Defaults to "default".
            fig_dir (Optional[str], optional): 
                The directory for saving the figure. Defaults to None.
            visualization_config (VisualizationConfig, optional):
                container of parameters used for figure. Defaults to VisualizationConfig().
        """
        if fig_dir is None:
            fig_dir = os.path.join(self.main_figure_dir, "OT")
            os.makedirs(fig_dir, exist_ok=True)
        
        for pairwise in self.pairwise_list:
                pairwise.plot_OT(
                    return_sorted=False if OT_format == "default" else True,
                    fig_dir=fig_dir,
                    visualization_config=visualization_config,
                )

    def show_optimization_log(
        self,
        fig_dir: Optional[str] = None,
        visualization_config: VisualizationConfig = VisualizationConfig()
    ) -> None:
        """
        Show both the relationships between epsilons and GWD, and between accuracy and GWD
        
        Args:
            fig_dir (Optional[str], optional):
                The directory for saving the figure. Defaults to None.
            visualization_config (VisualizationConfig, optional):
                container of parameters used for figure. Defaults to VisualizationConfig().
        """
        if fig_dir is None:
            fig_dir = os.path.join(self.main_figure_dir, "log")
            os.makedirs(fig_dir, exist_ok=True)
            
        for pairwise in self.pairwise_list:
            pairwise.plot_optimization_log(
                fig_dir=fig_dir,
                visualization_config=visualization_config,
            )

    def calc_accuracy(
        self,
        top_k_list: List[int],
        eval_type: str = "ot_plan",
        metric: str = "cosine",
        category_mat: Optional[Any] = None,
        barycenter: bool = False,
        return_dataframe: bool = False
    ) -> Optional[pd.DataFrame]:
        """Evaluation of the accuracy of the unsupervised alignment

        Args:
            top_k_list (List[int]):
                define the top k accuracy in list

            eval_type (str, optional):
                two ways to evaluate the accuracy as above. Defaults to "ot_plan".

            eval_mat (Optional[Any], optional):
                This will be used for the category info. Defaults to None.
            
            metric (str, optional):
                The metric used to calculate the distance between the representations. Defaults to "cosine".

            barycenter (bool, optional):
                Indicates if the accuracy should be evaluated with respect to a barycenter representation.
                Defaults to False.

            return_dataframe (bool, optional):
                If True, the accuracy result will be returned in pandas.DataFrame format. Defaults to False.

        Returns:
            accuracy (Optional[pd.DataFrame]):
                A DataFrame containing accuracy metrics. Only returned if `return_dataframe` is True.
        """

        accuracy = pd.DataFrame()
        accuracy["top_n"] = top_k_list

        for pairwise in self.pairwise_list:
            df = pairwise.calc_matching_rate(
                top_k_list, 
                eval_type=eval_type,
                category_mat=category_mat, 
                metric=metric, 
                barycenter=barycenter, 
            )

            accuracy = pd.merge(accuracy, df, on="top_n")

        accuracy = accuracy.set_index("top_n")

        if eval_type == "ot_plan":
            self.top_k_accuracy = accuracy
            print("Top k accuracy : \n", accuracy)

        elif eval_type == "k_nearest":
            self.k_nearest_matching_rate = accuracy
            print("K nearest matching rate : \n", accuracy)

        elif eval_type == "category":
            self.category_level_accuracy = accuracy
            print("category level accuracy : \n", accuracy)

        print("\nMean : \n", accuracy.iloc[:, 1:].mean(axis="columns"))

        if return_dataframe:
            return accuracy

    def _get_dataframe(self, eval_type="ot_plan", concat=True):
        if eval_type == "ot_plan":
            df = self.top_k_accuracy
        elif eval_type == "k_nearest":
            df = self.k_nearest_matching_rate
        elif eval_type == "category":
            df = self.category_level_accuracy

        cols = [col for col in df.columns if "top_n" not in col]
        df = df[cols]

        if concat:
            df = pd.concat([df[i] for i in df.columns], axis=0)
            df = df.rename("matching rate")
        return df

    def plot_accuracy(
        self,
        eval_type: str = "ot_plan",
        fig_dir: Optional[str] = None,
        fig_name: str = "Accuracy_ot_plan.png",
        scatter: bool = True
    ) -> None:
        """
        Plot the accuracy of the unsupervised alignment for each top_k

        Args:
            eval_type (str, optional):
                Specifies the method used to evaluate accuracy. Can be "ot_plan", "k_nearest", or "category".
                Defaults to "ot_plan".

            fig_dir (Optional[str], optional):
                Directory path where the generated figure will be saved. If None, the figure will not be saved
                but displayed. Defaults to None.

            fig_name (str, optional):
                Name of the saved figure if `fig_dir` is specified. Defaults to "Accuracy_ot_plan.png".

            scatter (bool, optional):
                If True, the accuracy will be visualized as a swarm plot. Otherwise, a line plot will be used.
                Defaults to True.
        """
        
        # default setting
        plt.rcParams.update(plt.rcParamsDefault)
        styles = matplotlib.style.available
        darkgrid_style = [s for s in styles if re.match(r"seaborn-.*-darkgrid", s)][0]
        plt.style.use(darkgrid_style)
        
        plt.figure(figsize=(5, 3))

        if scatter:
            df = self._get_dataframe(eval_type, concat=True)
            sns.set_style("darkgrid")
            sns.set_palette("pastel")
            sns.swarmplot(data=pd.DataFrame(df), x="top_n", y="matching rate", size=5, dodge=True)

        else:
            df = self._get_dataframe(eval_type, concat=False)
            for group in df.columns:
                plt.plot(df.index, df[group], c="blue")

        plt.ylim(0, 100)
        plt.title(eval_type)
        plt.xlabel("top k")
        plt.ylabel("Matching rate (%)")

        plt.tick_params(axis="both", which="major")
        plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2)
        if fig_dir is not None:
            plt.savefig(os.path.join(fig_dir, fig_name))
        plt.show()

        plt.clf()
        plt.close()

    def get_highest_top1_matching_rate(self):
        """
        Get the highest top1 matching rate for each pair.

        Returns:
            highest_top1 (pd.DataFrame):
                A DataFrame containing the highest top1 matching rate for each pair.
        """
        highest_top1_list = []
        for pairwise in self.pairwise_list:
            df = pairwise.study.trials_dataframe()
            top1 = df["user_attrs_best_acc"].max() * 100
            highest_top1_list.append(top1)
        
        highest_top1 = pd.DataFrame(
            index = [pairwise.pair_name for pairwise in self.pairwise_list], 
            data={"top1": highest_top1_list},
        )
        
        return highest_top1
    
    def _procrustes_to_pivot(self, pivot):
        the_others = set()
        pivot_idx_list = [] # [pair_idx, paivot_idx]
        for i, pair in enumerate(self.pairs_index_list):
            if pivot in pair:
                the_others.add(filter(lambda x: x != pivot, pair))

                pivot_idx = pair.index(pivot)
                pivot_idx_list.append([i, pivot_idx])

        # check whether 'pair_list' includes all pairs between the pivot and the other Representations
        assert len(the_others) == len(self.representations_list) - 1, "'pair_list' must include all pairs between the pivot and the other Representations."

        for pair_idx, pivot_idx in pivot_idx_list:
            pairwise = self.pairwise_list[pair_idx]

            if pivot_idx == 0: # when the pivot is the source of the pairwise
                source_idx = 1
                OT = pairwise.OT.T

            elif pivot_idx == 1:# when the pivot is the target of the pairwise
                source_idx = 0
                OT = pairwise.OT

            pivot = (pairwise.source, pairwise.target)[pivot_idx]
            source = (pairwise.source, pairwise.target)[source_idx]

            source.embedding = pairwise.procrustes(pivot.embedding, source.embedding, OT)

    def visualize_embedding(
        self,
        dim: int,
        method: Optional[str] = "PCA",
        emb_transformer: Optional[TransformerMixin] = None,
        pivot: Union[None, int, str] = 0,
        visualization_config: VisualizationConfig = VisualizationConfig(),
        category_name_list: Optional[List[str]] = None,
        num_category_list: Optional[List[int]] = None,
        category_idx_list: Optional[List[int]] = None,
        title: Optional[str] = None,
        legend: bool = True,
        fig_dir: Optional[str] = None,
        fig_name: str = "Aligned_embedding",
        **kwargs,
    ) -> Optional[Union[plt.Figure, List[np.ndarray]]]:
        """
        Visualizes the aligned embedding in the specified number of dimensions.

        Args:
            dim (int):
                The number of dimensions in which the points are embedded.
            method (str, optional):
                The method used to reduce the dimensionality of the embedding. Options include "PCA", "TSNE", "Isomap" and "MDS".
            emb_transformer (Optional[TransformerMixin], optional):
                Parameters used for the dimensionality reduction method.
                See sklearn documentation for details. Defaults to None.
            pivot (Union[None, int, str], optional):
                The index of the pivot Representation or the name of the pivot Representation for Procrustes.
                If None, no Procrustes analysis was done, and PCA will be done on just concatenated (in axis=0) data. Defaults to 0.
            visualization_config (VisualizationConfig, optional):
                Container of parameters used for figure. Defaults to VisualizationConfig().
            category_name_list (Optional[List[str]], optional):
                List of category names. Defaults to None.
            num_category_list (Optional[List[int]], optional):
                List of the number of categories. Defaults to None.
            category_idx_list (Optional[List[int]], optional):
                List of the indices of the categories. Defaults to None.
            title (Optional[str], optional):
                Title of the figure. Defaults to None.
            legend (bool, optional):
                If True, the legend will be displayed. Defaults to True.
            fig_dir (Optional[str], optional):
                Directory path where the generated figure will be saved. If None, the figure will not be saved. Defaults to None.
            fig_name (str, optional):
                Name of the saved figure if `fig_dir` is specified. Defaults to "Aligned_embedding.png".
            **kwargs:
                Additional arguments for the dimensionality reduction method.

        Returns:
            Optional[Union[plt.Figure, List[np.ndarray]]]:
            If `returned` is "figure", the function visualizes the plot and optionally saves it based
            on the `fig_dir` parameter. If `returned` is "row_data", a list of embeddings is returned.
        """

        if fig_dir is None:
            fig_dir = os.path.join(self.main_figure_dir, "visualize_embedding")
            os.makedirs(fig_dir, exist_ok=True)
        
        name_list = []
        embedding_list = []
        
        # chooose pivot.  
        if pivot == None:
            pass
        
        elif isinstance(pivot, int):
            self._procrustes_to_pivot(pivot)
            fig_name = "procrustes"
        
        elif pivot == "barycenter":
            assert self.barycenter is not None
            self._procrustes_to_pivot(pivot)
            fig_name = "barycenter"
        
        else:
            raise ValueError("pivot must be None, int or 'barycenter'.")
            
        # get sort idx
        if category_idx_list is not None:
            print("New category information is given.")
            sort_idx = np.concatenate(category_idx_list)  
        elif self.representations_list[0].category_idx_list is not None:
            print("Category information was already given.")
            sort_idx = np.concatenate(self.representations_list[0].category_idx_list)        
        else:
            print("No category information is given.")
            sort_idx = np.arange(self.representations_list[0].embedding.shape[0]) # This means that the order of data is not changed
        
        # sort the embedding after pivot.
        for i in range(len(self.representations_list)):
            name_list.append(self.representations_list[i].name)
            embedding_list.append(self.representations_list[i].embedding[sort_idx, :])
        
        if embedding_list[0].shape[1] > dim:
            # obtain the reduced dimension embedding list
            embedding_list, _ = utils_functions.obtain_embedding(
                embedding_list,
                dim=dim, 
                emb_name=method, 
                emb_transformer=emb_transformer,
                **kwargs,
            )
        
        if category_idx_list is None:
            if self.representations_list[0].category_idx_list is not None:
                category_name_list = self.representations_list[0].category_name_list
                num_category_list = self.representations_list[0].num_category_list
        
        visualize_functions.plot_embedding(
            embedding_list=embedding_list,
            dim=dim,
            name_list=name_list,
            category_name_list=category_name_list,
            num_category_list=num_category_list,
            title=title, 
            has_legend=legend,
            fig_name=fig_name,
            fig_dir=fig_dir, 
            **visualization_config(),
        )

    def calc_barycenter(self, X_init=None):
        embedding_list = [representation.embedding for representation in self.representations_list]

        if X_init is None:
            X_init = np.mean(embedding_list, axis=0)  # initial Dirac locations

        b = ot.unif(len(X_init))  # weights of the barycenter

        weights_list = []  # measures weights
        for representation in self.representations_list:
            weights = ot.unif(len(representation.embedding))
            weights_list.append(weights)

        # new location of the barycenter
        X = ot.lp.free_support_barycenter(embedding_list, weights_list, X_init, b)

        return X

    def barycenter_alignment(
        self,
        pivot: int,
        n_iter: int,
        return_data: bool = False,
        OT_format: str = "default",
        metric: str = "cosine",
        visualization_config: VisualizationConfig = VisualizationConfig(),
        fig_dir: Optional[str] = None,
    ) -> Optional[List[np.ndarray]]:
        """The unuspervised alignment method using Wasserstein barycenter proposed by Lian et al. (2021).

        Args:
            pivot (int):
                The representation to which the other representations are aligned initially using gw alignment
            n_iter (int):
                The number of iterations to calculate the location of the barycenter.
            return_data (bool, optional):
                If True, returns the computed OT. Defaults to False.
            OT_format (str, optional):
                Format of similarity matrix to visualize. Options include "default", "sorted", and "both".
                Defaults to "default".
            metric (str, optional):
                The metric used to calculate the Wasserstein distance. Defaults to "cosine".
            visualization_config (VisualizationConfig, optional):
                Container of parameters used for figure. Defaults to VisualizationConfig().
            fig_dir (Optional[str], optional):
                Directory where figures are saved. Defaults to None.

        Returns:
            Optional[List[np.ndarray]]:
                If return_data is True, returns the computed OT.
        """

        assert self.pairwise_method == "combination", "pairwise_method must be 'combination'."
        
        # Select the pivot
        pivot_representation = self.representations_list[pivot]
        others_representaions = self.representations_list[:pivot] + self.representations_list[pivot + 1 :]
        
        pair_name_list = [pair.pair_name for pair in self.pairwise_list]

        # # GW alignment to the pivot
        for representation in others_representaions:
            
            pair_name = f"{representation.name}_vs_{pivot_representation.name}"
            instance_name = self.data_name + "_" + pair_name
            pair_results_dir = self.main_results_dir + "/" + instance_name
            
            pairwise = PairwiseAnalysis(
                data_name=self.data_name,
                results_dir=pair_results_dir,
                config=self.config,
                source=representation,
                target=pivot_representation,
                pair_name=pair_name,
                instance_name=instance_name,
            )
            
            # check if the pair is already computed
            if pair_name in pair_name_list:
                representation.embedding = pairwise.get_new_source_embedding()
            
            elif f"{pivot_representation.name}_vs_{representation.name}" in pair_name_list:
                OT_t = self.pairwise_list[pair_name_list.index(f"{pivot_representation.name}_vs_{representation.name}")].OT
                representation.embedding = pairwise.procrustes(representation.embedding, pivot_representation.embedding, OT_t)
            
            else:
                # Compute GWOT
                raise ValueError(f"Pairwise {pair_name} is not computed yet. Please compute it first.")
            

        # Set up barycenter
        init_embedding = self.calc_barycenter()
        self.barycenter = Representation(
            name="barycenter",
            embedding=init_embedding,
            object_labels=pivot_representation.object_labels,
            category_name_list=pivot_representation.category_name_list,
            num_category_list=pivot_representation.num_category_list,
            category_idx_list=pivot_representation.category_idx_list,
            func_for_sort_sim_mat=pivot_representation.func_for_sort_sim_mat,
            save_conditional_rdm_path=pivot_representation.save_conditional_rdm_path,

        )

        # Set pairwises whose target are the barycenter
        pairwise_barycenters = []
        for representation in self.representations_list:
            
            pair_name = f"{representation.name}_vs_{self.barycenter.name}"
            instance_name = self.data_name + "_" + pair_name
            pair_results_dir = self.main_results_dir + "/" + instance_name
            
            pairwise = PairwiseAnalysis(
                data_name=self.data_name,
                results_dir=pair_results_dir,
                config=self.config,
                source=representation,
                target=self.barycenter,
                pair_name=pair_name,
                instance_name=instance_name,
            )
            pairwise_barycenters.append(pairwise)

        # Barycenter alignment
        loss_list = []
        embedding_barycenter = init_embedding
        for i in range(n_iter):
            embedding_barycenter = self.calc_barycenter(X_init=embedding_barycenter)

            loss = 0
            for pairwise in pairwise_barycenters:
                # update the embedding of the barycenter
                pairwise.target.embedding = embedding_barycenter

                # OT to the barycenter
                loss += pairwise.wasserstein_alignment(metric=metric)

                # update the embeddings of each representation
                pairwise.source.embedding = pairwise.get_new_source_embedding()

            loss /= len(pairwise_barycenters)
            loss_list.append(loss)

        plt.figure()
        plt.plot(loss_list)
        plt.xlabel("iteration")
        plt.ylabel("Mean Wasserstein distance")

        # replace OT of each pairwise by the product of OTs to the barycenter
        self._get_OT_all_pair(pairwise_barycenters)

        # visualize
        if fig_dir is None:
            fig_dir = os.path.join(self.main_figure_dir, "barycenter_alignment")
            os.makedirs(fig_dir, exist_ok=True)
        
        for pairwise in pairwise_barycenters:
            pairwise.plot_OT(
                return_sorted = False if OT_format == "default" else True,
                fig_dir=fig_dir,
                visualization_config=visualization_config,                
            )

        if return_data:
            OT_list = [pairwise.OT for pairwise in pairwise_barycenters]
            return OT_list
    
    def _get_OT_all_pair(self, pairwise_barycenters):
        # replace OT of each pairwise by the product of OTs to the barycenter
        pairs = list(itertools.combinations(pairwise_barycenters, 2))

        for i, (pairwise_1, pairwise_2) in enumerate(pairs):
            OT = np.matmul(pairwise_2.OT, pairwise_1.OT.T)
            OT *= len(OT)  # normalize
            self.pairwise_list[i].OT = OT
