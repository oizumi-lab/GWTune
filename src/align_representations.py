# %%
import itertools
from pathlib import Path
import os
import sys
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Union, Optional
import copy

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import seaborn as sns
import torch
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr
from sklearn import manifold
from sqlalchemy import create_engine, URL
from sqlalchemy_utils import create_database, database_exists, drop_database
import glob
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from .gw_alignment import GW_Alignment
from .histogram_matching import SimpleHistogramMatching
from .utils import gw_optimizer, visualize_functions, backend

# %%
class OptimizationConfig:
    def __init__(
        self,
        eps_list=[1, 10],
        eps_log=True,
        num_trial=4,
        sinkhorn_method='sinkhorn',
        device="cpu",
        to_types="numpy",
        data_type="double",
        n_jobs=1,
        multi_gpu: Union[bool, List[int]] = False,
        db_params={"drivername": "mysql", "username": "root", "password": "", "host": "localhost", "port": 3306},
        init_mat_plan="random",
        user_define_init_mat_list = None,
        n_iter=1,
        max_iter=200,
        sampler_name="tpe",
        pruner_name="hyperband",
        pruner_params={"n_startup_trials": 1, "n_warmup_steps": 2, "min_resource": 2, "reduction_factor": 3},
    ) -> None:
        """
        
        This is an instance for sharing the parameters to compute GWOT with the instance PairwiseAnalysis.
        
        Please check the tutoial.ipynb for detailed info.

        Args:
            eps_list (list, optional): _description_. Defaults to [1, 10].
            eps_log (bool, optional): _description_. Defaults to True.
            num_trial (int, optional): _description_. Defaults to 4.
            sinkhorn_method (str, optional): _description_. Defaults to 'sinkhorn'.
            device (str, optional): _description_. Defaults to "cpu".
            to_types (str, optional): _description_. Defaults to "numpy".
            data_type (str, optional): _description_. Defaults to "double".
            n_jobs (int, optional): _description_. Defaults to 1.
            multi_gpu (Union[bool, List[int]], optional): _description_. Defaults to False.
            db_params (dict, optional): _description_. Defaults to {"drivername": "mysql", "username": "root", "password": "", "host": "localhost", "port": 3306}.
            init_mat_plan (str, optional): _description_. Defaults to "random".
            n_iter (int, optional): _description_. Defaults to 1.
            max_iter (int, optional): _description_. Defaults to 200.
            data_name (str, optional): _description_. Defaults to "THINGS".
            sampler_name (str, optional): _description_. Defaults to "tpe".
            pruner_name (str, optional): _description_. Defaults to "hyperband".
            pruner_params (dict, optional): _description_. Defaults to {"n_startup_trials": 1, "n_warmup_steps": 2, "min_resource": 2, "reduction_factor": 3}.
            user_define_init_mat_list (_type_, optional): _description_. Defaults to None.
        """
        self.eps_list = eps_list
        self.eps_log = eps_log
        self.num_trial = num_trial
        self.sinkhorn_method = sinkhorn_method

        self.to_types = to_types
        self.data_type = data_type
        self.device = device

        self.n_jobs = n_jobs
        self.multi_gpu = multi_gpu

        self.db_params = db_params

        self.init_mat_plan = init_mat_plan
        self.n_iter = n_iter
        self.max_iter = max_iter

        self.sampler_name = sampler_name
        self.user_define_init_mat_list = user_define_init_mat_list

        self.pruner_name = pruner_name
        self.pruner_params = pruner_params

class VisualizationConfig:
    def __init__(
        self,
        show_figure=True,
        figsize=(8, 6),
        cbar_ticks_size=5,
        cbar_format=None,
        ticks_size=5,
        xticks_rotation=90,
        yticks_rotation=0,
        title_size=20,
        legend_size=5,
        xlabel=None,
        xlabel_size=15,
        ylabel=None,
        ylabel_size=15,
        zlabel=None,
        zlabel_size=15,
        color_labels=None,
        color_hue=None,
        colorbar_label=None,
        colorbar_range=[0, 1],
        colorbar_shrink=1,
        markers_list=None,
        marker_size=30,
        color = 'C0',
        cmap = 'cividis',
        ot_object_tick=False,
        ot_category_tick=False,
        draw_category_line=False,
        category_line_color='C2',
        category_line_alpha=0.2,
        category_line_style='dashed',
        plot_eps_log=False,
        lim_eps=None,
        lim_gwd=None,
        lim_acc=None,
    ) -> None:
        """
        This is an instance for sharing the parameters to make the figures of GWOT with the instance PairwiseAnalysis.
        
        Please check the tutoial.ipynb for detailed info.

        Args:
            show_figure (bool, optional): _description_. Defaults to True.
            figsize (tuple, optional): _description_. Defaults to (8, 6).
            cbar_ticks_size (int, optional): _description_. Defaults to 5.
            cbar_format (_type_, optional): _description_. Defaults to None.
            ticks_size (int, optional): _description_. Defaults to 5.
            xticks_rotation (int, optional): _description_. Defaults to 90.
            yticks_rotation (int, optional): _description_. Defaults to 0.
            title_size (int, optional): _description_. Defaults to 20.
            legend_size (int, optional): _description_. Defaults to 5.
            xlabel (_type_, optional): _description_. Defaults to None.
            xlabel_size (int, optional): _description_. Defaults to 15.
            ylabel (_type_, optional): _description_. Defaults to None.
            ylabel_size (int, optional): _description_. Defaults to 15.
            zlabel (_type_, optional): _description_. Defaults to None.
            zlabel_size (int, optional): _description_. Defaults to 15.
            color_labels (_type_, optional): _description_. Defaults to None.
            color_hue (_type_, optional): _description_. Defaults to None.
            colorbar_label (_type_, optional): _description_. Defaults to None.
            colorbar_range (list, optional): _description_. Defaults to [0, 1].
            colorbar_shrink (int, optional): _description_. Defaults to 1.
            markers_list (_type_, optional): _description_. Defaults to None.
            marker_size (int, optional): _description_. Defaults to 30.
            color (str, optional): _description_. Defaults to 'C0'.
            cmap (str, optional): _description_. Defaults to 'cividis'.
            ot_object_tick (bool, optional): _description_. Defaults to False.
            ot_category_tick (bool, optional): _description_. Defaults to False.
            draw_category_line (bool, optional): _description_. Defaults to False.
            category_line_color (str, optional): _description_. Defaults to 'C2'.
            category_line_alpha (float, optional): _description_. Defaults to 0.2.
            category_line_style (str, optional): _description_. Defaults to 'dashed'.
            plot_eps_log (bool, optional): _description_. Defaults to False.
            lim_eps (_type_, optional): _description_. Defaults to None.
            lim_gwd (_type_, optional): _description_. Defaults to None.
            lim_acc (_type_, optional): _description_. Defaults to None.
        """

        self.visualization_params = {
            'show_figure':show_figure,
            'figsize': figsize,
            'cbar_ticks_size': cbar_ticks_size,
            'cbar_format':cbar_format,
            'ticks_size': ticks_size,
            'xticks_rotation': xticks_rotation,
            'yticks_rotation': yticks_rotation,
            'title_size': title_size,
            'legend_size': legend_size,
            'xlabel': xlabel,
            'xlabel_size': xlabel_size,
            'ylabel': ylabel,
            'ylabel_size': ylabel_size,
            'zlabel': zlabel,
            'zlabel_size': zlabel_size,
            'color_labels': color_labels,
            'color_hue': color_hue,
            'colorbar_label': colorbar_label,
            'colorbar_range': colorbar_range,
            'colorbar_shrink': colorbar_shrink,
            'markers_list': markers_list,
            'marker_size': marker_size,
            'color':color,
            'cmap':cmap,
            'ot_object_tick': ot_object_tick,
            'ot_category_tick': ot_category_tick,
            'draw_category_line': draw_category_line,
            'category_line_color': category_line_color,
            'category_line_alpha': category_line_alpha,
            'category_line_style': category_line_style,
            'plot_eps_log':plot_eps_log,
            'lim_eps':lim_eps,
            'lim_ged':lim_gwd,
            'lim_acc':lim_acc,
        }

    def __call__(self):
        return self.visualization_params

    def set_params(self, **kwargs):
        for key, item in kwargs.items():
            self.visualization_params[key] = item

class Representation:
    def __init__(
        self,
        name,
        metric="cosine",
        sim_mat: np.ndarray = None,
        embedding: np.ndarray = None,
        get_embedding=False,
        MDS_dim=3,
        object_labels=None,
        category_name_list=None,
        num_category_list=None,
        category_idx_list=None,
        func_for_sort_sim_mat=None,
    ) -> None:
        """
        A class object that has information of a representation, such as embeddings and similarity matrices
        Args:
            name (_type_): The name of the representation. 
            metric (str, optional): The metric for computing distances between embeddings. Defaults to "cosine".
            sim_mat (np.ndarray, optional): Representational dissimilarity matrix. Defaults to None.
            embedding (np.ndarray, optional): The array of N-dimensional embeddings of all stimuli. Defaults to None.
            get_embedding (bool, optional): If True, the embeddings are automatically computed from the sim_mat using MDS method. Defaults to True.
            MDS_dim (int, optional): The dimension of the embeddings computed automatically from the sim_mat using MDS. Defaults to 3.
            object_labels (_type_, optional): The labels for each stimulus or points. Defaults to None.
            category_name_list (_type_, optional): If there is coarse category labels, you can select categories to be used in the unsupervised alignment. Defaults to None.
            num_category_list (_type_, optional): The list of numbers of stimuli each coarse category contains. Defaults to None.
            category_idx_list (_type_, optional): The list of indices that represents which coarse category each stimulus belongs to. Defaults to None.
            func_for_sort_sim_mat (_type_, optional): A function to rearrange the matrix so that stimuli belonging to the same coarse category are arranged adjacent to each other. Defaults to None.
        """

        self.name = name
        self.metric = metric

        # parameters for label information (e.g. pictures of dog, cat,...) in the dataset for the representation matrix.
        self.object_labels = object_labels
        self.category_name_list = category_name_list
        self.category_idx_list = category_idx_list
        self.num_category_list = num_category_list

        # define the function to sort the representation matrix by the label parameters above (Default is None). Users can define it by themselves.
        self.func_for_sort_sim_mat = func_for_sort_sim_mat

        # compute the dissimilarity matrix from embedding if sim_mat is None, or estimate embedding from the dissimilarity matrix using MDS if embedding is None.
        assert (sim_mat is not None) or (embedding is not None), "sim_mat and embedding are None."

        if sim_mat is None:
            assert isinstance(embedding, np.ndarray), "'embedding' needs to be numpy.ndarray. "
            self.embedding = embedding
            self.sim_mat = self._get_sim_mat()
        else:
            assert isinstance(sim_mat, np.ndarray), "'sim_mat' needs to be numpy.ndarray. "
            self.sim_mat = sim_mat

        if embedding is None:
            assert isinstance(sim_mat, np.ndarray), "'sim_mat' needs to be numpy.ndarray. "
            self.sim_mat = sim_mat
            if get_embedding:
                self.embedding = self._get_embedding(dim=MDS_dim)
        else:
            assert isinstance(embedding, np.ndarray), "'embedding' needs to be numpy.ndarray. "
            self.embedding = embedding

        if self.category_idx_list is not None:
            self.sorted_sim_mat = self.func_for_sort_sim_mat(self.sim_mat, category_idx_list=self.category_idx_list)
        else:
            self.sorted_sim_mat = None

    def _get_sim_mat(self):
        if self.metric == "dot":
            metric = "cosine"
        else:
            metric = self.metric

        return distance.cdist(self.embedding, self.embedding, metric=metric)

    def _get_embedding(self, dim):
        MDS_embedding = manifold.MDS(n_components=dim, dissimilarity="precomputed", random_state=0)
        embedding = MDS_embedding.fit_transform(self.sim_mat)
        return embedding

    def show_sim_mat(
        self,
        sim_mat_format:str="default",
        visualization_config: VisualizationConfig = VisualizationConfig(),
        fig_dir:Optional[str]=None,
        ticks:Optional[str]=None,
    ):
        """
        Show the dissimilarity matrix of the representation.

        Args:
            sim_mat_format (str, optional): "default", "sorted", or "both". If "sorted" is selected, the rearranged matrix is shown. Defaults to "default".
            visualization_config (VisualizationConfig, optional): container of parameters used for figure. Defaults to VisualizationConfig().
            fig_dir (_type_, optional): The directory for saving the figure. Defaults to None.
            ticks (_type_, optional): "numbers", "objects", or "category". Defaults to None.

        Raises:
            ValueError: _description_
        """

        if fig_dir is not None:
            fig_path = os.path.join(fig_dir, f"RDM_{self.name}")
        else:
            fig_path = None

        if sim_mat_format == "default" or sim_mat_format == "both":
            if sim_mat_format == "default":
                assert self.category_name_list is None, "please set the 'sim_mat_format = sorted'. "
            
            visualize_functions.show_heatmap(
                self.sim_mat,
                title=self.name,
                save_file_name=fig_path,
                ticks=ticks,
                category_name_list=None,
                num_category_list=None,
                object_labels=self.object_labels,
                **visualization_config(),
            )

        elif sim_mat_format == "sorted" or sim_mat_format == "both":
            assert self.category_idx_list is not None, "No label info to sort the 'sim_mat'."
            visualize_functions.show_heatmap(
                self.sorted_sim_mat,
                title=self.name + "_sorted",
                save_file_name=fig_path,
                ticks=ticks,
                category_name_list=self.category_name_list,
                num_category_list=self.num_category_list,
                object_labels=self.object_labels,
                **visualization_config(),
            )

        else:
            raise ValueError("sim_mat_format must be either 'default', 'sorted', or 'both'.")

    def show_sim_mat_distribution(self, **kwargs):
        """
        Show the distribution of the values of elements of the dissimilarity matrix.
        """
        # figsize = kwargs.get('figsize', (4, 3))
        xticks_rotation = kwargs.get("xticks_rotation", 90)
        yticks_rotation = kwargs.get("yticks_rotation", 0)
        title_size = kwargs.get("title_size", 60)
        xlabel_size = kwargs.get("xlabel_size", 40)
        ylabel_size = kwargs.get("ylabel_size", 40)
        color = kwargs.get("color", "C0")

        lower_triangular = np.tril(self.sim_mat)
        lower_triangular = lower_triangular.flatten()

        plt.figure()
        plt.hist(lower_triangular, bins=100, color=color)
        plt.title(f"Distribution of RDM ({self.name})", fontsize=title_size)
        plt.xlabel("RDM value")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

    def show_embedding(
        self,
        dim=3,
        visualization_config: VisualizationConfig = VisualizationConfig(),
        category_name_list=None,
        num_category_list=None,
        category_idx_list=None,
        title=None,
        legend=True,
        fig_dir=None,
        fig_name="Aligned_embedding.png",
    ):
        """
        Show the embeddings.

        Args:
            dim (int, optional): The dimension of the embedding space for visualization. If the original dimensions of the embeddings are higher than "dim", the dimension reduction method(PCA) is applied. Defaults to 3.
            visualization_config (VisualizationConfig, optional): . Defaults to VisualizationConfig().
            category_name_list (_type_, optional): Select the coarse category labels to be visualized in the embedding space. Defaults to None.
            num_category_list (_type_, optional):  Defaults to None.
            category_idx_list (_type_, optional):  Defaults to None.
            title (_type_, optional): The title of the figure. Defaults to None.
            legend (bool, optional): If True, the legend is shown. Defaults to True.
            fig_dir (_type_, optional): The directory for saving the figure. Defaults to None.
            fig_name (str, optional): The name of the figure. Defaults to "Aligned_embedding.png".
        """

        if fig_dir is not None:
            fig_path = os.path.join(fig_dir, fig_name)
        else:
            fig_path = None

        if category_idx_list is None:
            if self.category_idx_list is not None:
                category_name_list = self.category_name_list
                num_category_list = self.num_category_list
                category_idx_list = self.category_idx_list

        visualize_embedding = visualize_functions.VisualizeEmbedding(
            embedding_list=[self.embedding],
            dim=dim,
            category_name_list=category_name_list,
            num_category_list=num_category_list,
            category_idx_list=category_idx_list,
        )

        visualize_embedding.plot_embedding(
            name_list=[self.name], title=title, legend=legend, save_dir=fig_path, **visualization_config()
        )

class PairwiseAnalysis:
    def __init__(
        self,
        results_dir:str,
        config: OptimizationConfig,
        source: Representation,
        target: Representation,
        pair_name:str=None,
        data_name:str="no_defined",
        filename:str=None,
    ) -> None:
        """
        A class object that has methods conducting gw-alignment and corresponding results
        This object has information of a pair of Representations.

        Args:
            results_dir (str): the path to save the result data. 
            config (OptimizationConfig): all the parameters to compute the GWOT.
            source (Representation): instance of source represenation
            target (Representation): instance of target represenation
            
            pair_name (str, optional): Defaults to None. 
                                       If None, the name of this instance will be made by the names of two representations 
            
            data_name (str): the folder name directly under the `result_dir`. 
                             Defaults is "no_defined". If None, the results cannot be saved.
            
            filename (str, optional): the folder name directly under the `result_dir/data_name`, 
                                      and the name of database file or database table.   
                                      Defaults to None. If None, this will be automatically defined 
                                      from `data_name` and the names of two representations.
        """

        self.source = source
        self.target = target
        self.config = config

        if pair_name is None:
            self.pair_name = f"{source.name}_vs_{target.name}"
        else:
            self.pair_name = pair_name

        assert data_name is not None, "please define the `data_name` to save the result."
        self.data_name = data_name

        if filename is None:
            self.filename = self.data_name + "_" + self.pair_name
        else:
            self.filename = filename

        self.results_dir = results_dir
        self.save_path = os.path.join(results_dir, self.data_name, self.filename, self.config.init_mat_plan)
        self.figure_path = os.path.join(self.save_path, 'figure')
        self.data_path = os.path.join(self.save_path, 'data')

        assert np.array_equal(
            self.source.num_category_list, self.target.num_category_list
        ), "the label information doesn't seem to be the same."

        assert np.array_equal(
            self.source.object_labels, self.target.object_labels
        ), "the label information doesn't seem to be the same."

        # Generate the URL for the database. Syntax differs for SQLite and others.
        if self.config.db_params["drivername"] == "sqlite":
            self.storage = "sqlite:///" + self.save_path + "/" + self.filename + "_" + self.config.init_mat_plan + ".db"
        else:
            self.storage = URL.create(
                database=self.filename + "_" + self.config.init_mat_plan,
                **self.config.db_params).render_as_string(hide_password=False)

    def _change_types_to_numpy(self, *var):
        ret = []     
        for a in var:
            if isinstance(a, torch.Tensor):
                a = a.to('cpu').numpy()

            ret.append(a)
        
        return ret
    
    def show_both_sim_mats(self):
        """
        visualize the two sim_mat.
        """
    
        a = self.source.sim_mat
        b = self.target.sim_mat
        
        a, b = self._change_types_to_numpy(a, b)
    
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

    def RSA(self, metric="spearman", method="normal"):
        """
        Conventional representation similarity analysis (RSA).

        Args:
            metric (str, optional): spearman or pearson. Defaults to "spearman".
            method (str, optional): compute RSA from all ("all") or upper tri ("normal") 
                                    of the element in the matrices. Defaults to "normal".

        Returns:
            corr : RSA value
        """
        a = self.source.sim_mat
        b = self.target.sim_mat
        
        a, b = self._change_types_to_numpy(a, b)
        
        if method == "normal":
            upper_tri_a = a[np.triu_indices(a.shape[0], k=1)]
            upper_tri_b = b[np.triu_indices(b.shape[0], k=1)]

            if metric == "spearman":
                corr, _ = spearmanr(upper_tri_a, upper_tri_b)
            elif metric == "pearson":
                corr, _ = pearsonr(upper_tri_a, upper_tri_b)

        elif method == "all":
            if metric == "spearman":
                corr, _ = spearmanr(a.flatten(), b.flatten())
            elif metric == "pearson":
                corr, _ = pearsonr(a.flatten(), b.flatten())

        return corr

    def match_sim_mat_distribution(self, return_data=False, method="target"):
        """
        Performs simple histogram matching between two matrices.
        
        Args:
            return_data (bool, optional): If True, this method will return the matched result. 
                                          If False, self.target.sim_mat will be re-written. Defaults to False.
                                          
            method (str) : change the sim_mat of which representations, source or target. Defaults to "target".

        Returns:
            new target: matched results of sim_mat.
        """
        a = self.source.sim_mat
        b = self.target.sim_mat
        
        a, b = self._change_types_to_numpy(a, b)
        
        matching = SimpleHistogramMatching(a, b)

        new_target = matching.simple_histogram_matching(method=method)

        if return_data:
            return new_target
        else:
            self.target.sim_mat = new_target

    def run_entropic_gwot(
        self,
        compute_OT:bool=False,
        delete_results:bool=False,
        OT_format:str="default",
        return_data:bool=False,
        return_figure:bool=True,
        visualization_config: VisualizationConfig = VisualizationConfig(),
        show_log:bool=False,
        fig_dir:str=None,
        ticks:str=None,
        save_dataframe:bool=False,
        target_device:Optional[str]=None,
        sampler_seed=42,
    ):
        """
        compute entropic GWOT.

        Args:
            compute_OT (bool, optional): 
                If True, the GWOT will be computed. 
                If False, the saved results will be loaded. Defaults to False.
            
            delete_results (bool, optional): 
                If True, all the saved results will be deleted. Defaults to False.
            
            OT_format (str, optional):
                format of sim_mat to visualize. 
                Options are "default", "sorted", and "both". Defaults to "default".
            
            return_data (bool, optional): 
                return the computed OT. Defaults to False.
            
            return_figure (bool, optional): 
                make the result figures or not. Defaults to True.
            
            visualization_config (VisualizationConfig, optional): 
                container of parameters used for figure. Defaults to VisualizationConfig().
            
            show_log (bool, optional): 
                If True, the evaluation figure of GWOT will be made. Defaults to False.
            
            fig_dir (str, optional): 
                you can define the path to which you save the figures (.png).  
                If None, the figures will be saved in the same subfolder in "results_dir".   
                Defaults to None.
            
            ticks (str, optional): 
                you can use "objects" or "category (if existed)" or "None". Defaults to None.
            
            save_dataframe (bool, optional): 
                If True, you can save all the computed data stored in SQlite or PyMySQL in csv 
                format (pandas.DataFrame) in the result folder.  Defaults to False.
            
            target_device (str, optional): the device to compute GWOT. Defaults to None.

        Returns:
            OT: GWOT
        """

        # Delete the previous results if the flag is True.
        if delete_results:
            self.delete_prev_results()

        self.OT, df_trial = self._entropic_gw_alignment(
            compute_OT,
            target_device=target_device,
            sampler_seed=sampler_seed,
        )

        if fig_dir is None:
            fig_dir = self.figure_path

            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir, exist_ok=True)

        OT = self.show_OT(
            ot_to_plot = None,
            title=f"$\Gamma$ ({self.pair_name.replace('_', ' ')})",
            return_data=return_data,
            return_figure=return_figure,
            OT_format=OT_format,
            visualization_config=visualization_config,
            fig_dir=fig_dir,
            ticks=ticks,
        )

        if show_log:
            self.get_optimization_log(
                fig_dir=fig_dir,
                **visualization_config(),
            )

        if save_dataframe:
            df_trial.to_csv(self.save_path + '/' + self.filename + '.csv')

        return OT

    def delete_prev_results(self):
        """
        Delete the previous computed results of GWOT.
        """
        # drop database
        if database_exists(self.storage):
            drop_database(self.storage)
        # delete directory
        if os.path.exists(self.save_path):
            for root, dirs, files in os.walk(self.save_path, topdown=False):
                for name in files:
                    file_path = os.path.join(root, name)
                    os.remove(file_path)
                for name in dirs:
                    dir_path = os.path.join(root, name)
                    os.rmdir(dir_path)
            shutil.rmtree(self.save_path)

    def _entropic_gw_alignment(self, compute_OT, target_device=None, sampler_seed=42):
        if not os.path.exists(self.save_path):
            if compute_OT == False:
                warnings.simplefilter("always")
                warnings.warn(
                    "compute_OT is False, but this computing is running for the first time in the 'results_dir'.",
                    UserWarning
                )
                warnings.simplefilter("ignore")

            compute_OT = True

        study = self._run_optimization(
            compute_OT = compute_OT,
            target_device = target_device,
            sampler_seed = sampler_seed,
        )

        best_trial = study.best_trial
        df_trial = study.trials_dataframe()

        ot_path = glob.glob(self.data_path + f"/gw_{best_trial.number}.*")[0]

        if '.npy' in ot_path:
            OT = np.load(ot_path)

        elif '.pt' in ot_path:
            OT = torch.load(ot_path).to("cpu").numpy()

        return OT, df_trial

    def _run_optimization(
        self,
        compute_OT=False,
        target_device = None,
        sampler_seed = 42,
        n_jobs_for_pairwise_analysis=1,
    ):
        # generate instance optimize gw_alignment
        opt = gw_optimizer.load_optimizer(
            save_path=self.save_path,
            filename=self.filename,
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
            # generate instance solves gw_alignment
            gw = GW_Alignment(
                self.source.sim_mat,
                self.target.sim_mat,
                self.data_path,
                max_iter=self.config.max_iter,
                n_iter=self.config.n_iter,
                to_types=self.config.to_types,
                data_type=self.config.data_type,
                sinkhorn_method=self.config.sinkhorn_method,
            )

            # setting for optimization
            if self.config.init_mat_plan == "user_define":
                gw.main_compute.init_mat_builder.set_user_define_init_mat_list(self.config.user_define_init_mat_list)

            if self.config.sampler_name == "grid":
                # used only in grid search sampler below the two lines
                eps_space = opt.define_eps_space(self.config.eps_list, self.config.eps_log, self.config.num_trial)
                search_space = {"eps": eps_space}
            else:
                search_space = None

            if target_device == None:
                target_device = self.config.device

            # 2. run optimzation
            study = opt.run_study(
                gw,
                target_device,
                seed=sampler_seed,
                init_mat_plan=self.config.init_mat_plan,
                eps_list=self.config.eps_list,
                eps_log=self.config.eps_log,
                search_space=search_space,
            )

        else:
            study = opt.load_study()

        return study

    def get_optimization_log(self, fig_dir=None, **kwargs):
        """
        show both the relationships between epsilons and GWD, and between accuracy and GWD

        Args:
            fig_dir (_type_, optional): 
                you can define the path to which you save the figures (.png). 
                If None, the figures will be saved in the same subfolder in "results_dir". Defaults to None.
            
        """
        figsize = kwargs.get('figsize', (8,6))
        marker_size = kwargs.get('marker_size', 20)
        show_figure = kwargs.get('show_figure', False)
        plot_eps_log = kwargs.get('plot_eps_log', False)
        cmap = kwargs.get("cmap", 'viridis')
        ticks_size = kwargs.get("ticks_size", 5)

        lim_eps = kwargs.get("lim_eps", None)
        lim_gwd = kwargs.get("lim_gwd", None)
        lim_acc = kwargs.get("lim_acc", None)

        study = self._run_optimization(compute_OT = False)
        df_trial = study.trials_dataframe()

        # figure plotting epsilon as x-axis and GWD as y-axis
        plt.figure(figsize=figsize)
        plt.scatter(df_trial["params_eps"], df_trial["value"], c = 100 * df_trial["user_attrs_best_acc"], s = marker_size, cmap=cmap)
        plt.xlabel("$\epsilon$")
        plt.xticks(fontsize=ticks_size)
        plt.ylabel("GWD")
        plt.yticks(fontsize=ticks_size)

        if lim_eps is not None:
            plt.xlim(lim_eps)

        if lim_gwd is not None:
            plt.ylim(lim_gwd)

        if plot_eps_log:
            plt.xscale('log')

        plt.title(f"$\epsilon$ - GWD ({self.pair_name.replace('_', ' ')})")
        plt.grid(True, which = 'both')

        plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
        plt.tick_params(axis = 'x', rotation = 30,  which="both")
        plt.colorbar(label='accuracy (%)')

        plt.tight_layout()

        if fig_dir is None:
            fig_dir = self.figure_path

        plt.savefig(os.path.join(fig_dir, f"Optim_log_eps_GWD_{self.pair_name}.png"))

        if show_figure:
            plt.show()

        plt.clf()
        plt.close()

        plt.figure(figsize=figsize)
        plt.scatter(100 * df_trial["user_attrs_best_acc"], df_trial["value"].values, c = df_trial["params_eps"], cmap=cmap)
        plt.title(self.pair_name.replace('_', ' '))
        plt.xlabel("accuracy (%)")
        plt.xticks(fontsize=ticks_size)
        plt.ylabel("GWD")
        plt.yticks(fontsize=ticks_size)
        plt.colorbar(label='eps', format = "%.2e")
        plt.grid(True)

        if lim_acc is not None:
            plt.xlim(lim_acc)

        if lim_gwd is not None:
            plt.ylim(lim_gwd)

        plt.tight_layout()

        plt.savefig(os.path.join(fig_dir, f"acc_gwd_eps({self.pair_name}).png"))

        if show_figure:
            plt.show()

        plt.clf()
        plt.close()

    def show_OT(
        self,
        ot_to_plot:Optional[np.ndarray]=None,
        title:Optional[str]=None,
        OT_format="default",
        return_data=False,
        return_figure=True,
        visualization_config: VisualizationConfig = VisualizationConfig(),
        fig_dir=None,
        ticks=None,
    ):
        """
        
        visualize the OT

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
                
            fig_dir (_type_, optional): 
                you can define the path to which you save the figures (.png). 
                If None, the figures will be saved in the same subfolder in "results_dir". Defaults to None.
                
            ticks (str, optional): 
                you can use "objects" or "category (if existed)" or "None". Defaults to None.

        Returns:
            OT : the result of GWOT or sorted OT. This depends on OT_format.
        """
        if ot_to_plot is None:
            ot_to_plot = self.OT

        if OT_format == "sorted" or OT_format == "both":
            assert self.source.sorted_sim_mat is not None, "No label info to sort the 'sim_mat'."
            OT_sorted = self.source.func_for_sort_sim_mat(ot_to_plot, category_idx_list=self.source.category_idx_list)

        if return_figure:
            save_file = self.data_name + "_" + self.pair_name
            if fig_dir is not None:
                fig_path = os.path.join(fig_dir, f"{save_file}.png")
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

    def eval_accuracy(
        self,
        top_k_list,
        ot_to_evaluate = None,
        eval_type="ot_plan",
        metric="cosine",
        barycenter=False,
        supervised=False,
        category_mat=None
    ):
        """
        Evaluation of the accuracy of the unsupervised alignment
        
        There are two ways to evaluate the accuracy.  
        1. Calculate the accuracy based on the OT plan.  
        For using this method, please set the parameter `eval_type = "ot_plan"` in "calc_accuracy()".   
        
        2. Calculate the matching rate based on the k-nearest neighbors of the embeddings.   
        For using this method, please set the parameter `eval_type = "k_nearest"` in "calc_accuracy()".   

        For both cases, the accuracy evaluation criterion can be adjusted by setting `top_k_list`.  

        Args:
            top_k_list (_type_): _description_
            
            ot_to_evaluate (_type_, optional): 
                the OT to evaluate. Defaults to None.
                If None, the optimzed GWOT will be used.
            
            eval_type (str, optional): 
                two ways to evaluate the accuracy as above. Defaults to "ot_plan".
            
            metric (str, optional): 
                Please set the metric that can be used in "scipy.spatical.distance.cdist()". Defaults to "cosine".
            
            barycenter (bool, optional): _description_. Defaults to False.
            
            supervised (bool, optional): 
                define the accuracy based on a diagnoal matrix. Defaults to False.
            
            category_mat (_type_, optional): 
                This will be used for the category info. Defaults to None.

        Returns:
            df : dataframe which has accuracies for top_k. 
        """
        df = pd.DataFrame()
        df["top_n"] = top_k_list

        if supervised:
            OT = np.diag([1 / len(self.target.sim_mat)] * len(self.target.sim_mat))
        else:
            OT = self.OT

        if ot_to_evaluate is not None:
            OT = ot_to_evaluate
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

                acc = self._calc_accuracy_with_topk_diagonal(dist_mat, k=k, order="minimum")

            elif eval_type == "ot_plan":
                acc = self._calc_accuracy_with_topk_diagonal(OT, k=k, order="maximum")

            elif eval_type == "category":
                assert category_mat is not None
                acc = self._calc_accuracy_with_topk_diagonal(OT, k=k, order="maximum", category_mat=category_mat)

            acc_list.append(acc)

        df[self.pair_name] = acc_list

        return df

    def _calc_accuracy_with_topk_diagonal(self, matrix, k, order="maximum", category_mat=None):
        # Get the diagonal elements
        if category_mat is None:
            diagonal = np.diag(matrix)
        else:
            category_mat = category_mat.values

            diagonal = []
            for i in range(matrix.shape[0]):
                category = category_mat[i]

                matching_rows = np.where(np.all(category_mat == category, axis=1))[0]
                matching_elements = matrix[i, matching_rows] # get the columns of which category are the same as i-th row

                diagonal.append(np.max(matching_elements))

        # Get the top k values for each row
        if order == "maximum":
            topk_values = np.partition(matrix, -k)[:, -k:]
        elif order == "minimum":
            topk_values = np.partition(matrix, k - 1)[:, :k]
        else:
            raise ValueError("Invalid order parameter. Must be 'maximum' or 'minimum'.")

        # Count the number of rows where the diagonal is in the top k values
        count = np.sum(np.isin(diagonal, topk_values))

        # Calculate the accuracy as the proportion of counts to the total number of rows
        accuracy = count / matrix.shape[0]
        accuracy *= 100

        return accuracy

    def procrustes(self, embedding_target, embedding_source, OT):
        """
        Function that brings embedding_source closest to embedding_target by orthogonal matrix

        Args:
            embedding_target : shape (n_target, m)
            embedding_source : shape (n_source, m)
            OT : shape (n_source, n_target)
                Transportation matrix of sourse→target

        Returns:
            new_embedding_source : shape (n_source, m)
        """
        # assert self.source.shuffle == False, "you cannot use procrustes method if 'shuffle' is True."

        U, S, Vt = np.linalg.svd(np.matmul(embedding_source.T, np.matmul(OT, embedding_target)))
        Q = np.matmul(U, Vt)
        new_embedding_source = np.matmul(embedding_source, Q)
        return new_embedding_source

    def wasserstein_alignment(self, metric):
        a = ot.unif(len(self.source.embedding))
        b = ot.unif(len(self.target.embedding))

        M = distance.cdist(self.source.embedding, self.target.embedding, metric=metric)

        self.OT, log = ot.emd(a, b, M, log=True)

        return log["cost"]

    def get_new_source_embedding(self):
        return self.procrustes(self.target.embedding, self.source.embedding, self.OT)

    def _simulated(
        self,
        trials,
        top_k=None,
        device=None,
        to_types=None,
        data_type=None,
    ):
        """
        By using optimal transportation plan obtained with entropic GW
        as an initial transportation matrix, we run the optimization of GWOT without entropy.

        This procedure further minimizes GWD and enables us to fairly compare GWD values
        obtained with different entropy regularization values.
        """

        GWD0_list = list()
        OT0_list = list()

        trials = trials[trials['value'] != np.nan]
        top_k_trials = trials.sort_values(by="value", ascending=True)

        if top_k is not None:
            top_k_trials = top_k_trials.head(top_k)

        top_k_trials = top_k_trials[['number', 'value', 'params_eps']]
        top_k_trials = top_k_trials.reset_index(drop=True)

        drop_index_list = []
        for i in tqdm(top_k_trials['number']):
            try:
                ot_path = glob.glob(self.data_path + f"/gw_{i}.*")[0]
            except:
                ind = top_k_trials[top_k_trials['number'] == i].index
                drop_index_list.extend(ind.tolist())
                print(f"gw_{i}.npy (or gw_{i}.pt) doesn't exist in the result folder...")
                continue

            if '.npy' in ot_path:
                OT = np.load(ot_path)
            elif '.pt' in ot_path:
                OT = torch.load(ot_path).to("cpu").numpy()

            log0 = self.run_gwot_without_entropic(
                ot_mat=OT,
                max_iter=10000,
                device=device,
                to_types=to_types,
                data_type=data_type,
            )

            gwd = log0['gw_dist']
            new_ot = log0['ot0']

            if isinstance(new_ot, torch.Tensor):
                gwd = gwd.to('cpu').item()
                new_ot = new_ot.to('cpu').numpy()

            GWD0_list.append(gwd)
            OT0_list.append(new_ot)

        top_k_trials = top_k_trials.drop(top_k_trials.index[drop_index_list])

        return GWD0_list, OT0_list, top_k_trials

    def run_gwot_without_entropic(
        self,
        ot_mat = None,
        max_iter = 10000,
        device:str=None,
        to_types:str=None,
        data_type:str=None,
    ):
        """
        comnpute GWOT without entropy
        
        Args:
            ot_mat (_type_, optional): 
                initial OT Plan. Defaults to None.
                If None, uniform matrix will be used.
            
            max_iter (int, optional): 
                Maximum number of iterations for the Sinkhorn algorithm. 
                Defaults to 1000.
            
            device (str, optional): 
                the device to compute. Defaults to None.
            
            to_types (str, optional):
                Specifies the type of data structure to be used,
                either "torch" or "numpy". Defaults to None.
            
            data_type (str, optional):  
                Specifies the type of data to be used in computation. 
                Defaults to None.

        Returns:
            log0 : results of GWOT without entropy
        """

        if ot_mat is not None:
            p = ot_mat.sum(axis=1)
            q = ot_mat.sum(axis=0)
        else:
            p = ot.unif(len(self.source.sim_mat))
            q = ot.unif(len(self.target.sim_mat))

        sim_mat1 = self.source.sim_mat
        sim_mat2 = self.target.sim_mat

        if device is None and to_types is None and data_type is None:
            back_end = backend.Backend(self.config.device, self.config.to_types, self.config.data_type)
        else:
            back_end = backend.Backend(device, to_types, data_type)

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

        log0["ot0"] = new_ot_mat

        return log0

    def run_test_after_entropic_gwot(
        self,
        top_k:int=None,
        OT_format = "default",
        eval_type = "ot_plan",
        device=None,
        to_types=None,
        data_type=None,
        ticks=None,
        category_mat = None,
        visualization_config: VisualizationConfig = VisualizationConfig(),
    ):
        """
        run GWOT without entropy by setting the optimized entropic GWOT as the initial plan.
 
        Args:
            top_k (int, optional): 
                this will be used for loading the optimized OT from the bottom k of lowest GWD (= top_k) value.
                Defaults to None. If None, all the computed OT will be used for GWOT without entropy.
                
            OT_format (str, optional):
                format of sim_mat to visualize. 
                Options are "default", "sorted", and "both". Defaults to "default".
            
            eval_type (str, optional): 
                two ways to evaluate the accuracy. Defaults to "ot_plan".
            
            device (str, optional): 
                the device to compute. Defaults to None.
            
            to_types (str, optional):
                Specifies the type of data structure to be used,
                either "torch" or "numpy". Defaults to None.
            
            data_type (str, optional):  
                Specifies the type of data to be used in computation. 
                Defaults to None.
            
            ticks (str, optional): 
                you can use "objects" or "category (if existed)" or "None". Defaults to None.
            
            category_mat (_type_, optional): 
                This will be used for the category info. Defaults to None.
            
            visualization_config (VisualizationConfig, optional): 
                container of parameters used for figure. Defaults to VisualizationConfig().
            
        """

        self.OT, df_trial = self._entropic_gw_alignment(compute_OT=False)

        GWD0_list, OT0_list, top_k_trials = self._simulated(
            df_trial,
            top_k=top_k,
            device=device,
            to_types=to_types,
            data_type=data_type,
        )

        ot_no_ent = OT0_list[np.argmin(GWD0_list)]

        # plot results
        self.show_OT(
            ot_to_plot = ot_no_ent,
            title=f"$\Gamma$ (GWOT Without Entropy) ({self.pair_name.replace('_', ' ')})",
            return_data=False,
            return_figure=True,
            OT_format=OT_format,
            visualization_config=visualization_config,
            fig_dir=None,
            ticks=ticks,
        )

        self._evaluate_accuracy_and_plot(ot_no_ent, eval_type)

        if category_mat is not None:
            self._evaluate_accuracy_and_plot(ot_no_ent, "category", category_mat=category_mat)

        self._plot_GWD_optimization(top_k_trials, GWD0_list, **visualization_config())

    def _evaluate_accuracy_and_plot(self, ot_to_evaluate, eval_type, category_mat=None):
        top_k_list = [1, 5, 10]
        df_before = self.eval_accuracy(
            top_k_list = top_k_list,
            ot_to_evaluate = None,
            eval_type=eval_type,
            category_mat=category_mat,
        )

        df_after = self.eval_accuracy(
            top_k_list = top_k_list,
            ot_to_evaluate=ot_to_evaluate,
            eval_type=eval_type,
            category_mat=category_mat,
        )

        df_before = df_before.set_index('top_n')
        df_after  = df_after.set_index('top_n')

        plot_df = pd.concat([df_before, df_after], axis=1)
        plot_df.columns = ['before', 'after']
        plot_df.plot(
            kind='bar',
            title=f'{self.pair_name.replace("_", " ")}, {eval_type} accuracy',
            legend=True,
            grid=True,
            rot=0,
        )

        plt.savefig(os.path.join(self.figure_path, "accuracy_comparison_with_or_without.png"))
        plt.show()
        plt.clf()
        plt.close()

        print(plot_df)

    def _plot_GWD_optimization(self, top_k_trials, GWD0_list, marker_size = 10, **kwargs):
        figsize = kwargs.get('figsize', (8, 6))
        title_size = kwargs.get('title_size', 15)
        plot_eps_log = kwargs.get('plot_eps_log', False)
        show_figure = kwargs.get('show_figure', True)

        plt.rcParams.update(plt.rcParamsDefault)
        plt.style.use("seaborn-darkgrid")

        plt.figure(figsize = figsize)
        plt.title("$\epsilon$ - GWD (" + self.pair_name.replace("_", " ") + ")", fontsize = title_size)

        plt.scatter(top_k_trials["params_eps"], top_k_trials["value"], c = 'red', s=marker_size, label="before") # before
        plt.scatter(top_k_trials["params_eps"], GWD0_list, c = 'blue', s=marker_size, label = "after") # after

        if plot_eps_log:
            plt.xscale('log')

        plt.xlabel("$\epsilon$")
        plt.ylabel("GWD")
        plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
        plt.tick_params(axis = 'x', rotation = 30,  which="both")
        plt.grid(True, which = 'both')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_path, "eps_vs_gwd_comparison_with_or_without.png"))

        if show_figure:
            plt.show()

        plt.clf()
        plt.close()

class AlignRepresentations:
    def __init__(
        self,
        config: OptimizationConfig,
        representations_list:List[Representation],
        pairs_computed:List[str]=None,
        specific_eps_list:dict=None,
        histogram_matching=False,
        metric="cosine",
        main_results_dir:str = None,
        data_name:str = "NoDefined",
    ) -> None:
        """
        This object has methods for conducting N groups level analysis and corresponding results.
        This has information of all pairs of representations.

        Args:
            config (OptimizationConfig): 
                all the essential parameters for GWOT.
            
            representations_list (List[Representation]): 
                List of Representation.
            
            pairs_computed (List[str], optional):
                You can change the specific pairs to be computed by setting `pair_computed` . Defaults to None.
            
            specific_eps_list (dict, optional): 
                You can also change the epsilon range `specific_eps_list`. Defaults to None.
            
            histogram_matching (bool, optional): 
                This will adjust the histogram of target to that of source. Defaults to False.
            
            metric (str, optional): 
                Please set the metric that can be used in "scipy.spatical.distance.cdist()". Defaults to "cosine".
            
            main_results_dir (str, optional): 
                The main folder directory to save the results. Defaults to None.
                
            data_name (str, optional): 
                The name of the folder to save the result for each pair. Defaults to "NoDefined".
        """

        self.config = config
        self.data_name = data_name
        self.metric = metric
        self.representations_list = representations_list
        self.histogram_matching = histogram_matching

        self.main_results_dir = main_results_dir
        self.main_pair_name = None
        self.main_file_name = None

        self.RSA_corr = dict()

        self.name_list = [rep.name for rep in self.representations_list]

        self.all_pair_list = list(itertools.combinations(range(len(self.representations_list)), 2))

        print(f"data_name : {self.data_name}")

        self.set_specific_eps_list(specific_eps_list)

        self.set_pair_computed(pairs_computed)

    def set_pair_computed(self, pairs_computed:Optional[List]):
        """
        User can only re-run the optimization for specific pairs by using `set_pair_computed`.

        (examples)
        >>> pairs_computed = ["Group1", "Group2_vs_Group4"] 
        >>> align_representation.set_pair_computed(pairs_computed)
        """
        self.pairs_computed = pairs_computed

        if pairs_computed is not None:
            print("The pairs to compute was selected by pairs_computed...")
            self.specific_pair_list = self._specific_pair_list(pairs_computed)
            self.pairwise_list = self._get_pairwise_list(self.specific_pair_list)

        else:
            print("All the pairs in the list below will be computed. ")
            self.pairwise_list = self._get_pairwise_list(self.all_pair_list)

    def set_specific_eps_list(
        self,
        specific_eps_list:Optional[dict],
        specific_only:bool = False,
    ):
        """
        Also, user can re-define the epsilon range for some pairs by using `set_specific_eps_list`. 
        
        The rest of them will be computed with `config.eps_list`. 
        
        If `specific_only` is True (default is False), only these pairs will be computed and the rest of them were skipped.

        (examples) 
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
                self.pairwise_list = self._get_pairwise_list(self.all_pair_list)

    def _specific_pair_list(self, pair_list):
        if isinstance(pair_list, dict):
            key_loop = pair_list.keys()
        elif isinstance(pair_list, list):
            key_loop = pair_list

        specific_pair_list = []
        for key in key_loop:
            if not key in self.name_list:
                source_name, target_name = key.split('_vs_')

                source_idx = self.name_list.index(source_name)
                target_idx = self.name_list.index(target_name)

                rep_list = [(source_idx, target_idx)]

            else:
                rep_idx = self.name_list.index(key)
                rep_list = [nn for nn in self.all_pair_list if rep_idx in nn]

            specific_pair_list.extend(rep_list)

        return specific_pair_list

    def _get_pairwise_list(self, pair_list):
        pairwise_list = []

        for pair in pair_list:
            config_copy = copy.deepcopy(self.config)

            s = self.representations_list[pair[0]]
            t = self.representations_list[pair[1]]

            if self.specific_eps_list is not None:
                pair_name = f"{s.name}_vs_{t.name}"
                if s.name in self.specific_eps_list.keys():
                    config_copy.eps_list = self.specific_eps_list[s.name]
                elif pair_name in self.specific_eps_list.keys():
                    config_copy.eps_list = self.specific_eps_list[pair_name]

            pairwise = PairwiseAnalysis(
                results_dir=self.main_results_dir,
                config=config_copy,
                source=s,
                target=t,
                data_name=self.data_name,
                pair_name=self.main_pair_name,
                filename=self.main_file_name,
            )

            print('pair:', pairwise.pair_name, 'eps_list:', config_copy.eps_list)

            if self.histogram_matching:
                pairwise.match_sim_mat_distribution()

            pairwise_list.append(pairwise)

        return pairwise_list

    def RSA_get_corr(self, metric="spearman", method="normal"):
        """
        Conventional representation similarity analysis (RSA).

        Args:
            metric (str, optional): spearman or pearson. Defaults to "spearman".
            method (str, optional): compute RSA from all ("all") or upper tri ("normal") 
                                    of the element in the matrices. Defaults to "normal".
        """
        for pairwise in self.pairwise_list:
            corr = pairwise.RSA(metric=metric, method=method)
            self.RSA_corr[pairwise.pair_name] = corr
            print(f"Correlation {pairwise.pair_name.replace('_', ' ')} : {corr}")

    def show_sim_mat(
        self,
        sim_mat_format="default",
        visualization_config: VisualizationConfig = VisualizationConfig(),
        visualization_config_hist: VisualizationConfig = VisualizationConfig(),
        fig_dir=None,
        show_distribution=True,
        ticks=None,
    ):
        """
        Show the dissimilarity matrix of the representation.

        Args:
            sim_mat_format (str, optional): "default", "sorted", or "both". If "sorted" is selected, the rearranged matrix is shown. Defaults to "default".
            visualization_config (VisualizationConfig, optional): container of parameters used for figure. Defaults to VisualizationConfig().
            visualization_config_hist (VisualizationConfig, optional): container of parameters used for histogram figure. Defaults to VisualizationConfig().
            fig_dir (_type_, optional): The directory for saving the figure. Defaults to None.
            show_distribution : show the histogram figures. Defaults to True.
            ticks (_type_, optional): "numbers", "objects", or "category". Defaults to None.

        Raises:
            ValueError: _description_
        """
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

    def _single_computation(
        self,
        compute_OT=False,
        delete_results=False,
        return_data=False,
        return_figure=True,
        OT_format="default",
        visualization_config: VisualizationConfig = VisualizationConfig(),
        show_log=False,
        fig_dir=None,
        ticks=None,
        save_dataframe=False,
        target_device=None,
        change_sampler_seed=False,
        sampler_seed=42,
    ):

        OT_list = []
        for pairwise in self.pairwise_list:
            if change_sampler_seed:
                sampler_seed += 1

            OT = pairwise.run_entropic_gwot(
                compute_OT=compute_OT,
                delete_results=delete_results,
                return_data=return_data,
                return_figure=return_figure,
                OT_format=OT_format,
                visualization_config=visualization_config,
                show_log=show_log,
                fig_dir=fig_dir,
                ticks=ticks,
                save_dataframe=save_dataframe,
                target_device=target_device,
                sampler_seed=sampler_seed,
            )

            OT_list.append(OT)

        return OT_list

    def gw_alignment(
        self,
        compute_OT=False,
        delete_results=False,
        return_data=False,
        return_figure=True,
        OT_format="default",
        visualization_config: VisualizationConfig = VisualizationConfig(),
        show_log=False,
        fig_dir=None,
        ticks=None,
        save_dataframe=False,
        change_sampler_seed=False,
        fix_sampler_seed=42,
        parallel_method="multithread",
    ):
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
            
            fig_dir (str, optional): 
                you can define the path to which you save the figures (.png).  
                If None, the figures will be saved in the same subfolder in "results_dir".   
                Defaults to None.
            
            ticks (str, optional): 
                you can use "objects" or "category (if existed)" or "None". Defaults to None.
            
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
            
        """

        if isinstance(fix_sampler_seed, int) and fix_sampler_seed > -1:
            first_sampler_seed = fix_sampler_seed
        else:
            raise ValueError("please 'sampler_seed' = True or False or int > 0.")

        if self.config.n_jobs > 1:
            OT_list = []
            processes = []

            if parallel_method == "multiprocess":
                pool = ProcessPoolExecutor(self.config.n_jobs)
            elif parallel_method == "multithread":
                pool = ThreadPoolExecutor(self.config.n_jobs)
            else:
                raise ValueError('parallel_method = "multiprocess" or "multithread".')

            with pool:
                for pair_number in range(len(self.pairwise_list)):

                    if self.config.multi_gpu:
                        target_device = "cuda:" + str(pair_number % torch.cuda.device_count())
                    else:
                        target_device = self.config.device

                    if isinstance(self.config.multi_gpu, list):
                        gpu_idx = pair_number % len(self.config.multi_gpu)
                        target_device = "cuda:" + str(self.config.multi_gpu[gpu_idx])

                    pairwise = self.pairwise_list[pair_number]

                    if self.config.to_types == "numpy":
                        if self.config.multi_gpu != False:
                            warnings.warn("numpy doesn't use GPU. Please 'multi_GPU = False'.", UserWarning)
                        target_device = self.config.device

                    if change_sampler_seed:
                        sampler_seed = first_sampler_seed + pair_number
                    else:
                        sampler_seed = first_sampler_seed

                    future = pool.submit(
                        pairwise.run_entropic_gwot,
                        compute_OT=compute_OT,
                        delete_results=delete_results,
                        return_data=False,
                        return_figure=False,
                        OT_format="default",
                        visualization_config=visualization_config,
                        show_log=False,
                        fig_dir=None,
                        ticks=None,
                        save_dataframe=save_dataframe,
                        target_device=target_device,
                        sampler_seed=sampler_seed,
                    )

                    processes.append(future)

                for future in as_completed(processes):
                    future.result()

            if return_figure or return_data:
                OT_list = self._single_computation(
                    compute_OT=False,
                    delete_results=False,
                    return_data=return_data,
                    return_figure=return_figure,
                    OT_format=OT_format,
                    visualization_config=visualization_config,
                    show_log=show_log,
                    fig_dir=fig_dir,
                    ticks=ticks,
                    save_dataframe=save_dataframe,
                )

        if self.config.n_jobs == 1:
            OT_list = self._single_computation(
                compute_OT=compute_OT,
                delete_results=delete_results,
                return_data=return_data,
                return_figure=return_figure,
                OT_format=OT_format,
                visualization_config=visualization_config,
                show_log=show_log,
                fig_dir=fig_dir,
                ticks=ticks,
                save_dataframe=save_dataframe,
                change_sampler_seed=change_sampler_seed,
                sampler_seed=first_sampler_seed,
            )

        if self.config.n_jobs < 1:
            raise ValueError("n_jobs > 0 is required in this toolbox.")


        if return_data:
            return OT_list

    def gwot_after_entropic(
        self,
        top_k=None,
        parallel_method=None,
        OT_format="default",
        ticks=None,
        category_mat=None,
        visualization_config: VisualizationConfig = VisualizationConfig(),
    ):
        """
        run GWOT without entropy by setting the optimized entropic GWOT as the initial plan.
 
        Args:
            top_k (int, optional): 
                this will be used for loading the optimized OT from the bottom k of lowest GWD (= top_k) value.
                Defaults to None. If None, all the computed OT will be used for GWOT without entropy.
            
            parallel_method (str, optional): 
                parallel method to compute GWOT. Defaults to "multithread".
                
            OT_format (str, optional):
                format of sim_mat to visualize. 
                Options are "default", "sorted", and "both". Defaults to "default".

            ticks (str, optional): 
                you can use "objects" or "category (if existed)" or "None". Defaults to None.
            
            category_mat (_type_, optional): 
                This will be used for the category info. Defaults to None.
            
            visualization_config (VisualizationConfig, optional): 
                container of parameters used for figure. Defaults to VisualizationConfig().
            
        """
        if parallel_method == "multiprocess":
            pool = ProcessPoolExecutor(self.config.n_jobs)

        elif parallel_method == "multithread":
            pool = ThreadPoolExecutor(self.config.n_jobs)

        elif parallel_method is None:
            for idx, pair in enumerate(self.pairwise_list):
                pair.run_test_after_entropic_gwot(
                    top_k=top_k,
                    OT_format=OT_format,
                    device=None,
                    to_types=None,
                    data_type=None,
                    ticks=ticks,
                    category_mat=category_mat,
                    visualization_config = visualization_config,
                )

            return None

        with pool:
            if self.config.to_types == "numpy":
                if self.config.multi_gpu != False:
                    warnings.warn("numpy doesn't use GPU. Please 'multi_GPU = False'.", UserWarning)
                target_device = self.config.device

            processes = []
            for idx, pair in enumerate(self.pairwise_list):
                if self.config.multi_gpu:
                    target_device = "cuda:" + str(idx % torch.cuda.device_count())
                else:
                    target_device = self.config.device

                if isinstance(self.config.multi_gpu, list):
                    gpu_idx = idx % len(self.config.multi_gpu)
                    target_device = "cuda:" + str(self.config.multi_gpu[gpu_idx])

                future = pool.submit(
                    pair.run_test_after_entropic_gwot,
                    top_k=top_k,
                    OT_format=OT_format,
                    device=target_device,
                    to_types=self.config.to_types,
                    data_type=self.config.data_type,
                    ticks=ticks,
                    category_mat=category_mat,
                    visualization_config = visualization_config,
                )
                processes.append(future)

            for future in as_completed(processes):
                future.result()




    def drop_gw_alignment_files(self, drop_filenames: Optional[List[str]] = None, drop_all: bool = False):
        """Delete the specified database and directory with the given filename

        Args:
            drop_filenames (Optional[List[str]], optional): [description]. Defaults to None.
            drop_all (bool, optional): [description]. Defaults to False.
        """
        if drop_all:
            drop_filenames = [pairwise.filename for pairwise in self.pairwise_list]

        if drop_filenames is None:
            raise ValueError("Specify the results name in drop_filenames or set drop_all=True")

        for pairwise in self.pairwise_list:
            if (pairwise.filename not in drop_filenames) or (not database_exists(pairwise.storage)):
                continue
            pairwise.delete_prev_results()

    def show_optimization_log(
        self,
        fig_dir=None,
        visualization_config=VisualizationConfig(),
    ):
        """
        show both the relationships between epsilons and GWD, and between accuracy and GWD

        Args:
            fig_dir (_type_, optional): 
                you can define the path to which you save the figures (.png). 
                If None, the figures will be saved in the same subfolder in "results_dir". Defaults to None.
            
        """
        # default setting
        plt.rcParams.update(plt.rcParamsDefault)
        plt.style.use("seaborn-darkgrid")

        for pairwise in self.pairwise_list:
            pairwise.get_optimization_log(
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
        pivot,
        n_iter,
        compute_OT=False,
        delete_results=False,
        return_data=False,
        return_figure=True,
        OT_format="default",
        visualization_config: VisualizationConfig = VisualizationConfig(),
        show_log=False,
        fig_dir=None,
        ticks=None,
    ):
        """
        The unuspervised alignment method using Wasserstein barycenter proposed by Lian et al. (2021).
        

        Args:
            pivot (_type_): The representation to which the other representations are aligned initially using gw alignment
            n_iter (_type_): The number of iterations to calculate the location of the barycenter.
            compute_OT (bool, optional): _description_. Defaults to False.
            delete_results (bool, optional): _description_. Defaults to False.
            return_data (bool, optional): _description_. Defaults to False.
            return_figure (bool, optional): _description_. Defaults to True.
            OT_format (str, optional): _description_. Defaults to "default".
            visualization_config (VisualizationConfig, optional): _description_. Defaults to VisualizationConfig().
            show_log (bool, optional): _description_. Defaults to False.
            fig_dir (_type_, optional): _description_. Defaults to None.
            ticks (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        #assert self.all_pair_list == range(len(self.pairwise_list))

        # Select the pivot
        pivot_representation = self.representations_list[pivot]
        others_representaions = self.representations_list[:pivot] + self.representations_list[pivot + 1 :]

        # GW alignment to the pivot
        for representation in others_representaions:
            pairwise = PairwiseAnalysis(
                results_dir=self.main_results_dir,
                config=self.config,
                source=representation,
                target=pivot_representation,
                data_name=self.data_name,
                pair_name=self.main_pair_name,
                filename=self.main_file_name,
            )

            pairwise.run_entropic_gwot(
                compute_OT=compute_OT,
                delete_results=delete_results,
                return_data=return_data,
                return_figure=return_figure,
                OT_format=OT_format,
                visualization_config=visualization_config,
                show_log=show_log,
                fig_dir=fig_dir,
                ticks=ticks,
            )

            pairwise.source.embedding = pairwise.get_new_source_embedding()

        # Set up barycenter
        init_embedding = self.calc_barycenter()
        self.barycenter = Representation(
            name="barycenter",
            embedding=init_embedding,
            object_labels=self.representations_list[0].object_labels,
            category_name_list=self.representations_list[0].category_name_list,
            num_category_list=self.representations_list[0].num_category_list,
            category_idx_list=self.representations_list[0].category_idx_list,
            func_for_sort_sim_mat=self.representations_list[0].func_for_sort_sim_mat
            
        )

        # Set pairwises whose target are the barycenter
        pairwise_barycenters = []
        for representation in self.representations_list:
            pairwise = PairwiseAnalysis(
                results_dir=self.main_results_dir,
                config=self.config,
                source=representation,
                target=self.barycenter,
                data_name=self.data_name,
                pair_name=self.main_pair_name,
                filename=self.main_file_name,
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
                loss += pairwise.wasserstein_alignment(metric=self.metric)

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
        OT_list = []
        for pairwise in self.pairwise_list:
            OT = pairwise.show_OT(
                title=f"$\Gamma$ ({pairwise.pair_name})",
                return_data=return_data,
                return_figure=return_figure,
                OT_format=OT_format,
                visualization_config=visualization_config,
                fig_dir=fig_dir,
                ticks=ticks,
            )
            OT_list.append(OT)

        if return_data:
            return OT_list

    def _get_OT_all_pair(self, pairwise_barycenters):
        # replace OT of each pairwise by the product of OTs to the barycenter
        pairs = list(itertools.combinations(pairwise_barycenters, 2))

        for i, (pairwise_1, pairwise_2) in enumerate(pairs):
            OT = np.matmul(pairwise_2.OT, pairwise_1.OT.T)
            OT *= len(OT)  # normalize
            self.pairwise_list[i].OT = OT

    def calc_accuracy(
        self,
        top_k_list,
        eval_type="ot_plan",
        category_mat=None,
        barycenter=False,
        return_dataframe:bool=False,
    ):
        """
        Evaluation of the accuracy of the unsupervised alignment

        Args:
            top_k_list (list): define the top k accuracy in list 
            
            eval_type (str, optional): 
                two ways to evaluate the accuracy as above. Defaults to "ot_plan".
       
            category_mat (_type_, optional): 
                This will be used for the category info. Defaults to None.
            
            barycenter (bool, optional): 
                _description_. Defaults to False.
            
            return_dataframe (bool, optional): 
                If True, the accuracy result will be returned in pandas.DataFrame format. Defaults to False.
            
        """
        accuracy = pd.DataFrame()
        accuracy["top_n"] = top_k_list

        for pairwise in self.pairwise_list:
            df = pairwise.eval_accuracy(top_k_list, eval_type=eval_type, metric=self.metric, barycenter=barycenter, category_mat=category_mat)

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

        print("Mean : \n", accuracy.iloc[:, 1:].mean(axis="columns"))
    
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
        eval_type="ot_plan",
        fig_dir=None,
        fig_name="Accuracy_ot_plan.png",
        scatter=True,
    ):
        """
        plot the accuracy of the unsupervised alignment for each top_k
        """
        # default setting
        plt.rcParams.update(plt.rcParamsDefault)
        plt.style.use("seaborn-darkgrid")
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
        # plt.legend(loc = "best")
        plt.tick_params(axis="both", which="major")
        plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2)
        if fig_dir is not None:
            plt.savefig(os.path.join(fig_dir, fig_name))
        plt.show()

        plt.clf()
        plt.close()

    def _procrustes_to_pivot(self, pivot):
        the_others, pivot_idx_list = self._check_pairs(pivot)

        # check whether 'pair_list' includes all pairs between the pivot and the other Representations
        assert len(the_others) == len(self.representations_list)-1, "'pair_list' must include all pairs between the pivot and the other Representations."

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

    def _check_pairs(self, pivot):
        the_others = set()
        pivot_idx_list = [] # [pair_idx, paivot_idx]
        for i, pair in enumerate(self.all_pair_list):
            if pivot in pair:
                the_others.add(filter(lambda x: x != pivot, pair))

                pivot_idx = pair.index(pivot)
                pivot_idx_list.append([i, pivot_idx])

        return the_others, pivot_idx_list

    def visualize_embedding(
        self,
        dim,
        pivot=0,
        returned="figure",
        visualization_config: VisualizationConfig = VisualizationConfig(),
        category_name_list=None,
        num_category_list=None,
        category_idx_list=None,
        title=None,
        legend=True,
        fig_dir=None,
        fig_name="Aligned_embedding.png",
    ):
        """_summary_

        Args:
            dim (_type_): The number of dimensions the points are embedded.
            pivot (str, optional) : The pivot or "barycenter" to which all embeddings are aligned. Defaults to 0.
            returned (str, optional): "figure" or "row_data. Defaults to "figure".
            visualization_config (VisualizationConfig, optional): _description_. Defaults to VisualizationConfig().
            category_name_list (_type_, optional): _description_. Defaults to None.
            num_category_list (_type_, optional): _description_. Defaults to None.
            category_idx_list (_type_, optional): _description_. Defaults to None.
            title (_type_, optional): _description_. Defaults to None.
            legend (bool, optional): _description_. Defaults to True.
            fig_dir (_type_, optional): _description_. Defaults to None.
            fig_name (str, optional): _description_. Defaults to "Aligned_embedding.png".

        Returns:
            _type_: _description_
        """

        if fig_dir is not None:
            fig_path = os.path.join(fig_dir, fig_name)
        else:
            fig_path = None

        if pivot != "barycenter":
            self._procrustes_to_pivot(pivot)
            #for i in range(len(self.pairwise_list) // 2):
            #    pair = self.pairwise_list[i]
            #    pair.source.embedding = pair.get_new_source_embedding()

        else:
            assert self.barycenter is not None

        name_list = []
        embedding_list = []
        for i in range(len(self.representations_list)):
            embedding_list.append(self.representations_list[i].embedding)
            name_list.append(self.representations_list[i].name)

        if returned == "figure":
            if category_idx_list is None:
                if self.representations_list[0].category_idx_list is not None:
                    category_name_list = self.representations_list[0].category_name_list
                    num_category_list = self.representations_list[0].num_category_list
                    category_idx_list = self.representations_list[0].category_idx_list

            visualize_embedding = visualize_functions.VisualizeEmbedding(
                embedding_list=embedding_list,
                dim=dim,
                category_name_list=category_name_list,
                num_category_list=num_category_list,
                category_idx_list=category_idx_list,
            )

            visualize_embedding.plot_embedding(
                name_list=name_list, title=title, legend=legend, save_dir=fig_path, **visualization_config()
            )

        elif returned == "row_data":
            return embedding_list
