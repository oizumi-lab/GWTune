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

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from .gw_alignment import GW_Alignment
from .histogram_matching import SimpleHistogramMatching
from .utils import gw_optimizer, visualize_functions

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
        """_summary_

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

        self.visualization_params = {
            'show_figure':show_figure,
            'figsize': figsize,
            'cbar_ticks_size': cbar_ticks_size,
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
    """
    A class object that has information of a representation, such as embeddings and similarity matrices
    """

    def __init__(
        self,
        name,
        metric="cosine",
        sim_mat: np.ndarray = None,
        embedding: np.ndarray = None,
        get_embedding=True,
        MDS_dim=3,
        object_labels=None,
        category_name_list=None,
        num_category_list=None,
        category_idx_list=None,
        func_for_sort_sim_mat=None,
    ) -> None:

        # meta data for the representation matrix (dis-similarity matrix).
        self.name = name
        self.metric = metric

        # parameters for label information (e.g. pictures of dog, cat,...) in the dataset for the representation matrix.
        self.object_labels = object_labels
        self.category_name_list = category_name_list
        self.category_idx_list = category_idx_list
        self.num_category_list = num_category_list

        # define the function to sort the representation matrix by the label parameters above (Default is None). Users can define it by themselves.
        self.func_for_sort_sim_mat = func_for_sort_sim_mat

        # computing the representation matrix (or embedding) from embedding (or representation matrix) if sim_mat is None.
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
        sim_mat_format="default",
        visualization_config: VisualizationConfig = VisualizationConfig(),
        fig_dir=None,
        ticks=None,
    ):
        """_summary_

        Args:
            sim_mat_format (str, optional): _description_. Defaults to "default".
            visualization_config (VisualizationConfig, optional): _description_. Defaults to VisualizationConfig().
            fig_dir (_type_, optional): _description_. Defaults to None.
            ticks (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """

        if fig_dir is not None:
            fig_path = os.path.join(fig_dir, f"RDM_{self.name}")
        else:
            fig_path = None

        if sim_mat_format == "default" or sim_mat_format == "both":
            visualize_functions.show_heatmap(
                self.sim_mat,
                title=self.name,
                save_file_name=fig_path,
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
    """
    A class object that has methods conducting gw-alignment and corresponding results
    This object has information of a pair of Representations.
    """

    def __init__(
        self, 
        results_dir:str,
        config: OptimizationConfig, 
        source: Representation, 
        target: Representation,
        pair_name=None,
        data_name=None,
        filename=None,
    ) -> None:
        """_summary_

        Args:
            results_dir (str): _description_
            config (OptimizationConfig): _description_
            source (Representation): _description_
            target (Representation): _description_
            pair_name (_type_, optional): _description_. Defaults to None.
            data_name (_type_, optional): _description_. Defaults to None.
            filename (_type_, optional): _description_. Defaults to None.
        """
        
        self.source = source
        self.target = target
        self.config = config

        if pair_name is None:
            self.pair_name = f"{source.name}_vs_{target.name}"
        else:
            self.pair_name = pair_name
        
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

    def show_both_sim_mats(self):

        a = self.source.sim_mat
        b = self.target.sim_mat

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
        if method == "normal":
            upper_tri_source = self.source.sim_mat[np.triu_indices(self.source.sim_mat.shape[0], k=1)]
            upper_tri_target = self.target.sim_mat[np.triu_indices(self.target.sim_mat.shape[0], k=1)]

            if metric == "spearman":
                corr, _ = spearmanr(upper_tri_source, upper_tri_target)
            elif metric == "pearson":
                corr, _ = pearsonr(upper_tri_source, upper_tri_target)

        elif method == "all":
            if metric == "spearman":
                corr, _ = spearmanr(self.source.sim_mat.flatten(), self.target.sim_mat.flatten())
            elif metric == "pearson":
                corr, _ = pearsonr(self.source.sim_mat.flatten(), self.target.sim_mat.flatten())

        return corr

    def match_sim_mat_distribution(self, return_data=False):
        """
        Args:
            return_data (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        matching = SimpleHistogramMatching(self.source.sim_mat, self.target.sim_mat)

        new_target = matching.simple_histogram_matching()

        if return_data:
            return new_target
        else:
            self.target.sim_mat = new_target

    def run_entropic_gwot(
        self,
        compute_OT=False,
        delete_results=False,
        OT_format="default",
        return_data=False,
        return_figure=True,
        visualization_config: VisualizationConfig = VisualizationConfig(),
        show_log=False,
        fig_dir=None,
        ticks=None,
        save_dataframe=False,
        target_device=None,
        sampler_seed=42,
    ):
        """_summary_

        Args:
            compute_OT (bool, optional): _description_. Defaults to False.
            delete_results (bool, optional): _description_. Defaults to False.
            OT_format (str, optional): _description_. Defaults to "default".
            return_data (bool, optional): _description_. Defaults to False.
            return_figure (bool, optional): _description_. Defaults to True.
            visualization_config (VisualizationConfig, optional): _description_. Defaults to VisualizationConfig().
            show_log (bool, optional): _description_. Defaults to False.
            fig_dir (_type_, optional): _description_. Defaults to None.
            ticks (_type_, optional): _description_. Defaults to None.
            filename (_type_, optional): _description_. Defaults to None.
            save_dataframe (bool, optional): _description_. Defaults to False.
            target_device (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        # Delete the previous results if the flag is True.
        if delete_results:
            self.delete_prev_results()
            
        self.OT, df_trial = self._gw_alignment(
            compute_OT,
            target_device=target_device,
            sampler_seed=sampler_seed,
        )

        if fig_dir is None:
            fig_dir = self.figure_path

            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir, exist_ok=True)

        OT = self._show_OT(
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

    def _gw_alignment(self, compute_OT, target_device=None, sampler_seed=42):
        """_summary_

        Args:
            compute_OT (_type_): _description_
            target_device (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

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
            
        # GWD0_list, OT0_list = self.simulated_annealing(study, k = 1)
        # new_OT = OT0_list[np.argmin(GWD0_list)]

        return OT, df_trial

    def _run_optimization(
        self,
        compute_OT=False,
        target_device = None,
        sampler_seed = 42,
        n_jobs_for_pairwise_analysis=1,
    ):
        """_summary_

        Args:
            compute_OT (_type_): _description_
            target_device (_type_, optional): _description_. Defaults to None.
            n_jobs_for_pairwise_analysis (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
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
    
    def simulated_annealing(self, study, k):
        """
        GWOT without entropy
        
        By using optimal transportation plan obtained with entropic GW 
        as an initial transportation matrix, we run the optimization of GWOT without entropy.  
        
        This procedure further minimizes GWD and enables us to fairly compare GWD values 
        obtained with different entropy regularization values.  
        """
        
        GWD0_list = list()
        OT0_list = list()
        
        trials = study.trials_dataframe()
        sorted_trials = trials.sort_values(by="value", ascending=True)
        top_k_trials = sorted_trials.head(k)
        top_k_trials = top_k_trials[['number', 'value', 'params_eps']]
        
        for i in top_k_trials['number']:
            ot_path = glob.glob(self.data_path + f"/gw_{i}.*")[0]
            if '.npy' in ot_path:
                OT = np.load(ot_path)
            elif '.pt' in ot_path:
                OT = torch.load(ot_path).to("cpu").numpy()

            C1 = self.source.sim_mat
            C2 = self.target.sim_mat
            p = OT.sum(axis=1)
            q = OT.sum(axis=0)
            OT0, log0 = ot.gromov.gromov_wasserstein(C1, C2, p, q, log=True, verbose=False, G0=OT)
            GWD0 = log0['gw_dist']
            GWD0_list.append(GWD0)
            OT0_list.append(OT0)
        
        return GWD0_list, OT0_list

    def get_optimization_log(self, fig_dir=None, **kwargs):
        figsize = kwargs.get('figsize', (8,6))
        marker_size = kwargs.get('marker_size', 20)
        show_figure = kwargs.get('show_figure', False)
        plot_eps_log = kwargs.get('plot_eps_log', False)
        cmap = kwargs.get("cmap", 'viridis')

        lim_eps = kwargs.get("lim_eps", None)
        lim_gwd = kwargs.get("lim_gwd", None)
        lim_acc = kwargs.get("lim_acc", None)
       
        study = self._run_optimization(compute_OT = False)
        df_trial = study.trials_dataframe()

        # figure plotting epsilon as x-axis and GWD as y-axis
        plt.figure(figsize=figsize)
        plt.scatter(df_trial["params_eps"], df_trial["value"], c = 100 * df_trial["user_attrs_best_acc"], s = marker_size, cmap=cmap)
        plt.xlabel("$\epsilon$")
        plt.ylabel("GWD")

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
        plt.ylabel("GWD")
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

    def _show_OT(
        self,
        title,
        OT_format="default",
        return_data=False,
        return_figure=True,
        visualization_config: VisualizationConfig = VisualizationConfig(),
        fig_dir=None,
        ticks=None,
    ):

        if OT_format == "sorted" or OT_format == "both":
            assert self.source.sorted_sim_mat is not None, "No label info to sort the 'sim_mat'."
            OT_sorted = self.source.func_for_sort_sim_mat(self.OT, category_idx_list=self.source.category_idx_list)

        if return_figure:
            save_file = self.data_name + "_" + self.pair_name
            if fig_dir is not None:
                fig_path = os.path.join(fig_dir, f"{save_file}.png")
            else:
                fig_path = None

            if OT_format == "default" or OT_format == "both":
                visualize_functions.show_heatmap(
                    self.OT,
                    title=title,
                    save_file_name=fig_path,
                    object_labels = self.source.object_labels,
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
                return self.OT

            elif OT_format == "sorted":
                return OT_sorted

            elif OT_format == "both":
                return self.OT, OT_sorted

            else:
                raise ValueError("OT_format must be either 'default', 'sorted', or 'both'.")

    def eval_accuracy(
        self,
        top_k_list,
        eval_type="ot_plan",
        metric="cosine",
        barycenter=False,
        supervised=False,
        category_mat=None
    ):
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

class AlignRepresentations:
    """
    This object has methods for conducting N groups level analysis and corresponding results.
    This has information of all pairs of representations.
    """
    def __init__(
        self,
        config: OptimizationConfig,
        representations_list:List[Representation],
        pairs_computed:List[str]=None,
        specific_eps_list:dict=None,
        histogram_matching=False,
        metric="cosine",
        main_results_dir:str = None,
        data_name:str = None,
    ) -> None:
        """_summary_

        Args:
            config (OptimizationConfig): _description_
            representations_list (List[Representation]): _description_
            pairs_computed (List[str], optional): _description_. Defaults to None.
            specific_eps_list (dict, optional): _description_. Defaults to None.
            histogram_matching (bool, optional): _description_. Defaults to False.
            metric (str, optional): _description_. Defaults to "cosine".
            main_results_dir (str, optional): _description_. Defaults to None.
            data_name (str, optional): _description_. Defaults to None.
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
        """_summary_

        Args:
            sim_mat_format (str, optional): _description_. Defaults to "default".
            visualization_config (VisualizationConfig, optional): _description_. Defaults to VisualizationConfig().
            fig_dir (_type_, optional): _description_. Defaults to None.
            show_distribution (bool, optional): _description_. Defaults to True.
            ticks (_type_, optional): _description_. Defaults to None.
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
        """_summary_

        Args:
            compute_OT (bool, optional): _description_. Defaults to False.
            delete_results (bool, optional): _description_. Defaults to False.
            return_data (bool, optional): _description_. Defaults to False.
            return_figure (bool, optional): _description_. Defaults to True.
            OT_format (str, optional): _description_. Defaults to "default".
            visualization_config (VisualizationConfig, optional): _description_. Defaults to VisualizationConfig().
            show_log (bool, optional): _description_. Defaults to False.
            fig_dir (_type_, optional): _description_. Defaults to None.
            ticks (_type_, optional): _description_. Defaults to None.
            save_dataframe (bool, optional): _description_. Defaults to False.
            change_sampler_seed (bool, optional): _description_. Defaults to False.
            fix_sampler_seed (int, optional): _description_. Defaults to 42.
            parallel_method (str, optional): _description_. Defaults to "multithread".

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
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

        assert self.all_pair_list == range(len(self.pairwise_list))

        # Select the pivot
        pivot_representation = self.representations_list[pivot]
        others_representaions = self.representations_list[:pivot] + self.representations_list[pivot + 1 :]

        # GW alignment to the pivot
        # ここの部分はあとでself.gw_alignmentの中に組み込む
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
            category_mat=self.representations_list[0].category_mat,
            category_name_list=self.representations_list[0].category_name_list,
        )

        # Set pairwises whose target is the barycenter
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
            OT = pairwise._show_OT(
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
        barycenter=False
    ):
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
