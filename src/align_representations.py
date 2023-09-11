# %%
import copy
import glob
import itertools
import json
import os
import shutil
import sys
import warnings
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import optuna
import ot
import pandas as pd
import torch
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr
from sklearn import manifold
from sklearn.base import TransformerMixin
from sqlalchemy import URL
from sqlalchemy_utils import database_exists, drop_database
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from .embeddings import obtain_embedding
from .gw_alignment import GW_Alignment
from .histogram_matching import SimpleHistogramMatching
from .utils import backend, gw_optimizer


# %%
class OptimizationConfig:
    """This is an instance for sharing the parameters to compute GWOT with the instance PairwiseAnalysis.

    Please check the tutoial.ipynb for detailed info.
    """

    def __init__(
        self,
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
        sampler_name: str = "tpe",
        pruner_name: str = "hyperband",
        pruner_params: Dict[str, Union[int, float]] = {
            "n_startup_trials": 1,
            "n_warmup_steps": 2,
            "min_resource": 2,
            "reduction_factor": 3,
        },
    ) -> None:
        """Initialization of the instance.

        Args:
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
            db_params (dict, optional):
                Parameters for creating the database URL.
                Defaults to {"drivername": "mysql", "username": "root", "password": "", "host": "localhost", "port": 3306}.
            init_mat_plan (str, optional):
                The method to initialize transportation plan. Defaults to "random".
            n_iter (int, optional):
                Number of initial plans evaluated in single optimization. Defaults to 1.
            max_iter (int, optional):
                Maximum number of iterations for entropic Gromov-Wasserstein alignment by POT. Defaults to 200.
            sampler_name (str, optional):
                Name of the sampler used in optimization. Options are "random", "grid", and "tpe". Defaults to "tpe".
            pruner_name (str, optional):
                Name of the pruner used in optimization. Options are "hyperband", "median", and "nop". Defaults to "hyperband".
            pruner_params (dict, optional):
                Additional parameters for the pruner. See Optuna's pruner page for more details.
                Defaults to {"n_startup_trials": 1, "n_warmup_steps": 2, "min_resource": 2, "reduction_factor": 3}.
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

        self.storage = storage
        self.db_params = db_params
        assert (
            storage is not None or db_params is not None
        ), "storage or db_params must be provided."

        self.init_mat_plan = init_mat_plan
        self.n_iter = n_iter
        self.max_iter = max_iter

        self.sampler_name = sampler_name
        self.user_define_init_mat_list = user_define_init_mat_list

        self.pruner_name = pruner_name
        self.pruner_params = pruner_params


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
        metric (str):
            The metric for computing distances between embeddings.
        object_labels (List[str]):
            The labels for each stimulus or points.
        category_name_list (List[str], optional):
            List of category names.
        category_idx_list (List[int], optional):
            List of category indices.
        num_category_list (List[int], optional):
            List of numbers of stimuli each coarse category contains.
        func_for_sort_sim_mat (Callable, optional):
            A function to rearrange the matrix so that stimuli belonging to the same coarse category are arranged adjacent to each other.
        sim_mat (np.ndarray):
            Representational dissimilarity matrix.
        embedding (np.ndarray):
            The array of N-dimensional embeddings of all stimuli.
        sorted_sim_mat (np.ndarray, optional):
            Similarity matrix rearranged according to category labels, if applicable.
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
    ) -> None:
        """Initialize the Representation class.

        Args:
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
        """

        self.name = name
        self.metric = metric

        # parameters for label information (e.g. pictures of dog, cat,...) in the dataset for the representation matrix.
        self.object_labels = object_labels
        self.category_name_list = category_name_list
        self.category_idx_list = category_idx_list
        self.num_category_list = num_category_list

        # define the function to sort the representation matrix by the label parameters above (Default is None).]
        # Users can define it by themselves.
        self.func_for_sort_sim_mat = func_for_sort_sim_mat

        # compute the dissimilarity matrix from embedding if sim_mat is None,
        # or estimate embedding from the dissimilarity matrix using MDS if embedding is None.
        assert (sim_mat is not None) or (
            embedding is not None
        ), "sim_mat and embedding are None."

        if sim_mat is None:
            assert isinstance(
                embedding, np.ndarray
            ), "'embedding' needs to be numpy.ndarray. "
            self.embedding = embedding
            self.sim_mat = self._get_sim_mat()
        else:
            assert isinstance(
                sim_mat, np.ndarray
            ), "'sim_mat' needs to be numpy.ndarray. "
            self.sim_mat = sim_mat

        if embedding is None:
            assert isinstance(
                sim_mat, np.ndarray
            ), "'sim_mat' needs to be numpy.ndarray. "
            self.sim_mat = sim_mat
            if get_embedding:
                self.embedding = self._get_embedding(dim=MDS_dim)
        else:
            assert isinstance(
                embedding, np.ndarray
            ), "'embedding' needs to be numpy.ndarray. "
            self.embedding = embedding

        if self.category_idx_list is not None:
            self.sorted_sim_mat = self.func_for_sort_sim_mat(
                self.sim_mat, category_idx_list=self.category_idx_list
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
            n_components=dim, dissimilarity="precomputed", random_state=0
        )
        embedding = MDS_embedding.fit_transform(self.sim_mat)
        return embedding

    def _get_sim_mat(self) -> np.ndarray:
        """Compute the dissimilarity matrix based on the given metric.

        Returns:
            np.ndarray: The computed dissimilarity matrix.
        """
        if self.metric == "dot":
            metric = "cosine"
        else:
            metric = self.metric

        return distance.cdist(self.embedding, self.embedding, metric=metric)


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
        OT_sorted (np.ndarray):
            GWOT sorted according to category labels, if applicable.
    """

    def __init__(
        self,
        results_dir: str,
        config: OptimizationConfig,
        source: Representation,
        target: Representation,
        data_name: str = "no_defined",
        pair_name: Optional[str] = None,
        instance_name: Optional[str] = None,
    ) -> None:
        """Initializes the PairwiseAnalysis class.

        Args:
            results_dir (str):
                Path to save the result data.
            config (OptimizationConfig):
                Parameters to compute the GWOT.
            source (Representation):
                Instance of the source representation.
            target (Representation):
                Instance of the target representation.
            data_name (str, optional):
                Name of the data to be analyzed. For example, 'color'. Defaults to "no_defined".
            pair_name (Optional[str], optional):
                Name of this instance. Derived from source and target if not provided. Defaults to None.
            instance_name (Optional[str], optional):
                Name of this instance. Derived from data_name and pair_name if not provided. Defaults to None.

        Raises:
            AssertionError: If the label information from source and target representations is not the same.
        """

        # information of the representations and optimization
        self.config = config
        self.source = source
        self.target = target

        # information of the data
        self.data_name = data_name  # name of align representations
        self.pair_name = (
            f"{source.name}_vs_{target.name}" if pair_name is None else pair_name
        )
        self.instance_name = (
            self.data_name + "_" + self.pair_name
            if instance_name is None
            else instance_name
        )
        self.study_name = self.instance_name + "_" + self.config.init_mat_plan

        # path setting
        self.results_dir = results_dir
        self.save_path = os.path.join(results_dir, self.config.init_mat_plan)
        self.figure_path = os.path.join(self.save_path, "figure")
        self.data_path = os.path.join(self.save_path, "data")

        for p in [self.results_dir, self.save_path, self.figure_path, self.data_path]:
            if not os.path.exists(p):
                os.makedirs(p)

        # check label information is the same
        assert np.array_equal(
            self.source.num_category_list, self.target.num_category_list
        ), "the label information doesn't seem to be the same."

        assert np.array_equal(
            self.source.object_labels, self.target.object_labels
        ), "the label information doesn't seem to be the same."

        # Generate the URL for the database. Syntax differs for SQLite and others.
        self.storage = self.config.storage
        if self.storage is None:
            if self.config.db_params["drivername"] == "sqlite":
                self.storage = (
                    "sqlite:///"
                    + self.save_path
                    + "/"
                    + self.instance_name
                    + "_"
                    + self.config.init_mat_plan
                    + ".db"
                )

            else:
                self.storage = URL.create(
                    database=self.instance_name + "_" + self.config.init_mat_plan,
                    **self.config.db_params,
                ).render_as_string(hide_password=False)

        # results
        self.OT = None
        self.OT_sorted = None

    # setting methods
    def _change_types_to_numpy(self, *var):
        ret = []
        for a in var:
            if isinstance(a, torch.Tensor):
                a = a.to("cpu").numpy()

            ret.append(a)

        return ret

    def match_sim_mat_distribution(
        self,
        return_data: bool = False,
        method: str = "target"
    ) -> np.ndarray:
        """Performs simple histogram matching between two matrices.

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

    # RSA methods
    def RSA(self, metric: str = "spearman") -> float:
        """Conventional representation similarity analysis (RSA).

        Args:
            metric (str, optional):
                spearman or pearson. Defaults to "spearman".

        Returns:
            corr :
                RSA value
        """
        a = self.source.sim_mat
        b = self.target.sim_mat

        a, b = self._change_types_to_numpy(a, b)

        upper_tri_a = a[np.triu_indices(a.shape[0], k=1)]
        upper_tri_b = b[np.triu_indices(b.shape[0], k=1)]

        if metric == "spearman":
            corr, _ = spearmanr(upper_tri_a, upper_tri_b)
        elif metric == "pearson":
            corr, _ = pearsonr(upper_tri_a, upper_tri_b)

        return corr

    # GWOT methods
    def delete_prev_results(
        self,
        delete_database: bool = True,
        delete_directory: bool = True
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
        if database_exists(self.storage):
            if delete_database:
                drop_database(self.storage)
            else:
                try:
                    optuna.delete_study(study_name=self.study_name, storage=self.storage)
                except KeyError:
                    print(f"study {self.study_name} doesn't exist in {self.storage}")

        # delete directosry
        if delete_directory:
            shutil.rmtree(self.save_path)
        else:
            for p in [self.data_path, self.figure_path]:
                if os.path.exists(p):
                    for root, _, files in os.walk(p, topdown=False):
                        for file in files:
                            os.remove(os.path.join(root, file))

    def run_entropic_gwot(
        self,
        compute_OT: bool = False,
        delete_results: bool = False,
        delete_database: bool = True,
        delete_directory: bool = True,
        OT_format: str = "default",
        save_dataframe: bool = False,
        target_device: Optional[str] = None,
        sampler_seed: int = 42,
    ) -> Tuple[Optional[Union[np.ndarray, torch.Tensor]], Optional[np.ndarray]]:
        """Compute entropic GWOT.

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
            target_device (str, optional):
                the device to compute GWOT. Defaults to None.

        Returns:
            OT (Optional[Union[np.ndarray, torch.Tensor]]): GWOT.
            OT_sorted (Optional[np.ndarray]): GWOT sorted according to category labels, if applicable.

        Raises:
            ValueError: If OT_format is not one of "default", "sorted", or "both".
        """

        # Delete the previous results if the flag is True.
        if delete_results:
            self.delete_prev_results(delete_database, delete_directory)

        # run gwot
        self.OT, df_trial = self._run_entropic_gwot(
            compute_OT,
            target_device=target_device,
            sampler_seed=sampler_seed,
        )

        if save_dataframe:
            df_trial.to_csv(self.save_path + "/" + self.filename + ".csv")

        # sort OT
        if OT_format == "default":
            return self.OT, None

        else:
            assert (self.source.sorted_sim_mat is not None), "No label info to sort the 'sim_mat'."
            self.OT_sorted = self.source.func_for_sort_sim_mat(
                self.OT, category_idx_list=self.source.category_idx_list
            )

            if OT_format == "sorted":
                return None, self.OT_sorted

            elif OT_format == "both":
                return self.OT, self.OT_sorted

            else:
                ValueError("OT_format must be either 'default', 'sorted', or 'both'.")

    def _run_entropic_gwot(
        self,
        compute_OT: bool,
        target_device: Optional[str] = None,
        sampler_seed: int = 42,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], pd.DataFrame]:
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

        Returns:
            OT (Union[np.ndarray, torch.Tensor]): GWOT.
            df_trial (pd.DataFrame): dataframe of the optimization log
        """

        if not os.path.exists(self.save_path):
            if compute_OT == False:
                warnings.simplefilter("always")
                warnings.warn(
                    "compute_OT is False, but this computing is running for the first time in the 'results_dir'.",
                    UserWarning,
                )
                warnings.simplefilter("ignore")

            compute_OT = True

        study = self._run_optimization(
            compute_OT=compute_OT,
            target_device=target_device,
            sampler_seed=sampler_seed,
        )

        best_trial = study.best_trial
        df_trial = study.trials_dataframe()

        ot_path = glob.glob(self.data_path + f"/gw_{best_trial.number}.*")[0]

        if ".npy" in ot_path:
            OT = np.load(ot_path)

        elif ".pt" in ot_path:
            OT = torch.load(ot_path).to("cpu").numpy()

        return OT, df_trial

    def _run_optimization(
        self,
        compute_OT: bool = False,
        target_device: Optional[str] = None,
        sampler_seed: int = 42,
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
                Number of jobs to run in parallel for pairwise analysis. Defaults to 1.

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
                gw.main_compute.init_mat_builder.set_user_define_init_mat_list(
                    self.config.user_define_init_mat_list
                )

            if self.config.sampler_name == "grid":
                # used only in grid search sampler below the two lines
                eps_space = opt.define_eps_space(
                    self.config.eps_list,
                    self.config.eps_log,
                    self.config.num_trial,
                )
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

            # 3. save study information
            study_info = {
                "storage": self.storage,
                "study_name": study.study_name,
            }
            with open(self.save_path + "/study_info.json", "w") as f:
                json.dump(study_info, f)

        else:
            study = opt.load_study()

        return study

    # evaluation methods
    def calc_accuracy(
        self,
        top_k_list: List[int],
        ot_to_evaluate: Optional[np.ndarray] = None,
        eval_type: str = "ot_plan",
        metric: str = "cosine",
        barycenter: bool = False,
        supervised: bool = False,
        category_mat: Optional[np.ndarray] = None,
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

            ot_to_evaluate (Optional[np.ndarray], optional):
                the OT to evaluate. Defaults to None.
                If None, the optimzed GWOT will be used.

            eval_type (str, optional):
                two ways to evaluate the accuracy as above. Defaults to "ot_plan".

            metric (str, optional):
                Please set the metric that can be used in "scipy.spatical.distance.cdist()". Defaults to "cosine".

            barycenter (bool, optional):
                Indicates if the accuracy should be evaluated with respect to a barycenter representation.
                Defaults to False.

            supervised (bool, optional):
                define the accuracy based on a diagnoal matrix. Defaults to False.

            category_mat (Optional[np.ndarray], optional):
                This will be used for the category info. Defaults to None.

        Returns:
            df : dataframe which has accuracies for top_k.
        """

        df = pd.DataFrame({"top_n": top_k_list})

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

                acc = self._calc_accuracy_with_topk_diagonal(
                    dist_mat,
                    k=k,
                    order="minimum"
                )

            elif eval_type == "ot_plan":
                acc = self._calc_accuracy_with_topk_diagonal(
                    OT,
                    k=k,
                    order="maximum"
                )

            elif eval_type == "category":
                assert category_mat is not None
                acc = self._calc_accuracy_with_topk_diagonal(
                    OT,
                    k=k,
                    order="maximum",
                    category_mat=category_mat
                )

            acc_list.append(acc)

        df[self.pair_name] = acc_list

        return df

    def _calc_accuracy_with_topk_diagonal(
        self,
        matrix,
        k,
        order="maximum",
        category_mat=None
    ):
        # Get the diagonal elements
        if category_mat is None:
            diagonal = np.diag(matrix)
        else:
            category_mat = category_mat.values

            diagonal = []
            for i in range(matrix.shape[0]):
                category = category_mat[i]

                matching_rows = np.where(np.all(category_mat == category, axis=1))[0]
                matching_elements = matrix[
                    i, matching_rows
                ]  # get the columns of which category are the same as i-th row

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
        U, S, Vt = np.linalg.svd(
            np.matmul(embedding_source.T, np.matmul(OT, embedding_target))
        )
        Q = np.matmul(U, Vt)
        new_embedding_source = np.matmul(embedding_source, Q)

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

    # experimental GWOT methods
    def _simulated(
        self,
        trials: pd.DataFrame,
        top_k: Optional[int] = None,
        device: Optional[str] = None,
        to_types: Optional[str] = None,
        data_type: Optional[str] = None,
    ) -> Tuple[List[float], List[np.ndarray], pd.DataFrame]:
        """By using optimal transportation plan obtained with entropic GW as an initial transportation matrix, we run the optimization of GWOT without entropy.

        This procedure further minimizes GWD and enables us to fairly compare GWD values
        obtained with different entropy regularization values.

        Args:
            trials (pd.DataFrame):
                DataFrame containing trial results.
            top_k (Optional[int], optional):
                Number of top trials to consider. If None, all trials are considered. Defaults to None.
            device (Optional[str], optional):
                Device for computations. Defaults to None.
            to_types (Optional[str], optional):
                Type conversion for data. Defaults to None.
            data_type (Optional[str], optional):
                Type of data being used. Defaults to None.

        Returns:
            GWD0_list (List[float]):
                List of GWD values obtained with entropic GW.
            OT0_list (List[np.ndarray]):
                List of optimal transportation plans obtained with entropic GW.
            trials (pd.DataFrame):
                DataFrame containing trial results.
        """

        GWD0_list = list()
        OT0_list = list()

        trials = trials[trials["value"] != np.nan]
        top_k_trials = trials.sort_values(by="value", ascending=True)

        if top_k is not None:
            top_k_trials = top_k_trials.head(top_k)

        top_k_trials = top_k_trials[["number", "value", "params_eps"]]
        top_k_trials = top_k_trials.reset_index(drop=True)

        drop_index_list = []
        for i in tqdm(top_k_trials["number"]):
            try:
                ot_path = glob.glob(self.data_path + f"/gw_{i}.*")[0]
            except:
                ind = top_k_trials[top_k_trials["number"] == i].index
                drop_index_list.extend(ind.tolist())
                print(
                    f"gw_{i}.npy (or gw_{i}.pt) doesn't exist in the result folder..."
                )
                continue

            if ".npy" in ot_path:
                OT = np.load(ot_path)
            elif ".pt" in ot_path:
                OT = torch.load(ot_path).to("cpu").numpy()

            log0 = self.run_gwot_without_entropic(
                ot_mat=OT,
                max_iter=10000,
                device=device,
                to_types=to_types,
                data_type=data_type,
            )

            gwd = log0["gw_dist"]
            new_ot = log0["ot0"]

            if isinstance(new_ot, torch.Tensor):
                gwd = gwd.to("cpu").item()
                new_ot = new_ot.to("cpu").numpy()

            GWD0_list.append(gwd)
            OT0_list.append(new_ot)

        top_k_trials = top_k_trials.drop(top_k_trials.index[drop_index_list])

        return GWD0_list, OT0_list, top_k_trials

    def run_gwot_without_entropic(
        self,
        ot_mat: Optional[np.ndarray] = None,
        max_iter: int = 10000,
        device: Optional[str] = None,
        to_types: Optional[str] = None,
        data_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Comnpute GWOT without entropy

        Args:
            ot_mat (Optional[np.ndarray], optional):
                initial OT Plan. Defaults to None.
                If None, uniform matrix will be used.

            max_iter (int, optional):
                Maximum number of iterations for the Sinkhorn algorithm.
                Defaults to 1000.

            device (Optional[str], optional):
                the device to compute. Defaults to None.

            to_types (Optional[str], optional):
                Specifies the type of data structure to be used,
                either "torch" or "numpy". Defaults to None.

            data_type (Optional[str], optional):
                Specifies the type of data to be used in computation.
                Defaults to None.

        Returns:
            log0 (Dict[str, Any]):
                Dictionary containing the results of GWOT without entropy.
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
            back_end = backend.Backend(
                self.config.device, self.config.to_types, self.config.data_type
            )
        else:
            back_end = backend.Backend(device, to_types, data_type)

        sim_mat1, sim_mat2, p, q, ot_mat = back_end(sim_mat1, sim_mat2, p, q, ot_mat)

        new_ot_mat, log0 = ot.gromov.gromov_wasserstein(
            sim_mat1,
            sim_mat2,
            p,
            q,
            loss_fun="square_loss",
            symmetric=None,
            log=True,
            armijo=False,
            G0=ot_mat,
            max_iter=max_iter,
            tol_rel=1e-9,
            tol_abs=1e-9,
            verbose=False,
        )

        log0["ot0"] = new_ot_mat

        return log0

    def run_test_after_entropic_gwot(
        self,
        top_k: Optional[int] = None,
        OT_format: str = "default",
        eval_type: str = "ot_plan",
        device: Optional[str] = None,
        to_types: Optional[str] = None,
        data_type: Optional[str] = None,
        ticks: Optional[str] = None,
        category_mat: Optional[np.ndarray] = None,
    ) -> None:
        """Run GWOT without entropy by setting the optimized entropic GWOT as the initial plan.

        Args:
            top_k (Optional[int], optional):
                this will be used for loading the optimized OT from the bottom k of lowest GWD (= top_k) value.
                Defaults to None. If None, all the computed OT will be used for GWOT without entropy.

            OT_format (str, optional):
                format of sim_mat to visualize.
                Options are "default", "sorted", and "both". Defaults to "default".

            eval_type (str, optional):
                two ways to evaluate the accuracy. Defaults to "ot_plan".

            device (Optional[str], optional):
                the device to compute. Defaults to None.

            to_types (Optional[str], optional):
                Specifies the type of data structure to be used,
                either "torch" or "numpy". Defaults to None.

            data_type (Optional[str], optional):
                Specifies the type of data to be used in computation.
                Defaults to None.

            ticks (Optional[str], optional):
                you can use "objects" or "category (if existed)" or "None". Defaults to None.

            category_mat (Optional[np.ndarray], optional):
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
            ot_to_plot=ot_no_ent,
            title=f"$\Gamma$ (GWOT Without Entropy) ({self.pair_name.replace('_', ' ')})",
            return_data=False,
            return_figure=True,
            OT_format=OT_format,
            visualization_config=visualization_config,
            fig_dir=None,
            ticks=ticks,
        )

        self._evaluate_accuracy_and_plot(ot_no_ent, eval_type, **visualization_config())

        if category_mat is not None:
            self._evaluate_accuracy_and_plot(
                ot_no_ent, "category", category_mat=category_mat
            )

        self._plot_GWD_optimization(top_k_trials, GWD0_list, **visualization_config())


class AlignRepresentations:
    """This object has methods for conducting N groups level analysis and corresponding results.

    This class has information of all pairs of representations. The class provides functionality
    to conduct group-level analysis across various representations and compute pairwise alignments between them.

    Attributes:
        config (OptimizationConfig):
            all the essential parameters for GWOT.
        representations_list (List[Representation]):
            List of Representation.
        data_name (str, optional):
            The general name of the data to be analyzed. Defaults to "NoDefined".
        metric (str, optional):
            Please set the metric for RDM that can be used in "scipy.spatical.distance.cdist()".
            Defaults to "cosine".
        histogram_matching (bool, optional):
            This will adjust the histogram of target to that of source. Defaults to False.
        main_results_dir (Optional[str], optional):
            The main folder directory to save the results. Defaults to None.
        specific_eps_list (Optional[dict], optional):
            User can define a specific range of epsilon for some pairs. Defaults to None.
        pairs_computed (Optional[List[str]], optional):
            User can define the specific pairs to be computed. Defaults to None.
        name_list (List[str]):
            List of names from the provided representations.
        all_pair_list (List[Tuple[int, int]]):
            All possible pairwise combinations of representations.
        main_pair_name (str):
            Name identifier for the main representation pair. (Set internally)
        main_file_name (str):
            Filename identifier for the main results. (Set internally)
        RSA_corr (dict):
            Dictionary to store RSA correlation values. (Set internally)
        OT_list (List[Optional[np.ndarray]]):
            List of OT matrices. (Set internally)
        OT_sorted_list (List[Optional[np.ndarray]]):
            List of sorted OT matrices. (Set internally)
        top_k_accuracy (pd.DataFrame):
            DataFrame to store top-k accuracy values. (Set internally)
        k_nearest_matching_rate (pd.DataFrame):
            DataFrame to store k-nearest matching rate values. (Set internally)
        low_embedding_list (List[Optional[np.ndarray]]):
            List of low-dimensional embeddings. (Set internally)
        embedding_transformer (Optional[Transformer]):
            Transformer object to transform embeddings. (Set internally)
    """

    def __init__(
        self,
        config: OptimizationConfig,
        representations_list: List[Representation],
        data_name: str = "NoDefined",
        metric: str = "cosine",
        histogram_matching: bool = False,
        main_results_dir: Optional[str] = None,
        specific_eps_list: Optional[dict] = None,
        pairs_computed: Optional[List[str]] = None,
    ) -> None:
        """Initialize the AlignRepresentations object.

        Args:
            config (OptimizationConfig):
                all the essential parameters for GWOT.

            representations_list (List[Representation]):
                List of Representation.

            pairs_computed (Optional[List[str]], optional):
                You can change the specific pairs to be computed by setting `pair_computed` . Defaults to None.

            specific_eps_list (Optional[dict], optional):
                You can also change the epsilon range `specific_eps_list`. Defaults to None.

            histogram_matching (bool, optional):
                This will adjust the histogram of target to that of source. Defaults to False.

            metric (str, optional):
                Please set the metric that can be used in "scipy.spatical.distance.cdist()". Defaults to "cosine".

            main_results_dir (Optional[str], optional):
                The main folder directory to save the results. Defaults to None.

            data_name (str, optional):
                The name of the folder to save the result for each pair. Defaults to "NoDefined".
        """

        self.config = config
        self.representations_list = representations_list

        self.data_name = data_name
        self.metric = metric
        self.histogram_matching = histogram_matching

        self.main_results_dir = main_results_dir

        self.name_list = [rep.name for rep in self.representations_list]

        self.all_pair_list = list(
            itertools.combinations(range(len(self.representations_list)), 2)
        )

        print(f"data_name : {self.data_name}")

        self.set_specific_eps_list(specific_eps_list)

        self.set_pair_computed(pairs_computed)

        # results
        self.RSA_corr = dict()
        self.OT_list = [None] * len(self.all_pair_list)
        self.OT_sorted_list = [None] * len(self.all_pair_list)

        # evaluation
        self.top_k_accuracy = pd.DataFrame()
        self.k_nearest_matching_rate = pd.DataFrame()

        # embedding
        self.low_embedding_list = [None] * len(self.representations_list)
        self.embedding_transformer = None

    def set_pair_computed(self, pairs_computed: Optional[List[str]]) -> None:
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

        if pairs_computed is not None:
            print("The pairs to compute was selected by pairs_computed...")
            self.specific_pair_list = self._specific_pair_list(pairs_computed)
            self.pairwise_list = self._get_pairwise_list(self.specific_pair_list)

        else:
            print("All the pairs in the list below will be computed. ")
            self.pairwise_list = self._get_pairwise_list(self.all_pair_list)

    def set_specific_eps_list(
        self,
        specific_eps_list: Optional[Dict[str, List[float]]],
        specific_only: bool = False,
    ) -> None:
        """Also, user can re-define the epsilon range for some pairs by using `set_specific_eps_list`.

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
            assert isinstance(
                self.specific_eps_list, dict
            ), "specific_eps_list needs to be dict."
            print(
                "The range of epsilon for some pairs in the list below was changed ..."
            )

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
                source_name, target_name = key.split("_vs_")

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

        for source_idx, target_idx in pair_list:
            config_copy = copy.deepcopy(self.config)

            source = self.representations_list[source_idx]
            target = self.representations_list[target_idx]

            if self.specific_eps_list is not None:
                pair_name = f"{source.name}_vs_{target.name}"

                if source.name in self.specific_eps_list.keys():
                    config_copy.eps_list = self.specific_eps_list[source.name]

                elif pair_name in self.specific_eps_list.keys():
                    config_copy.eps_list = self.specific_eps_list[pair_name]

            pairwise = self._pairwise_from_representations(
                config=config_copy,
                source=source,
                target=target,
            )

            print("pair:", pairwise.pair_name, "eps_list:", config_copy.eps_list)

            if self.histogram_matching:
                pairwise.match_sim_mat_distribution()

            pairwise_list.append(pairwise)

        return pairwise_list

    def _pairwise_from_representations(
        self,
        config: OptimizationConfig,
        source: Representation,
        target: Representation,
    ) -> PairwiseAnalysis:
        """Create a PairwiseAnalysis object from two representations.

        pair_name will be automatically generated as "source_name_vs_target_name".
        pairwise_name will be automatically generated as "data_name_pair_name".
        directory of PairwiseAnalysis will be automatically generated as "main_results_dir/pairwise_name".

        Args:
            config (OptimizationConfig): configuration for GWOT.
            source (Representation): source representation
            target (Representation): target representation

        Returns:
            pairwise (PairwiseAnalysis): PairwiseAnalysis object.
        """

        # set information for the pair
        pair_name = f"{source.name}_vs_{target.name}"
        pairwise_name = self.data_name + "_" + pair_name
        pair_results_dir = self.main_results_dir + "/" + pairwise_name

        if not os.path.exists(pair_results_dir):
            os.makedirs(pair_results_dir)

        # create PairwiseAnalysis object
        pairwise = PairwiseAnalysis(
            results_dir=pair_results_dir,
            config=config,
            source=source,
            target=target,
            data_name=self.data_name,
            pair_name=pair_name,
            instance_name=pairwise_name,
        )
        return pairwise

    # RSA methods
    def calc_RSA_corr(self, metric: str = "spearman") -> None:
        """Conventional representation similarity analysis (RSA).

        Args:
            metric (str, optional):
                spearman or pearson. Defaults to "spearman".
        """
        for pairwise in self.pairwise_list:
            corr = pairwise.RSA(metric=metric)
            self.RSA_corr[pairwise.pair_name] = corr
            print(f"Correlation {pairwise.pair_name.replace('_', ' ')} : {corr}")

    # GWOT methods
    def gw_alignment(
        self,
        compute_OT: bool = False,
        delete_results: bool = False,
        return_data: bool = False,
        OT_format: str = "default",
        save_dataframe: bool = False,
        change_sampler_seed: bool = False,
        fix_sampler_seed: int = 42,
        parallel_method: str = "multithread",
    ) -> Optional[List[np.ndarray]]:
        """compute GWOT for each pair.

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

            save_dataframe (bool, optional):
                If True, you can save all the computed data stored in SQlite or PyMySQL in csv
                format (pandas.DataFrame) in the result folder.  Defaults to False.

            ticks (Optional[str], optional):
                you can use "objects" or "category (if existed)" or "None". Defaults to None.

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
                        target_device = "cuda:" + str(
                            pair_number % torch.cuda.device_count()
                        )
                    else:
                        target_device = self.config.device

                    if isinstance(self.config.multi_gpu, list):
                        gpu_idx = pair_number % len(self.config.multi_gpu)
                        target_device = "cuda:" + str(self.config.multi_gpu[gpu_idx])

                    pairwise = self.pairwise_list[pair_number]

                    if self.config.to_types == "numpy":
                        if self.config.multi_gpu != False:
                            warnings.warn(
                                "numpy doesn't use GPU. Please 'multi_GPU = False'.",
                                UserWarning,
                            )
                        target_device = self.config.device

                    if change_sampler_seed:
                        sampler_seed = first_sampler_seed + pair_number
                    else:
                        sampler_seed = first_sampler_seed

                    future = pool.submit(
                        pairwise.run_entropic_gwot,
                        compute_OT=compute_OT,
                        delete_results=delete_results,
                        OT_format="default",
                        save_dataframe=save_dataframe,
                        target_device=target_device,
                        sampler_seed=sampler_seed,
                    )

                    processes.append(future)

                for future in as_completed(processes):
                    future.result()

            OT_list, OT_sorted_list = self._single_computation(
                compute_OT=False,
                delete_results=False,
                OT_format=OT_format,
                save_dataframe=save_dataframe,
            )

        if self.config.n_jobs == 1:
            OT_list, OT_sorted_list = self._single_computation(
                compute_OT=compute_OT,
                delete_results=delete_results,
                OT_format=OT_format,
                save_dataframe=save_dataframe,
                change_sampler_seed=change_sampler_seed,
                sampler_seed=first_sampler_seed,
            )

        if self.config.n_jobs < 1:
            raise ValueError("n_jobs > 0 is required in this toolbox.")

        self.OT_list = OT_list
        self.OT_sorted_list = OT_sorted_list

        if return_data:
            if OT_format == "default":
                return self.OT_list
            elif OT_format == "sorted":
                return self.OT_sorted_list
            elif OT_format == "both":
                return self.OT_list, self.OT_sorted_list
            else:
                raise ValueError(
                    "OT_format must be either 'default', 'sorted', or 'both'."
                )

    def _single_computation(
        self,
        compute_OT: bool = False,
        delete_results: bool = False,
        OT_format: str = "default",
        save_dataframe: bool = False,
        target_device: str = "cpu",
        change_sampler_seed: bool = False,
        sampler_seed: int = 42,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Run GWOT for each pair.

        Args:
            compute_OT (bool, optional): Whether to compute the OT. Defaults to False.
            delete_results (bool, optional): Whether to delete the results. Defaults to False.
            OT_format (str, optional): Format of sim_mat to visualize. Defaults to "default".
            save_dataframe (bool, optional): Whether to save the computed data. Defaults to False.
            target_device (str, optional): "cuda:0", "cuda:1", ..., "cpu". Defaults to None.
            change_sampler_seed (bool, optional): Whether to change the sampler seed. Defaults to False.
            sampler_seed (int, optional): The sampler seed. Defaults to 42.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: List of computed OT matrices and sorted OT matrices.
        """

        OT_list, OT_sorted_list = [], []

        for pairwise in self.pairwise_list:

            if change_sampler_seed:
                sampler_seed += 1

            OT, OT_sorted = pairwise.run_entropic_gwot(
                compute_OT=compute_OT,
                delete_results=delete_results,
                OT_format=OT_format,
                save_dataframe=save_dataframe,
                target_device=target_device,
                sampler_seed=sampler_seed,
            )

            OT_list.append(OT)
            OT_sorted_list.append(OT_sorted)

        return OT_list, OT_sorted_list

    # evaluation methods
    def calc_accuracy(
        self,
        top_k_list: List[int],
        eval_type: str = "ot_plan",
        category_mat: Optional[Any] = None,
        barycenter: bool = False,
        return_dataframe: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Evaluation of the accuracy of the unsupervised alignment

        Args:
            top_k_list (List[int]):
                define the top k accuracy in list

            eval_type (str, optional):
                two ways to evaluate the accuracy as above. Defaults to "ot_plan".

            category_mat (Optional[Any], optional):
                This will be used for the category info. Defaults to None.

            barycenter (bool, optional):
                Indicates if the accuracy should be evaluated with respect to a barycenter representation.
                Defaults to False.

            return_dataframe (bool, optional):
                If True, the accuracy result will be returned in pandas.DataFrame format. Defaults to False.

        Returns:
            accuracy (Optional[pd.DataFrame]):
                A DataFrame containing accuracy metrics. Only returned if `return_dataframe` is True.
        """

        accuracy = pd.DataFrame({"top_n": top_k_list})

        for pairwise in self.pairwise_list:
            df = pairwise.calc_accuracy(
                top_k_list,
                eval_type=eval_type,
                metric=self.metric,
                barycenter=barycenter,
                category_mat=category_mat,
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

        print("Mean : \n", accuracy.iloc[:, 1:].mean(axis="columns"))

        if return_dataframe:
            return accuracy

    def _get_dataframe(self, eval_type="ot_plan", melt=True):
        if eval_type == "ot_plan":
            df = self.top_k_accuracy
        elif eval_type == "k_nearest":
            df = self.k_nearest_matching_rate
        elif eval_type == "category":
            df = self.category_level_accuracy

        cols = [col for col in df.columns if "top_n" not in col]
        df = df[cols]

        if melt:
            df = pd.melt(df.reset_index(), id_vars=['top_n'], value_name='matching rate').drop(columns=['variable'])

        return df

    # embedding methods
    def calc_embedding(
        self,
        dim: int,
        pivot: Union[int, str] = 0,
        emb_name: Optional[str] = "PCA",
        emb_transformer: Optional[TransformerMixin] = None,
        category_idx_list: Optional[List[int]] = None,
        return_data: bool = False,
        **kwargs,
    ) -> None:

        # procrustes alignment
        if pivot != "barycenter":
            self._procrustes_to_pivot(pivot)
            # for i in range(len(self.pairwise_list) // 2):
            #    pair = self.pairwise_list[i]
            #    pair.source.embedding = pair.get_new_source_embedding()
        else:
            assert self.barycenter is not None

        # get sort idx)
        sort_idx = (
            np.concatenate(category_idx_list) if category_idx_list is not None else None
        )

        if (sort_idx is None) and (
            self.representations_list[0].category_idx_list is not None
        ):
            sort_idx = np.concatenate(self.representations_list[0].category_idx_list)

        if sort_idx is None:
            sort_idx = np.arange(self.representations_list[0].embedding.shape[0])

        # get embedding
        embedding_list = []
        for i in range(len(self.representations_list)):
            embedding = self.representations_list[i].embedding[sort_idx, :]
            embedding_list.append(embedding)

        # get low dimensional embedding
        low_embedding_list, embedding_transformer = obtain_embedding(
            embedding_list,
            dim=dim,
            emb_name=emb_name,
            emb_transformer=emb_transformer,
            **kwargs,
        )

        # save
        self.low_embedding_list = low_embedding_list
        self.embedding_transformer = embedding_transformer

        if return_data:
            return low_embedding_list

    def _check_pairs(self, pivot):
        the_others = set()
        pivot_idx_list = []  # [pair_idx, paivot_idx]
        for i, pair in enumerate(self.all_pair_list):
            if pivot in pair:
                the_others.add(filter(lambda x: x != pivot, pair))

                pivot_idx = pair.index(pivot)
                pivot_idx_list.append([i, pivot_idx])

        return the_others, pivot_idx_list

    def _procrustes_to_pivot(self, pivot):
        the_others, pivot_idx_list = self._check_pairs(pivot)

        # check whether 'pair_list' includes all pairs between the pivot and the other Representations
        assert (
            len(the_others) == len(self.representations_list) - 1
        ), "'pair_list' must include all pairs between the pivot and the other Representations."

        for pair_idx, pivot_idx in pivot_idx_list:
            pairwise = self.pairwise_list[pair_idx]

            if pivot_idx == 0:  # when the pivot is the source of the pairwise
                source_idx = 1
                OT = pairwise.OT.T

            elif pivot_idx == 1:  # when the pivot is the target of the pairwise
                source_idx = 0
                OT = pairwise.OT

            pivot = (pairwise.source, pairwise.target)[pivot_idx]
            source = (pairwise.source, pairwise.target)[source_idx]

            source.embedding = pairwise.procrustes(
                pivot.embedding, source.embedding, OT
            )

    # experimental GWOT methods
    def gwot_after_entropic(
        self,
        top_k: Optional[int] = None,
        parallel_method: Optional[str] = None,
        OT_format: str = "default",
        ticks: Optional[str] = None,
        category_mat: Optional[Any] = None,
    ) -> None:
        """Run GWOT without entropy by setting the optimized entropic GWOT as the initial plan.

        This method computes the GWOT for each pair of representations without entropy regularization
        by initializing the transport plan with the result obtained from entropic GWOT.

        Args:
            top_k (int, optional):
                this will be used for loading the optimized OT from the bottom k of lowest GWD (= top_k) value.
                Defaults to None. If None, all the computed OT will be used for GWOT without entropy.

            parallel_method (Optional[str], optional):
                parallel method to compute GWOT. Defaults to "multithread".

            OT_format (str, optional):
                format of sim_mat to visualize.
                Options are "default", "sorted", and "both". Defaults to "default".

            ticks (Optional[str], optional):
                you can use "objects" or "category (if existed)" or "None". Defaults to None.

            category_mat (Optional[Any], optional):
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
                    visualization_config=visualization_config,
                )

            return None

        with pool:
            if self.config.to_types == "numpy":
                if self.config.multi_gpu != False:
                    warnings.warn(
                        "numpy doesn't use GPU. Please 'multi_GPU = False'.",
                        UserWarning,
                    )
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
                    visualization_config=visualization_config,
                )
                processes.append(future)

            for future in as_completed(processes):
                future.result()

    def drop_gw_alignment_files(
        self,
        drop_filenames: Optional[List[str]] = None,
        drop_all: bool = False,
    ) -> None:
        """Delete the specified database and directory with the given filename

        Args:
            drop_filenames (Optional[List[str]], optional):
                A list of filenames corresponding to the results that are to be deleted.
                If `drop_all` is set to True, this parameter is ignored. Defaults to None.
            drop_all (bool, optional):
                If set to True, all results will be deleted regardless of the `drop_filenames` parameter.
                Defaults to False.

        Raises:
            ValueError: If neither `drop_filenames` is specified nor `drop_all` is set to True.
        """
        if drop_all:
            drop_filenames = [pairwise.filename for pairwise in self.pairwise_list]

        if drop_filenames is None:
            raise ValueError(
                "Specify the results name in drop_filenames or set drop_all=True"
            )

        for pairwise in self.pairwise_list:
            if (pairwise.filename not in drop_filenames) or (
                not database_exists(pairwise.storage)
            ):
                continue
            pairwise.delete_prev_results()

    # barycenter GWOT methods
    def calc_barycenter(self, X_init=None):
        embedding_list = [
            representation.embedding for representation in self.representations_list
        ]

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
        compute_OT: bool = False,
        delete_results: bool = False,
        return_data: bool = False,
        return_figure: bool = True,
        OT_format: str = "default",
        show_log: bool = False,
        fig_dir: Optional[str] = None,
        ticks: Optional[str] = None,
    ) -> Optional[List[np.ndarray]]:
        """The unuspervised alignment method using Wasserstein barycenter proposed by Lian et al. (2021).

        Args:
            pivot (int):
                The representation to which the other representations are aligned initially using gw alignment
            n_iter (int):
                The number of iterations to calculate the location of the barycenter.
            compute_OT (bool, optional):
                If True, the GWOT will be computed. If False, saved results will be loaded. Defaults to False.
            delete_results (bool, optional):
                If True, all saved results will be deleted. Defaults to False.
            return_data (bool, optional):
                If True, returns the computed OT. Defaults to False.
            return_figure (bool, optional):
                If True, visualizes the results. Defaults to True.
            OT_format (str, optional):
                Format of similarity matrix to visualize. Options include "default", "sorted", and "both".
                Defaults to "default".
            visualization_config (VisualizationConfig, optional):
                Container of parameters used for figure. Defaults to VisualizationConfig().
            show_log (bool, optional):
                If True, displays the evaluation figure of GWOT. Defaults to False.
            fig_dir (Optional[str], optional):
                Directory where figures are saved. Defaults to None.
            ticks (Optional[str], optional):
                Specifies the labels for the ticks.. Defaults to None.

        Returns:
            Optional[List[np.ndarray]]:
                If return_data is True, returns the computed OT.
        """

        # assert self.all_pair_list == range(len(self.pairwise_list))

        # Select the pivot
        pivot_representation = self.representations_list[pivot]
        others_representaions = (
            self.representations_list[:pivot] + self.representations_list[pivot + 1 :]
        )

        # GW alignment to the pivot
        for representation in others_representaions:

            pairwise = self._pairwise_from_representations(
                config=self.config,
                source=representation,
                target=pivot_representation,
            )

            pairwise.run_entropic_gwot(
                compute_OT=compute_OT,
                delete_results=delete_results,
                return_data=return_data,
                return_figure=return_figure,
                OT_format=OT_format,
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
            func_for_sort_sim_mat=self.representations_list[0].func_for_sort_sim_mat,
        )

        # Set pairwises whose target are the barycenter
        pairwise_barycenters = []
        for representation in self.representations_list:

            pairwise = self._pairwise_from_representations(
                config=self.config,
                source=representation,
                target=self.barycenter,
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
