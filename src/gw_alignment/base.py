#%%
import gc
import os
from typing import Any, List, Tuple

import optuna
import torch

# warnings.simplefilter("ignore")
from .main_compute import EntropicGWComputation, EntropicSemirelaxedGWComputation, GWComputation, EntropicPartialGWComputation

# nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv


# %%
class GW_Alignment:
    """A class that encapsulates parameters and methods for the Gromov-Wasserstein alignment.

    This class encapsulates the necessary parameters and methods for the alignment process,
    including the dissimilarity matrices for the source and target, the path to save results,
    and various algorithm parameters such as the maximum number of iterations and the method for the Sinkhorn algorithm.
    This class also sets up the main computation object for performing the GW alignment.

    Attributes:
        to_types (str): Specifies the type of data structure to be used, either "torch" or "numpy".
        data_type (str): Specifies the type of data to be used in computation.
        sinkhorn_method (str): Method used for the solver. Options are "sinkhorn", "sinkhorn_log", "greenkhorn",
                               "sinkhorn_stabilized", or "sinkhorn_epsilon_scaling".
        source_size (int): Size of the source distribution.
        target_size (int): Size of the target distribution.
        data_path (str): Directory to save the computation results.
        n_iter (int): Number of initial plans evaluated in optimization.
        main_compute (MainGromovWasserstainComputation): Main computation object for GW alignment.
    """

    def __init__(
        self,
        source_dist: Any,
        target_dist: Any,
        data_path: str,
        max_iter: int = 1000,
        numItermax: int = 1000,
        n_iter: int = 20,
        problem_type: str = "entropic_gromov_wasserstein",
        to_types: str = "torch",
        data_type: str = "double",
        sinkhorn_method: str = "sinkhorn",
        **kwargs
    ) -> None:
        """Initialize the Gromov-Wasserstein alignment object.

        Args:
            source_dist (Any):  Array-like, shape (n_source, n_source).
                                Dissimilarity matrix of the source data.
            target_dist (Any):  Array-like, shape (n_target, n_target).
                                Dissimilarity matrix of the target data.
            data_path (str):    Directory to save the computation results.
            max_iter (int, optional):   Maximum number of iterations for entropic
                                        Gromov-Wasserstein alignment by POT.
                                        Defaults to 1000.
            numItermax (int, optional): Maximum number of iterations for the
                                        Sinkhorn algorithm. Defaults to 1000.
            n_iter (int, optional): Number of initial plans evaluated in optimization. Defaults to 20.
            to_types (str, optional):   Specifies the type of data structure to be used,
                                        either "torch" or "numpy". Defaults to "torch".
            data_type (str, optional):  Specifies the type of data to be used
                                        in computation. Defaults to "double".
            sinkhorn_method (str, optional):    Method used for the solver. Options are
                                                "sinkhorn", "sinkhorn_log", "greenkhorn",
                                                "sinkhorn_stabilized", or "sinkhorn_epsilon_scaling".
                                                Defaults to "sinkhorn".
        """
        self.to_types = to_types
        self.data_type = data_type
        self.sinkhorn_method = sinkhorn_method
        self.problem_type = problem_type

        # distribution in the source space, and target space
        self.source_size = len(source_dist)
        self.target_size = len(target_dist)

        self.data_path = data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)

        self.n_iter = n_iter

        self.main_compute = self.load_gw_computation(
            problem_type=problem_type,
            source_dist=source_dist,
            target_dist=target_dist,
            to_types=to_types,
            data_type=data_type,
            max_iter=max_iter,
            numItermax=numItermax,
            n_iter=n_iter,
            sinkhorn_method=sinkhorn_method,
            **kwargs
        )

    def __call__(
        self,
        trial: optuna.trial.Trial,
        device: str,
        init_mat_plan: str,
        eps_list: List[float],
        eps_log: bool = True
    ) -> float:
        """The function that performs one trial of Gromov-Wasserstein alignment.

        The main computation is performed by an instance of MainGromovWasserstainComputation.

        Args:
            trial (optuna.trial.Trial): The trial object from the Optuna for hyperparameter optimization.
            device (str): The device to be used for computation, either "cpu" or "cuda".
            init_mat_plan (str): The method to be used for the initial plan. Options are "uniform",
                                 "diag", "random", "permutation" or "user_define".
            eps_list (List[float]): A list containing the lower and upper bounds for epsilon.
            eps_log (bool, optional): A flag to determine if the epsilon search is in logarithmic scale.
        """
        if self.to_types == "numpy":
            assert device == "cpu", "numpy does not run in CUDA."

        """
        1.  define hyperparameter (eps, T)
        """

        trial, eps = self.define_eps_range(trial, eps_list, eps_log)
        trial.set_user_attr("source_size", self.source_size)
        trial.set_user_attr("target_size", self.target_size)

        """
        2.  Compute GW alignment with hyperparameters defined above.
        """
        logv, trial = self.main_compute.compute_GW_with_init_plans(
            trial,
            eps,
            init_mat_plan,
            device,
        )

        """
        3.  count the accuracy of alignment and save the results if computation was finished in the right way.
            If not, set the result of accuracy and gw_loss as float('nan'), respectively. This will be used as a handy marker as bad results to be removed in the evaluation analysis.
        """
        gw = logv["ot"]
        gw_loss = logv["gw_dist"]
        self.main_compute.back_end.save_computed_results(gw, self.data_path, trial.number)

        """
        4. delete unnecessary memory for next computation. If not, memory error would happen especially when using CUDA.
        """

        del gw, logv
        torch.cuda.empty_cache()
        gc.collect()

        return gw_loss

    def define_eps_range(
        self,
        trial: optuna.trial.Trial,
        eps_list: List[float],
        eps_log: bool
    ) -> Tuple[optuna.trial.Trial, float]:
        """The function that defines the range of epsilon for the Gromov-Wasserstein computation based on the given trial and epsilon list.

        Args:
            trial (optuna.trial.Trial): The trial object from the Optuna for hyperparameter optimization.
            eps_list (List[float]):     A list containing the lower and upper bounds for epsilon.
                                        If a third value is provided, it is used as the step size for epsilon.
            eps_log (bool): A flag to determine if the epsilon search is in logarithmic scale.

        Raises:
            ValueError: Raised if the provided epsilon list and epsilon log does not match.

        Returns:
            trial (optuna.trial.Trial): The same trial object provided as input.
            eps (float):    The sampled epsilon value.
        """
        if len(eps_list) == 2:
            ep_lower, ep_upper = eps_list
            eps = trial.suggest_float("eps", ep_lower, ep_upper, log=eps_log)
        elif len(eps_list) == 3:
            ep_lower, ep_upper, ep_step = eps_list
            eps = trial.suggest_float("eps", ep_lower, ep_upper, step=ep_step)
        else:
            raise ValueError("The eps_list and/or eps_log doesn't match.")

        return trial, eps

    def load_gw_computation(
        self,
        problem_type: str,
        **kwargs
    ) -> GWComputation:
        if problem_type == "entropic_gromov_wasserstein":
            gwc = EntropicGWComputation(**kwargs)

        elif problem_type == "entropic_semirelaxed_gromov_wasserstein":
            gwc = EntropicSemirelaxedGWComputation(**kwargs)

        elif problem_type == "entropic_partial_gromov_wasserstein":
            gwc = EntropicPartialGWComputation(**kwargs)

        else:
            raise ValueError(f"problem type {problem_type} is not defined.")

        return gwc
