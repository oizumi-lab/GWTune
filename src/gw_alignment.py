#%%
import gc
import math
import os
from typing import List, Tuple, Any, Union, Dict, Iterable, Optional

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import optuna
import ot
import seaborn as sns
import torch
from tqdm.auto import tqdm

# warnings.simplefilter("ignore")
from .utils.backend import Backend
from .utils.init_matrix import InitMatrix

# nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv


# %%
class GW_Alignment:
    """The main object for entropic Gromov-Wasserstein (GW) alignment.

    This class encapsulates the necessary parameters and methods for the alignment process,
    including the dissimilarity matrices for the source and target, the path to save results, and various algorithm parameters such as the maximum number of iterations and the method for the Sinkhorn algorithm. This class also sets up the main computation object for performing the GW alignment.

    Attributes:
        to_types (str): Specifies the type of data structure to be used for computations,
                        either "torch" or "numpy".
        data_type (str): Specifies the type of data to be used in computation.
        sinkhorn_method (str): Method used for the solver. Options are "sinkhorn", "sinkhorn_log",
                               "greenkhorn", "sinkhorn_stabilized", or "sinkhorn_epsilon_scaling".
        source_size (int): The size (number of elements) of the source distribution.
        target_size (int): The size (number of elements) of the target distribution.
        data_path (str): Directory to save the computation results.
        n_iter (int):   The number of initial plans evaluated during optimization
                        when init_mat_plan is set to "random", "permutation", or "user_define".
        main_compute (MainGromovWasserstainComputation): The main computation object for performing the GW alignment.
    """
    def __init__(
        self,
        source_dist: Any,
        target_dist: Any,
        data_path: str,
        max_iter: int = 1000,
        numItermax: int = 1000,
        n_iter: int = 20,
        to_types: str = "torch",
        data_type: str = "double",
        sinkhorn_method: str = "sinkhorn",
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
            n_iter (int, optional):  Number of trials, i.e., the number of
                                        initial plans evaluated in optimization.
                                        Defaults to 20.
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

        # distribution in the source space, and target space
        self.source_size = len(source_dist)
        self.target_size = len(target_dist)

        self.data_path = data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)

        self.n_iter = n_iter

        self.main_compute = MainGromovWasserstainComputation(
            source_dist,
            target_dist,
            self.to_types,
            data_type=self.data_type,
            max_iter=max_iter,
            numItermax=numItermax,
            n_iter=n_iter,
        )

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
            device (str):   The device to be used for computation, either "cpu" or "cuda".
            init_mat_plan (str):    The method to be used for the initial plan. Options are "uniform",
                                    "diag", "random", "permutation" or "user_define".
            eps_list (List[float]): A list containing the lower and upper bounds for epsilon.
            eps_log (bool, optional):   A flag to determine if the epsilon search is in logarithmic scale.
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
            sinkhorn_method = self.sinkhorn_method
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


class MainGromovWasserstainComputation:
    """The object responsible for the specific computations of the entropic Gromov-Wasserstein alignment.

    Attributes:
        to_types (str): Specifies the type of data structure to be used, either "torch" or "numpy".
        data_type (str): Specifies the type of data to be used in computation.
        source_dist (Any): Array-like, shape (n_source, n_source). Dissimilarity matrix of the source data.
        target_dist (Any): Array-like, shape (n_target, n_target). Dissimilarity matrix of the target data.
        p (Any): Probability distribution for the source data.
        q (Any): Probability distribution for the target data.
        source_size (int): Size of the source data.
        target_size (int): Size of the target data.
        init_mat_builder (InitMatrix): Object for building initial matrices.
        max_iter (int): Maximum number of iterations for entropic Gromov-Wasserstein alignment by POT.
        numItermax (int): Maximum number of iterations for the Sinkhorn algorithm.
        n_iter (int): Number of trials, i.e., the number of initial plans evaluated in optimization.
        back_end (Backend): Backend object to handle computations on the specified device and data type.
    """
    def __init__(
        self,
        source_dist: Any,
        target_dist: Any,
        to_types: str,
        data_type: str = 'double',
        max_iter: int = 1000,
        numItermax: int = 1000,
        n_iter: int = 20
    ) -> None:
        """Initialize the Gromov-Wasserstein alignment computation object.

        Args:
            source_dist (Any):          Array-like, shape (n_source, n_source).
                                        Dissimilarity matrix of the source data.
            target_dist (Any):          Array-like, shape (n_target, n_target).
                                        Dissimilarity matrix of the target data.
            to_types (str, optional):   Specifies the type of data structure to be used,
                                        either "torch" or "numpy". Defaults to "torch".
            data_type (str, optional):  Specifies the type of data to be used
                                        in computation. Defaults to "double".
            max_iter (int, optional):   Maximum number of iterations for entropic
                                        Gromov-Wasserstein alignment by POT.
                                        Defaults to 1000.
            numItermax (int, optional): Maximum number of iterations for the
                                        Sinkhorn algorithm. Defaults to 1000.
            n_iter (int, optional):  Number of trials, i.e., the number of
                                        initial plans evaluated in optimization.
                                        Defaults to 20.
        """
        self.to_types = to_types
        self.data_type = data_type

        p = ot.unif(len(source_dist))
        q = ot.unif(len(target_dist))

        self.source_dist, self.target_dist, self.p, self.q = source_dist, target_dist, p, q

        self.source_size = len(source_dist)
        self.target_size = len(target_dist)

        # init matrix
        self.init_mat_builder = InitMatrix(self.source_size, self.target_size)

        # parameter for entropic gw alignment by POT
        self.max_iter = max_iter

        # parameter for sinkhorn
        self.numItermax = numItermax

        # the number of iteration mainly used for random init matrix.
        self.n_iter = n_iter

        self.back_end = Backend("cpu", self.to_types, self.data_type)

    def entropic_gw(
        self,
        device: str,
        eps: float,
        T: Any,
        tol: float = 1e-9,
        sinkhorn_method: str = "sinkhorn",
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Performs the entropic Gromov-Wasserstein alignment.

        This function utilizes Sinkhorn's algorithm to iteratively solve the entropic Gromov-Wasserstein problem.
        The algorithm terminates when the error between iterations is less than the provided tolerance or
        when the maximum number of iterations is reached.

        Args:
            device (str):   The device to be used for computation, either "cpu" or "cuda".
            epsilon (float):    Regularization term for Gromov-Wasserstein alignment.
            T (Any):    Initial plan for Gromov-Wasserstein alignment.
            tol (float, optional):  Stop threshold on error. Defaults to 1e-9.
            sinkhorn_method (str, optional):    Method used for the solver. Options are "sinkhorn",
                                                "sinkhorn_log", "greenkhorn", "sinkhorn_stabilized",
                                                or "sinkhorn_epsilon_scaling". Defaults to "sinkhorn".
            verbose (bool, optional): Print information along iterations. Defaults to False.

        Returns:
            log (Dict[str, Any]):  A dictionary containing the optimization results.
        """

        # all the variable are placed on "device" here.
        self.back_end.device = device
        C1, C2, p, q, T = self.back_end(self.source_dist, self.target_dist, self.p, self.q, T)
        constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun="square_loss")

        cpt = 0
        err = 1
        log = {"err": []}

        while err > tol and cpt < self.max_iter:
            Tprev = T
            tens = ot.gromov.gwggrad(constC, hC1, hC2, T)
            T = ot.bregman.sinkhorn(p, q, tens, eps, method=sinkhorn_method, numItermax=self.numItermax)

            if cpt % 10 == 0:
                err = self.back_end.nx.norm(T - Tprev)
                if log:
                    log["err"].append(err)
                if verbose:
                    if cpt % 200 == 0:
                        print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                    print("{:5d}|{:8e}|".format(cpt, err))
            cpt += 1

        log["gw_dist"] = ot.gromov.gwloss(constC, hC1, hC2, T)
        log["ot"] = T

        return log

    def gw_alignment_computation(
        self,
        init_mat: Any,
        eps: float,
        device: str,
        sinkhorn_method: str = "sinkhorn",
    ) -> Dict[str, Any]:
        """Modify the result of Gromov-Wasserstein alignment to make it easier to optimize.

        Args:
            init_mat (Any): The initial value of transportation plan for Gromov-Wasserstein alignment.
            eps (float):    Regularization term for Gromov-Wasserstein alignment.
            device (str):   The device to be used for computation, either "cpu" or "cuda".
            sinkhorn_method (str, optional):    Method used for the solver. Options are "sinkhorn", "sinkhorn_log",
                                                "greenkhorn", "sinkhorn_stabilized", or "sinkhorn_epsilon_scaling".
                                                Defaults to "sinkhorn".

        Returns:
            log:    A dictionary containing the optimization results.
        """

        logv = self.entropic_gw(
            device,
            eps,
            init_mat,
            sinkhorn_method=sinkhorn_method,
        )

        if self.back_end.check_zeros(logv["ot"]):
            logv["gw_dist"] = float("nan")
            logv["acc"] = float("nan")

        else:
            pred = self.back_end.nx.argmax(logv["ot"], 1)
            correct = (pred == self.back_end.nx.arange(len(logv["ot"]), type_as=logv["ot"])).sum()
            logv["acc"] = correct / len(logv["ot"])

        return logv

    def _save_results(
        self,
        gw_loss: float,
        acc: float,
        trial: optuna.trial.Trial,
        init_mat_plan: str,
        num_iter: Optional[int] = None,
        seed: Optional[int] = None
    ) -> optuna.trial.Trial:
        """Save the results of one trial of Gromov-Wasserstein alignment.

        This function takes the Gromov-Wasserstein loss and accuracy from a trial, converts them to a suitable
        format using the backend, and then stores them as user attributes in the trial object.

        Args:
            gw_loss (float): The Gromov-Wasserstein loss(distance).
            acc (float): The accuracy of the optimal transportation plan.
            trial (optuna.trial.Trial): The trial object from the Optuna.
            init_mat_plan (str):    The initialization method of transportation plan for Gromov-Wasserstein alignment.
                                    Options are "random", "permutation", "user_define", "uniform", or "diag".
            num_iter (int, optional): The number of optimizations within a single trial. Defaults to None.
            seed (int, optional): The seed used for random number generation in the trial. Defaults to None.

        Returns:
            trial (optuna.trial.Trial): The trial object from the Optuna.
        """

        gw_loss, acc = self.back_end.get_item_from_torch_or_jax(gw_loss, acc)

        trial.set_user_attr("best_acc", acc)
        if init_mat_plan in ["random", "permutation"]:
            assert num_iter is not None, "num_iter must be provided for random and permutation initialization."
            assert seed is not None, "seed must be provided for random and permutation initialization."
            trial.set_user_attr("best_iter", num_iter)
            trial.set_user_attr("best_seed", int(seed))

        return trial

    def _check_pruner_should_work(
        self,
        gw_loss: float,
        trial: optuna.trial.Trial,
        init_mat_plan: str,
        eps: float,
        num_iter: Optional[int] = None
    ):
        """Pruner will work here.

        This function evaluates the current state of a trial based on the Gromov-Wasserstein loss and certain conditions.
        It reports the loss to the trial and raises a pruning exception if necessary.
        The function is part of an optimization process where it helps in identifying and removing less promising trials,
        thereby improving the efficiency of the optimization.

        Args:
            gw_loss (float): The Gromov-Wasserstein loss(distance).
            trial (optuna.trial.Trial): The trial object from the Optuna.
            init_mat_plan (str):    The initialization method of transportation plan for Gromov-Wasserstein alignment.
                                    Options are "random", "permutation", "user_define", "uniform", or "diag".
            eps (float): Regularization term.
            num_iter (int, optional): The number of optimizations within a single trial. Defaults to None.

        Raises:
            optuna.TrialPruned: The trial was pruned.
        """

        if math.isinf(gw_loss) or gw_loss <= 0.0:
            raise optuna.TrialPruned(f"Trial for '{init_mat_plan}' was pruned with parameters: {{'eps': {eps:.5e}, 'gw_loss': '{gw_loss:.5e}'}}")

        if (init_mat_plan not in ["random", "permutation", "user_define"]) and math.isnan(gw_loss):
            raise optuna.TrialPruned(f"Trial for '{init_mat_plan}' was pruned with parameters: {{'eps': {eps:.5e}, 'gw_loss': '{gw_loss:.5e}'}}")

        if (init_mat_plan in ["uniform", "diag"]) and math.isnan(gw_loss):
            raise optuna.TrialPruned(f"Trial for '{init_mat_plan}' was pruned with parameters: {{'eps': {eps:.5e}, 'gw_loss': '{gw_loss:.5e}'}}")

        if num_iter is None:
            trial.report(gw_loss, 0)
        else:
            trial.report(gw_loss, num_iter)

        if trial.should_prune():
            if num_iter is None:
                raise optuna.TrialPruned(
                    f"Trial for '{init_mat_plan}' was pruned with parameters: {{'eps': {eps:.5e}, 'gw_loss': '{gw_loss:.5e}'}}"
                )
            else:
                raise optuna.TrialPruned(
                    f"Trial for '{init_mat_plan}' was pruned at iteration {num_iter} with parameters: {{'eps': {eps:.5e}, 'gw_loss': '{gw_loss:.5e}'}}"
                )

    def _compute_GW_with_init_plans(
        self,
        trial: optuna.trial.Trial,
        init_mat_plan: str,
        eps: float,
        device: Any,
        sinkhorn_method: str,
        num_iter: Optional[int] = None,
        seed: Optional[Any] = None
    ) -> Tuple[dict, optuna.trial.Trial, Optional[bool]]:
        """Computes the Gromov-Wasserstein alignment with the provided initial transportation plan.

        Args:
            trial (optuna.trial.Trial): The trial object from the Optuna.
            init_mat_plan (str): The initialization method of transportation plan for Gromov-Wasserstein alignment. Options are "random", "permutation", "user_define", "uniform", or "diag".
            eps (float): Regularization term.
            device (Any): The device to be used for computation, either "cpu" or "cuda".
            sinkhorn_method (str):  Method used for the solver. Options are "sinkhorn", "sinkhorn_log", "greenkhorn",
                                    "sinkhorn_stabilized", or "sinkhorn_epsilon_scaling".
            num_iter (int, optional): The number of optimizations within a single trial. Defaults to None.
            seed (Any, optional): Seed for generating the initial matrix. Defaults to None.

        Returns:
            logv (dict): A dictionary containing the Gromov-Wasserstein loss(distance) and accuracy.
            trial (optuna.trial.Trial): The trial object from the Optuna.
            best_flag (Optional[bool]): A flag indicating whether the current trial is the best trial.
        """

        if init_mat_plan == "user_define":
            init_mat = seed
        else:
            init_mat = self.init_mat_builder.make_initial_T(init_mat_plan, seed)

        logv = self.gw_alignment_computation(
            init_mat,
            eps,
            device,
            sinkhorn_method=sinkhorn_method,
        )

        if init_mat_plan in ["uniform", "diag"]:
            best_flag = None
            trial = self._save_results(
                logv["gw_dist"],
                logv["acc"],
                trial,
                init_mat_plan,
            )

        elif init_mat_plan in ["random", "permutation", "user_define"]:
            if logv["gw_dist"] < self.best_gw_loss:
                best_flag = True
                self.best_gw_loss = logv["gw_dist"]

                trial = self._save_results(
                    logv["gw_dist"],
                    logv["acc"],
                    trial,
                    init_mat_plan,
                    num_iter=num_iter,
                    seed=seed,
                )

            else:
                best_flag = False

        self._check_pruner_should_work(
            logv["gw_dist"],
            trial,
            init_mat_plan,
            eps,
            num_iter=num_iter,
        )

        return logv, trial, best_flag

    def compute_GW_with_init_plans(
        self,
        trial: optuna.trial.Trial,
        eps: float,
        init_mat_plan: str,
        device: str,
        sinkhorn_method: str = "sinkhorn"
    ) -> Tuple[dict, optuna.trial.Trial]:
        """Calculate Gromov-Wasserstein (GW) alignment based on specified parameters.

        This function computes the GW alignment given the initialization plan and the parameter.

        Args:
            trial (optuna.trial.Trial): The trial object from the Optuna.
            eps (float): Regularization term.
            init_mat_plan (str):    The initialization method of transportation plan for Gromov-Wasserstein alignment.
                                    Options are "random", "permutation", "user_define", "uniform", or "diag".
            device (str): The device to be used for computation, either "cpu" or "cuda".
            sinkhorn_method (str, optional):    Method used for the solver. Options are "sinkhorn", "sinkhorn_log",
                                                "greenkhorn", "sinkhorn_stabilized", or "sinkhorn_epsilon_scaling".
                                                Defaults to "sinkhorn".

        Raises:
            optuna.TrialPruned: Raised when all iterations fail with the given parameters.
            ValueError: Raised when an undefined initialization method is provided.

        Returns:
            best_logv (dict): A dictionary containing the best Gromov-Wasserstein loss(distance) and accuracy.
            trial (optuna.trial.Trial): The trial object from the Optuna.
        """

        if init_mat_plan in ["uniform", "diag"]:
            logv, trial, _ = self._compute_GW_with_init_plans(
                trial,
                init_mat_plan,
                eps,
                device,
                sinkhorn_method,
            )
            return logv, trial

        elif init_mat_plan in ["random", "permutation", "user_define"]:
            self.best_gw_loss = float("inf")

            if init_mat_plan in ["random", "permutation"]:
                pbar = tqdm(np.random.randint(0, 100000, self.n_iter))

            if init_mat_plan == "user_define":
                pbar = tqdm(self.init_mat_builder.user_define_init_mat_list)

            pbar.set_description(f"Trial No.{trial.number}, eps:{eps:.3e}")

            for i, seed in enumerate(pbar):
                logv, trial, best_flag = self._compute_GW_with_init_plans(
                    trial,
                    init_mat_plan,
                    eps,
                    device,
                    sinkhorn_method,
                    num_iter=i,
                    seed=seed,
                )

                if best_flag:
                    best_logv = logv

            if math.isinf(self.best_gw_loss) or self.best_gw_loss <= 0.0 or math.isnan(self.best_gw_loss):
                raise optuna.TrialPruned(
                    f"All iteration was failed with parameters: {{'eps': {eps}, 'initialize': '{init_mat_plan}'}}"
                )

            else:
                return best_logv, trial

        else:
            raise ValueError("Not defined initialize matrix.")

# %%
