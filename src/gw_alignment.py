#%%
import math, os, gc, warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import optuna
import ot
import torch
from tqdm.auto import tqdm

# warnings.simplefilter("ignore")
from .utils.backend import Backend
from .utils.init_matrix import InitMatrix


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
        storage: str,
        study_name: str,
        max_iter: int = 1000,
        numItermax: int = 1000,
        n_iter: int = 20,
        fix_random_init_seed : Optional[int] = None,
        device: str = "cpu",
        gw_type: str = "entropic_gromov_wasserstein",
        to_types: str = "numpy",
        data_type: str = "double",
        sinkhorn_method: str = "sinkhorn",
        instance_name: Optional[str]= None,
        **kwargs,
    ) -> None:
        """Initialize the Gromov-Wasserstein alignment object.

        Args:
            source_dist (Any):
                Array-like, shape (n_source, n_source). Dissimilarity matrix of the source data.
            target_dist (Any):
                Array-like, shape (n_target, n_target).Dissimilarity matrix of the target data.
            data_path (str):
                Directory to save the computation results.
            max_iter (int, optional):
                Maximum number of iterations for entropic Gromov-Wasserstein alignment by POT.
                Defaults to 1000.
            numItermax (int, optional):
                Maximum number of iterations for the Sinkhorn algorithm. Defaults to 1000.
            n_iter (int, optional):
                Number of initial plans evaluated in optimization. Defaults to 20.
            gw_type (str, optional):
                Type of Gromov-Wasserstein alignment to be used. Options are "entropic_gromov_wasserstein",
                "entropic_semirelaxed_gromov_wasserstein", or "entropic_partial_gromov_wasserstein".
                Defaults to "entropic_gromov_wasserstein".
            to_types (str, optional):
                Specifies the type of data structure to be used, either "torch" or "numpy".
                Defaults to "torch".
            data_type (str, optional):
                Specifies the type of data to be used in computation. Defaults to "double".
            sinkhorn_method (str, optional):
                Method used for the solver. Options are "sinkhorn", "sinkhorn_log", "greenkhorn",
                "sinkhorn_stabilized", or "sinkhorn_epsilon_scaling". Defaults to "sinkhorn".
        """
        self.to_types = to_types
        self.data_type = data_type
        self.sinkhorn_method = sinkhorn_method
        self.instance_name = instance_name
        self.gw_type = gw_type
        self.device = device
        
        if self.to_types == "numpy":
            assert device == "cpu", "numpy does not run in CUDA."

        # distribution in the source space, and target space
        self.source_size = len(source_dist)
        self.target_size = len(target_dist)

        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)

        self.n_iter = n_iter
        
        # check a existed database to get the best gw_loss
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            df = study.trials_dataframe()
            min_loss = df["value"].min()
            
            if math.isnan(min_loss):
                self.best_gw_loss = float("inf")
            else: 
                self.best_gw_loss = min_loss

            # print("existed best_loss :", self.best_gw_loss)
        except KeyError:
            self.best_gw_loss = float("inf")
            # print("best_loss :", self.best_gw_loss)

        self.main_compute = MainGromovWasserstainComputation(
            source_dist=source_dist,
            target_dist=target_dist,
            device=device,
            to_types=to_types,
            data_type=data_type,
            max_iter=max_iter,
            numItermax=numItermax,
            n_iter=n_iter,
            fix_random_init_seed=fix_random_init_seed,
            gw_type=gw_type,
            sinkhorn_method=sinkhorn_method,
            instance_name=instance_name,
            **kwargs
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
        init_mat_plan: str,
        eps_list: List[float],
        eps_log: bool = True
    ) -> float:
        """The function that performs one trial of Gromov-Wasserstein alignment.

        The main computation is performed by an instance of MainGromovWasserstainComputation.

        Args:
            trial (optuna.trial.Trial): The trial object from the Optuna for hyperparameter optimization.
            init_mat_plan (str): The method to be used for the initial plan. Options are "uniform",
                                 "diag", "random", "permutation" or "user_define".
            eps_list (List[float]): A list containing the lower and upper bounds for epsilon.
            eps_log (bool, optional): A flag to determine if the epsilon search is in logarithmic scale.
        """

        """
        1.  define hyperparameter (eps, T)
        """

        trial, eps = self.define_eps_range(trial, eps_list, eps_log)
        trial.set_user_attr("source_size", self.source_size)
        trial.set_user_attr("target_size", self.target_size)

        """
        2.  Compute GW alignment with hyperparameters defined above.
        """
        logv, trial = self.main_compute.compute_GW_with_init_plans(trial, eps, init_mat_plan)

        """
        3.  count the accuracy of alignment and save the results if computation was finished in the right way.
            If not, set the result of accuracy and gw_loss as float('nan'), respectively. This will be used as a handy marker as bad results to be removed in the evaluation analysis.
        """
        gw = logv["ot"]
        gw_loss = logv["gw_dist"]
        iteration = logv["cpt"]
        
        trial.set_user_attr("iteration", iteration)

        if gw_loss < self.best_gw_loss:
            self.best_gw_loss = gw_loss
        
        self.main_compute.back_end.save_computed_results(gw, self.data_path, trial.number)

        """
        4. delete unnecessary memory for next computation. If not, memory error would happen especially when using CUDA.
        """

        del gw, logv
        torch.cuda.empty_cache()
        gc.collect()

        return gw_loss

class MainGromovWasserstainComputation:
    """A class responsible for the specific computations of the entropic Gromov-Wasserstein alignment.

    This class manages the core computation processes for the Gromov-Wasserstein alignment. It deals with
    initializing the computations, performing the entropic Gromov-Wasserstein optimization, and saving
    the results of the optimization. The class provides a comprehensive suite of methods to manage and
    manipulate the optimization process.
    """

    def __init__(
        self,
        source_dist: Any,
        target_dist: Any,
        p: Optional[Any] = None,
        q: Optional[Any] = None,
        device: str = "cpu",
        to_types: str = "numpy",
        data_type: str = 'double',
        max_iter: int = 1000,
        numItermax: int = 1000,
        n_iter: int = 20,
        fix_random_init_seed: Optional[int] = None,
        gw_type:str = "entropic_gromov_wasserstein",
        sinkhorn_method: str = "sinkhorn",
        instance_name: Optional[str]= None,
        *,
        first_random_init_seed: Optional[int] = None,
        tol: float = 1e-9,
        verbose: bool = False,
        m: Optional[float]=None,
    ) -> None:
        """Initialize the Gromov-Wasserstein alignment computation object.

        Args:
            source_dist (Any):          Array-like, shape (n_source, n_source).
                                        Dissimilarity matrix of the source data.
            target_dist (Any):          Array-like, shape (n_target, n_target).
                                        Dissimilarity matrix of the target data.
            device (str):                 The device to be used for computation, either "cpu" or "cuda".
            to_types (str, optional):   Specifies the type of data structure to be used,
                                        either "torch" or "numpy". Defaults to "torch".
            data_type (str, optional):  Specifies the type of data to be used
                                        in computation. Defaults to "double".
            max_iter (int, optional):   Maximum number of iterations for entropic
                                        Gromov-Wasserstein alignment by POT.
                                        Defaults to 1000.
            numItermax (int, optional): Maximum number of iterations for the
                                        Sinkhorn algorithm. Defaults to 1000.
            n_iter (int, optional):     Number of initial plans evaluated in single optimization.
                                        Defaults to 20.
            fix_random_init_seed (int, optional): Seed for generating the random initial matrix.
            gw_type (str, optional):    The type of Gromov-Wasserstein alignment.
                                        Options are "entropic_gromov_wasserstein",  "entropic_semirelaxed_gromov_wasserstein",  or "entropic_partial_gromov_wasserstein".
                                        Defaults to "entropic_gromov_wasserstein".
            sinkhorn_method (str, optional): The method used for Sinkhorn algorithm.
                                        Options are "sinkhorn", "sinkhorn_stabilized", or "sinkhorn_epsilon_scaling".
                                        Defaults to "sinkhorn".
            first_random_init_seed (int, optional): The first seed for generating the random initial matrix.
            tol (float, optional):      Stop threshold on error. Defaults to 1e-9.
            verbose (bool, optional):   Print information along iterations. Defaults to False.
            m (Optional[float], optional): The number of points to be used in partial Gromov-Wasserstein alignment.
                                        Defaults to None.
        """

        self.to_types = to_types
        self.data_type = data_type
        self.device = device
        self.instance_name = instance_name

        if p is None:
            p = ot.unif(len(source_dist))
        if q is None:
            q = ot.unif(len(target_dist))

        assert np.isclose(p.sum(), 1.0, atol=1e-8), "p must be a distribution."
        assert np.isclose(q.sum(), 1.0, atol=1e-8), "q must be a distribution."
        
        # init matrix
        self.init_mat_builder = InitMatrix(len(source_dist), len(target_dist), p, q)

        self.source_dist, self.target_dist, self.p, self.q = source_dist, target_dist, p, q
        
        self.back_end = Backend(device, to_types, data_type)
        
        self.ot_row_check = np.array([1.0/len(target_dist)] * len(target_dist))
        self.ot_col_check = np.array([1.0/len(source_dist)] * len(source_dist))

        # parameter for entropic gw alignment by POT
        self.max_iter = max_iter

        # parameter for sinkhorn
        self.numItermax = numItermax

        # the number of iteration mainly used for random init matrix.
        self.n_iter = n_iter
        
        # fix the seed for random init matrix.
        self.fix_random_init_seed = fix_random_init_seed
        
        if self.fix_random_init_seed is not None:
            if first_random_init_seed is not None:
                self.fix_seed = [i for i in range(first_random_init_seed, first_random_init_seed + fix_random_init_seed)]
            else:
                self.fix_seed = [i for i in range(fix_random_init_seed)]

        # sinkhorn method
        self.sinkhorn_method = sinkhorn_method
        
        # gw method
        self.gw_type = gw_type

        # parameters for gw alignment
        self.verbose = verbose        
        self.tol = tol
        
        if self.gw_type == "entropic_partial_gromov_wasserstein":
            self.tol = 1e-7
            self.m = m

    # main function
    def compute_GW_with_init_plans(
        self,
        trial: optuna.trial.Trial,
        eps: float,
        init_mat_plan: str,
    ) -> Tuple[dict, optuna.trial.Trial]:
        """Calculate Gromov-Wasserstein alignment with user-specified parameters.

        Args:
            trial (optuna.trial.Trial): 
                The trial object from the Optuna.
            eps (float): 
                Regularization term.
            init_mat_plan (str): 
                The initialization method of transportation plan for Gromov-Wasserstein alignment.
                Options are "random", "permutation", "user_define", "uniform", or "diag".

        Raises:
            optuna.TrialPruned: If all iterations failed with the given parameters.
            ValueError: If the initialization matrix method is not defined.

        Returns:
            best_logv (dict): The dictionary containing the Gromov-Wasserstein loss(distance) and accuracy.
            trial (optuna.trial.Trial): The trial object from the Optuna.
        """

        # define init matrices and seeds
        if init_mat_plan in ["uniform", "diag"]:
            seeds = None
            init_mat_list = [self.init_mat_builder.make_initial_T(init_mat_plan)]  # 1 initial matrix

        elif init_mat_plan in ["random", "permutation"]:
            if self.fix_random_init_seed is None:
                seeds = np.random.randint(0, 100000, self.n_iter)
            else:
                seeds = [self.fix_seed.pop(i) for i in range(self.n_iter)]
                
            init_mat_list = [self.init_mat_builder.make_initial_T(init_mat_plan, seed) for seed in seeds]  # n_iter initial matrices

        elif init_mat_plan == "user_define":
            seeds = None
            init_mat_list = self.init_mat_builder.user_define_init_mat_list

        else:
            raise ValueError("Not defined initialize matrix.")

        best_logv, trial = self._compute_GW_with_init_plans(
            trial,
            eps,
            init_mat_plan,
            init_mat_list,
            seeds=seeds
        )

        return best_logv, trial

    def _compute_GW_with_init_plans(
        self,
        trial: optuna.trial.Trial,
        eps: float,
        init_mat_plan: str,
        init_mat_list: List[Any],
        seeds: Optional[Iterable[int]] = None,
    ) -> Tuple[dict, optuna.trial.Trial, Optional[bool]]:
        """Computes the Gromov-Wasserstein alignment with the provided initial transportation plan.

        Args:
            trial (optuna.trial.Trial): The trial object from the Optuna.
            init_mat_plan (str): The initialization method of transportation plan for Gromov-Wasserstein alignment.
                                 Options are "random", "permutation", "user_define", "uniform", or "diag".
            eps (float): Regularization term.
            init_mat_list (List[Any]): The list of initial transportation plans.
            seed (Any, optional): Seed for generating the initial matrix. Defaults to None.

        Returns:
            logv (dict): The dictionary containing the Gromov-Wasserstein loss(distance) and accuracy.
            trial (optuna.trial.Trial): The trial object from the Optuna.
            best_flag (Optional[bool]): The flag indicating whether the current trial is the best trial.
        """

        # set pseudo seeds
        if seeds is None:
            seeds = np.zeros(len(init_mat_list))

        best_gw_loss = float("inf")

        pbar = tqdm(zip(init_mat_list, seeds), total=len(init_mat_list))
        pbar.set_description(f"{self.instance_name} No.{trial.number}, eps:{eps:.3e}")

        for i, (init_mat, seed) in enumerate(pbar):
            logv = self.gw_computation(eps, init_mat)

            if logv["gw_dist"] < best_gw_loss:
                best_gw_loss = logv["gw_dist"]
                best_logv = logv

                trial = self._save_results(
                    logv["gw_dist"],
                    logv["acc"],
                    logv["err"][-1],
                    trial,
                    init_mat_plan,
                    num_iter=i,
                    seed=seed,
                )
                
                elapsed_time = pbar.format_dict["elapsed"]
                trial.set_user_attr("elapsed_time", elapsed_time)

            self._check_pruner_should_work(
                logv["gw_dist"],
                trial,
                init_mat_plan,
                eps,
                num_iter=i,
            )
            

        if math.isinf(best_gw_loss) or best_gw_loss <= 0.0 or math.isnan(best_gw_loss):
            raise optuna.TrialPruned(
                f"All iteration was failed with parameters: {{'eps': {eps}, 'initialize': '{init_mat_plan}'}}"
            )

        return best_logv, trial

    def _save_results(
        self,
        gw_loss: float,
        acc: float,
        err: float, 
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
            err (float): The error after the while loop in main computation.
            trial (optuna.trial.Trial): The trial object from the Optuna.
            init_mat_plan (str):    The initialization method of transportation plan for Gromov-Wasserstein alignment.
                                    Options are "random", "permutation", "user_define", "uniform", or "diag".
            num_iter (int, optional): The number of optimizations within a single trial. Defaults to None.
            seed (int, optional): The seed used for random number generation in the trial. Defaults to None.

        Returns:
            trial (optuna.trial.Trial): The trial object from the Optuna.
        """

        gw_loss, acc, err = self.back_end.get_item_from_torch_or_jax(gw_loss, acc, err)

        trial.set_user_attr("best_acc", acc)
        trial.set_user_attr("error", err)
        
        if init_mat_plan in ["random", "permutation", "user_define"]:
            assert num_iter is not None, "num_iter must be provided for random, permutation and user_define initialization."
            trial.set_user_attr("best_iter", num_iter)

            if init_mat_plan in ["random", "permutation"]:
                assert seed is not None, "seed must be provided for random and permutation initialization."
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

    def gw_computation(self, eps: float, T: Any) -> Dict[str, Any]:
        """Performs the entropic Gromov-Wasserstein alignment.

        This function utilizes Sinkhorn's algorithm to iteratively solve the entropic Gromov-Wasserstein problem.
        The algorithm terminates when the error between iterations is less than the provided tolerance or
        when the maximum number of iterations is reached.

        Args:
            epsilon (float):    Regularization term for Gromov-Wasserstein alignment.
            T (Any):    Initial plan for Gromov-Wasserstein alignment.
            tol (float, optional):  Stop threshold on error. Defaults to 1e-9.

        Returns:
            log (Dict[str, Any]):  A dictionary containing the optimization results.
        """
        if self.gw_type == "entropic_gromov_wasserstein":
            logv = self.entropic_gw_computation(eps, T)

        elif self.gw_type == "entropic_semirelaxed_gromov_wasserstein":
            logv = self.entropic_semirelaxed_gw_computation(eps, T)

        elif self.gw_type == "entropic_partial_gromov_wasserstein":
            logv = self.entropic_partial_gw_computation(eps, T)

        else:
            raise ValueError(f"gw type {self.gw_type} is not defined.")

        return logv

    def entropic_gw_computation(self, eps: float, T: Any) -> Dict[str, Any]:
        """Performs the entropic Gromov-Wasserstein alignment.

        This function utilizes Sinkhorn's algorithm to iteratively solve the entropic Gromov-Wasserstein problem.
        The algorithm terminates when the error between iterations is less than the provided tolerance or
        when the maximum number of iterations is reached.

        Args:
            epsilon (float):    Regularization term for Gromov-Wasserstein alignment.
            T (Any):    Initial plan for Gromov-Wasserstein alignment.

        Returns:
            log (Dict[str, Any]):  A dictionary containing the optimization results.
        """
        
        # all the variable are placed on "device" here.
        C1, C2, p, q, T = self.back_end(
            self.source_dist, 
            self.target_dist, 
            self.p, 
            self.q, 
            T
        )
    
        constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun="square_loss")

        cpt = 0
        err = 1
        logv = {"err": []}

        while (err > self.tol and cpt < self.max_iter):
            Tprev = T
            tens = ot.gromov.gwggrad(constC, hC1, hC2, T)
            T = ot.bregman.sinkhorn(p, q, tens, eps, method=self.sinkhorn_method, numItermax=self.numItermax)

            if cpt % 10 == 0:
                err = self.back_end.nx.norm(T - Tprev)

                logv["err"].append(err)

                if self.verbose:
                    if cpt % 200 == 0:
                        print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                    print("{:5d}|{:8e}|".format(cpt, err))
            
            cpt += 1

        else:
            # if the while loop is not broken
            logv["gw_dist"] = ot.gromov.gwloss(constC, hC1, hC2, T)
            logv["ot"] = T
            logv["cpt"] = cpt
                        
            # original POT function
            if abs(self.back_end.nx.sum(T) - 1) > 1e-5:
                warnings.warn("Solver failed to produce a transport plan (checked by original POT). ")
                logv["gw_dist"] = float("nan")
                logv["acc"] = float("nan")
                return logv

            # additonal part   
            if self.back_end.check_zeros(logv["ot"]):
                logv["gw_dist"] = float("nan")
                logv["acc"] = float("nan")
                return logv

            else:
                pred = self.back_end.nx.argmax(logv["ot"], 1)
                correct = (pred == self.back_end.nx.arange(len(logv["ot"]), type_as=logv["ot"])).sum()
                logv["acc"] = correct / len(logv["ot"])
                return logv            
        

    def entropic_semirelaxed_gw_computation(self, eps: float, T: Any) -> Dict[str, Any]:
        """Performs the entropic Gromov-Wasserstein alignment.

        This function utilizes Sinkhorn's algorithm to iteratively solve the entropic Gromov-Wasserstein problem.
        The algorithm terminates when the error between iterations is less than the provided tolerance or
        when the maximum number of iterations is reached.

        Args:
            epsilon (float):    Regularization term for Gromov-Wasserstein alignment.
            T (Any):    Initial plan for Gromov-Wasserstein alignment.

        Returns:
            log (Dict[str, Any]):  A dictionary containing the optimization results.
        """

        # all the variable are placed on "device" here.
        C1, C2, p, q, T = self.back_end(self.source_dist, self.target_dist, self.p, self.q, T)
        constC, hC1, hC2, fC2t = ot.gromov.init_matrix_semirelaxed(C1, C2, p, loss_fun="square_loss")
        ones_p = self.back_end.nx.ones(p.shape, type_as=p)

        def df(G):
            qG = self.back_end.nx.sum(G, 0)
            marginal_product = self.back_end.nx.outer(ones_p, self.back_end.nx.dot(qG, fC2t))
            return ot.gromov.gwggrad(constC + marginal_product, hC1, hC2, G, self.back_end.nx)

        cpt = 0
        err = 1e15

        logv = {"err": []}

        while (err > self.tol and cpt < self.max_iter):
            Tprev = T
            # compute the kernel
            K = T * self.back_end.nx.exp(- df(T) / eps)
            scaling = p / self.back_end.nx.sum(K, 1)
            T = self.back_end.nx.reshape(scaling, (-1, 1)) * K
            if cpt % 10 == 0:
                err = self.back_end.nx.norm(T - Tprev)
                logv["err"].append(err)

                if self.verbose:
                    if cpt % 200 == 0:
                        print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                    print("{:5d}|{:8e}|".format(cpt, err))
            cpt += 1

        qT = self.back_end.nx.sum(T, 0)
        marginal_product = self.back_end.nx.outer(ones_p, self.back_end.nx.dot(qT, fC2t))

        logv['gw_dist'] = ot.gromov.gwloss(constC + marginal_product, hC1, hC2, T, self.back_end.nx)
        logv["ot"] = T
        logv["cpt"] = cpt

        # original part
        if self.back_end.check_zeros(logv["ot"]):
            logv["gw_dist"] = float("nan")
            logv["acc"] = float("nan")

        else:
            pred = self.back_end.nx.argmax(logv["ot"], 1)
            correct = (pred == self.back_end.nx.arange(len(logv["ot"]), type_as=logv["ot"])).sum()
            logv["acc"] = correct / len(logv["ot"])

        return logv

    def entropic_partial_gw_computation(self, eps: float, T: Any) -> Dict[str, Any]:
        """Performs the entropic partial Gromov-Wasserstein alignment.

        This function utilizes Sinkhorn's algorithm to iteratively solve the entropic Gromov-Wasserstein problem.
        The algorithm terminates when the error between iterations is less than the provided tolerance or
        when the maximum number of iterations is reached.

        Args:
            epsilon (float):    Regularization term for Gromov-Wasserstein alignment.
            T (Any):    Initial plan for Gromov-Wasserstein alignment.

        Returns:
            log (Dict[str, Any]):  A dictionary containing the optimization results.
        """

        # all the variable are placed on "device" here.
        C1, C2, p, q, T = self.back_end(self.source_dist, self.target_dist, self.p, self.q, T)

        if self.m is None:
            self.m = np.min((np.sum(p), np.sum(q)))

        elif self.m < 0:
            raise ValueError("Problem infeasible. Parameter m should be greater than 0.")
        
        elif self.m > np.min((np.sum(p), np.sum(q))):
            raise ValueError("Problem infeasible. Parameter m should lower or"
                            " equal than min(|a|_1, |b|_1).")

        cpt = 0
        err = 1

        logv = {"err": []}

        while (err > self.tol and cpt < self.max_iter):
            Tprev = T
            M_entr = ot.partial.gwgrad_partial(C1, C2, T)
            T = ot.partial.entropic_partial_wasserstein(p, q, M_entr, eps, self.m)

            if cpt % 10 == 0:  # to speed up the computations
                err =  self.back_end.nx.norm(T - Tprev)
                logv['err'].append(err)

                if self.verbose:
                    if cpt % 200 == 0:
                        print('{:5s}|{:12s}|{:12s}'.format(
                            'It.', 'Err', 'Loss') + '\n' + '-' * 31)
                    print('{:5d}|{:8e}|{:8e}'.format(cpt, err, ot.partial.gwloss_partial(C1, C2, T)))

            cpt += 1

        logv['gw_dist'] = ot.partial.gwloss_partial(C1, C2, T)
        logv["ot"] = T
        logv["cpt"] = cpt

        # original part
        if self.back_end.check_zeros(logv["ot"]):
            logv["gw_dist"] = float("nan")
            logv["acc"] = float("nan")

        else:
            pred = self.back_end.nx.argmax(logv["ot"], 1)
            correct = (pred == self.back_end.nx.arange(len(logv["ot"]), type_as=logv["ot"])).sum()
            logv["acc"] = correct / len(logv["ot"])

        return logv


# %%
