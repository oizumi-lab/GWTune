#%%
import gc
import os
from typing import Any, List, Tuple, Optional, Dict, Iterable
import math
import warnings

import numpy as np
import optuna
import ot
from tqdm.auto import tqdm

# warnings.simplefilter("ignore")
from ..utils.backend import Backend
from ..utils.init_matrix import InitMatrix

# nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv



class GWComputation:
    """A class responsible for the specific computations of the entropic Gromov-Wasserstein alignment.

    This class manages the core computation processes for the Gromov-Wasserstein alignment. It deals with
    initializing the computations, performing the entropic Gromov-Wasserstein optimization, and saving
    the results of the optimization. The class provides a comprehensive suite of methods to manage and
    manipulate the optimization process.

    Attributes:
        to_types (str): Specifies the type of data structure to be used, either "torch" or "numpy".
        data_type (str): Specifies the type of data to be used in computation.
        source_dist (Any): Array-like, shape (n_source, n_source). Dissimilarity matrix of the source data.
        target_dist (Any): Array-like, shape (n_target, n_target). Dissimilarity matrix of the target data.
        p (array-like): Distribution over the source data.
        q (array-like): Distribution over the target data.
        source_size (int): Number of elements in the source distribution.
        target_size (int): Number of elements in the target distribution.
        init_mat_builder (InitMatrix): Builder object for creating initial transportation plans.
        max_iter (int): Maximum number of iterations for entropic Gromov-Wasserstein alignment by POT.
        numItermax (int): Maximum number of iterations for the Sinkhorn algorithm.
        n_iter (int): Number of initial plans evaluated in single optimization.
        back_end (Backend): Backend object responsible for handling device-specific operations.
        best_gw_loss (float, optional): Best Gromov-Wasserstein loss achieved during optimization.
                                        Only used for certain initialization methods.
    """

    def __init__(
        self,
        source_dist: Any,
        target_dist: Any,
        p: Optional[Any] = None,
        q: Optional[Any] = None,
        to_types: str = "torch",
        data_type: str = 'double',
        max_iter: int = 1000,
        numItermax: int = 1000,
        n_iter: int = 20,
        sinkhorn_method: str = "sinkhorn",
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
            n_iter (int, optional):     Number of initial plans evaluated in single optimization.
                                        Defaults to 20.
        """

        self.to_types = to_types
        self.data_type = data_type

        if p is None:
            p = ot.unif(len(source_dist))
        if q is None:
            q = ot.unif(len(target_dist))

        assert np.isclose(p.sum(), 1.0, atol=1e-8), "p must be a distribution."
        assert np.isclose(q.sum(), 1.0, atol=1e-8), "q must be a distribution."

        self.source_dist, self.target_dist, self.p, self.q = source_dist, target_dist, p, q

        self.source_size, self.target_size = len(source_dist), len(target_dist)

        # init matrix
        self.init_mat_builder = InitMatrix(self.source_size, self.target_size, self.p, self.q)

        # parameter for entropic gw alignment by POT
        self.max_iter = max_iter

        # parameter for sinkhorn
        self.numItermax = numItermax

        # the number of iteration mainly used for random init matrix.
        self.n_iter = n_iter

        # sinkhorn method
        self.sinkhorn_method = sinkhorn_method

        self.back_end = Backend("cpu", self.to_types, self.data_type)

    # main function
    def compute_GW_with_init_plans(
        self,
        trial: optuna.trial.Trial,
        eps: float,
        init_mat_plan: str,
        device: str,
    ) -> Tuple[dict, optuna.trial.Trial]:
        """Calculate Gromov-Wasserstein alignment with user-specified parameters.

        Args:
            trial (optuna.trial.Trial): The trial object from the Optuna.
            eps (float): Regularization term.
            init_mat_plan (str): The initialization method of transportation plan for Gromov-Wasserstein alignment.
                                    Options are "random", "permutation", "user_define", "uniform", or "diag".
            device (str): The device to be used for computation, either "cpu" or "cuda".
            sinkhorn_method (str, optional): Method used for the solver. Options are "sinkhorn", "sinkhorn_log", "greenkhorn",
                                                "sinkhorn_stabilized", or "sinkhorn_epsilon_scaling". Defaults to "sinkhorn".

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
            seeds = np.random.randint(0, 100000, self.n_iter)
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
            device,
            seeds=seeds
        )

        return best_logv, trial

    def _compute_GW_with_init_plans(
        self,
        trial: optuna.trial.Trial,
        eps: float,
        init_mat_plan: str,
        init_mat_list: List[Any],
        device: str,
        seeds: Optional[Iterable[int]] = None,
    ) -> Tuple[dict, optuna.trial.Trial, Optional[bool]]:
        """Computes the Gromov-Wasserstein alignment with the provided initial transportation plan.

        Args:
            trial (optuna.trial.Trial): The trial object from the Optuna.
            init_mat_plan (str): The initialization method of transportation plan for Gromov-Wasserstein alignment.
                                 Options are "random", "permutation", "user_define", "uniform", or "diag".
            eps (float): Regularization term.
            device (str): The device to be used for computation, either "cpu" or "cuda".
            sinkhorn_method (str): Method used for the solver. Options are "sinkhorn", "sinkhorn_log", "greenkhorn",
                                   "sinkhorn_stabilized", or "sinkhorn_epsilon_scaling".
            num_iter (int, optional): The number of optimizations within a single trial. Defaults to None.
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
        pbar.set_description(f"Trial No.{trial.number}, eps:{eps:.3e}")

        for i, (init_mat, seed) in enumerate(pbar):
            logv = self.gw_computation(
                device,
                eps,
                init_mat
            )

            if logv["gw_dist"] < best_gw_loss:
                best_gw_loss = logv["gw_dist"]
                best_logv = logv

                trial = self._save_results(
                    logv["gw_dist"],
                    logv["acc"],
                    trial,
                    init_mat_plan,
                    num_iter=i,
                    seed=seed,
                )

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

    def gw_computation(
        self,
        device: str,
        eps: float,
        T: Any,
        tol: float = 1e-9,
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
            verbose (bool, optional): Print information along iterations. Defaults to False.

        Returns:
            log (Dict[str, Any]):  A dictionary containing the optimization results.
        """
        pass

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


class EntropicGWComputation(GWComputation):
    def __init__(
        self,
        source_dist: Any,
        target_dist: Any,
        p: Optional[Any] = None,
        q: Optional[Any] = None,
        to_types: str = "torch",
        data_type: str = 'double',
        max_iter: int = 1000,
        numItermax: int = 1000,
        n_iter: int = 20,
        sinkhorn_method: str = "sinkhorn",
    ) -> None:
        super().__init__(
            source_dist,
            target_dist,
            p,
            q,
            to_types,
            data_type,
            max_iter,
            numItermax,
            n_iter,
            sinkhorn_method,
        )

    def gw_computation(
        self,
        device: str,
        eps: float,
        T: Any,
        tol: float = 1e-9,
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
        logv = {"err": []}

        while (err > tol and cpt < self.max_iter):
            Tprev = T
            tens = ot.gromov.gwggrad(constC, hC1, hC2, T)
            T = ot.bregman.sinkhorn(p, q, tens, eps, method=self.sinkhorn_method, numItermax=self.numItermax)

            if cpt % 10 == 0:
                err = self.back_end.nx.norm(T - Tprev)

                if logv:
                    logv["err"].append(err)

                if verbose:
                    if cpt % 200 == 0:
                        print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                    print("{:5d}|{:8e}|".format(cpt, err))
            cpt += 1

        if abs(self.back_end.nx.sum(T) - 1) > 1e-5:
            warnings.warn("Solver failed to produce a transport plan. You might "
                      "want to increase the regularization parameter `epsilon`.")

        logv["gw_dist"] = ot.gromov.gwloss(constC, hC1, hC2, T)
        logv["ot"] = T

        # original part
        if self.back_end.check_zeros(logv["ot"]):
            logv["gw_dist"] = float("nan")
            logv["acc"] = float("nan")

        else:
            pred = self.back_end.nx.argmax(logv["ot"], 1)
            correct = (pred == self.back_end.nx.arange(len(logv["ot"]), type_as=logv["ot"])).sum()
            logv["acc"] = correct / len(logv["ot"])

        return logv



class EntropicSemirelaxedGWComputation(GWComputation):
    def __init__(
        self,
        source_dist: Any,
        target_dist: Any,
        p: Optional[Any] = None,
        q: Optional[Any] = None,
        to_types: str = "torch",
        data_type: str = 'double',
        max_iter: int = 1000,
        numItermax: int = 1000,
        n_iter: int = 20,
        sinkhorn_method: str = "sinkhorn",
    ) -> None:
        super().__init__(
            source_dist,
            target_dist,
            p,
            q,
            to_types,
            data_type,
            max_iter,
            numItermax,
            n_iter,
            sinkhorn_method,
        )

    def gw_computation(
        self,
        device: str,
        eps: float,
        T: Any,
        tol: float = 1e-9,
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
            verbose (bool, optional): Print information along iterations. Defaults to False.

        Returns:
            log (Dict[str, Any]):  A dictionary containing the optimization results.
        """

        # all the variable are placed on "device" here.
        self.back_end.device = device
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

        while (err > tol and cpt < self.max_iter):
            Tprev = T
            # compute the kernel
            K = T * self.back_end.nx.exp(- df(T) / eps)
            scaling = p / self.back_end.nx.sum(K, 1)
            T = self.back_end.nx.reshape(scaling, (-1, 1)) * K
            if cpt % 10 == 0:
                err = self.back_end.nx.norm(T - Tprev)
                logv["err"].append(err)

                if verbose:
                    if cpt % 200 == 0:
                        print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                    print("{:5d}|{:8e}|".format(cpt, err))
            cpt += 1

        qT = self.back_end.nx.sum(T, 0)
        marginal_product = self.back_end.nx.outer(ones_p, self.back_end.nx.dot(qT, fC2t))

        logv['gw_dist'] = ot.gromov.gwloss(constC + marginal_product, hC1, hC2, T, self.back_end.nx)
        logv["ot"] = T

        # original part
        if self.back_end.check_zeros(logv["ot"]):
            logv["gw_dist"] = float("nan")
            logv["acc"] = float("nan")

        else:
            pred = self.back_end.nx.argmax(logv["ot"], 1)
            correct = (pred == self.back_end.nx.arange(len(logv["ot"]), type_as=logv["ot"])).sum()
            logv["acc"] = correct / len(logv["ot"])

        return logv



class EntropicPartialGWComputation(GWComputation):
    def __init__(
        self,
        source_dist: Any,
        target_dist: Any,
        p: Optional[Any] = None,
        q: Optional[Any] = None,
        to_types: str = "torch",
        data_type: str = 'double',
        max_iter: int = 1000,
        numItermax: int = 1000,
        n_iter: int = 20,
        sinkhorn_method: str = "sinkhorn",
        m: Optional[float]=None,
    ) -> None:
        super().__init__(
            source_dist,
            target_dist,
            p,
            q,
            to_types,
            data_type,
            max_iter,
            numItermax,
            n_iter,
            sinkhorn_method,
        )
        self.m = m


    def gw_computation(
        self,
        device: str,
        eps: float,
        T: Any,
        tol: float = 1e-7,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Performs the entropic partial Gromov-Wasserstein alignment.

        This function utilizes Sinkhorn's algorithm to iteratively solve the entropic Gromov-Wasserstein problem.
        The algorithm terminates when the error between iterations is less than the provided tolerance or
        when the maximum number of iterations is reached.

        Args:
            device (str):   The device to be used for computation, either "cpu" or "cuda".
            epsilon (float):    Regularization term for Gromov-Wasserstein alignment.
            T (Any):    Initial plan for Gromov-Wasserstein alignment.
            tol (float, optional):  Stop threshold on error. Defaults to 1e-9.
            verbose (bool, optional): Print information along iterations. Defaults to False.

        Returns:
            log (Dict[str, Any]):  A dictionary containing the optimization results.
        """

        # all the variable are placed on "device" here.
        self.back_end.device = device
        C1, C2, p, q, T = self.back_end(self.source_dist, self.target_dist, self.p, self.q, T)

        if self.m is None:
            self.m = np.min((np.sum(p), np.sum(q)))

        elif self.m < 0:
            raise ValueError("Problem infeasible. Parameter m should be greater"
                            " than 0.")
        elif self.m > np.min((np.sum(p), np.sum(q))):
            raise ValueError("Problem infeasible. Parameter m should lower or"
                            " equal than min(|a|_1, |b|_1).")

        cpt = 0
        err = 1

        logv = {"err": []}

        while (err > tol and cpt < self.max_iter):
            Tprev = T
            M_entr = ot.partial.gwgrad_partial(C1, C2, T)
            T = ot.partial.entropic_partial_wasserstein(p, q, M_entr, eps, self.m)

            if cpt % 10 == 0:  # to speed up the computations
                err =  self.back_end.nx.norm(T - Tprev)
                if logv:
                    logv['err'].append(err)

                if verbose:
                    if cpt % 200 == 0:
                        print('{:5s}|{:12s}|{:12s}'.format(
                            'It.', 'Err', 'Loss') + '\n' + '-' * 31)
                    print('{:5d}|{:8e}|{:8e}'.format(cpt, err, ot.partial.gwloss_partial(C1, C2, T)))

            cpt += 1

        logv['gw_dist'] = ot.partial.gwloss_partial(C1, C2, T)
        logv["ot"] = T

        # original part
        if self.back_end.check_zeros(logv["ot"]):
            logv["gw_dist"] = float("nan")
            logv["acc"] = float("nan")

        else:
            pred = self.back_end.nx.argmax(logv["ot"], 1)
            correct = (pred == self.back_end.nx.arange(len(logv["ot"]), type_as=logv["ot"])).sum()
            logv["acc"] = correct / len(logv["ot"])

        return logv




# class EntropicUnbalancedGWComputation(GWComputation):
#     def __init__(
#         self,
#         source_dist: Any,
#         target_dist: Any,
#         p: Optional[Any] = None,
#         q: Optional[Any] = None,
#         to_types: str = "torch",
#         data_type: str = 'double',
#         max_iter: int = 1000,
#         numItermax: int = 1000,
#         n_iter: int = 20,
#         sinkhorn_method: str = "sinkhorn",
#     ) -> None:
#         super().__init__(
#             source_dist,
#             target_dist,
#             p,
#             q,
#             to_types,
#             data_type,
#             max_iter,
#             numItermax,
#             n_iter,
#             sinkhorn_method,
#         )

#     def gw_computation(
#         self,
#         device: str,
#         eps: float,
#         T: Any,
#         tol: float = 1e-9,
#         verbose: bool = False
#     ) -> Dict[str, Any]:
#         couplingM = log_ugw_sinkhorn(torch.Tensor(p).cuda(),
#                                         torch.Tensor(Cx).cuda(),
#                                         torch.Tensor(q).cuda(),
#                                         torch.Tensor(Cy).cuda(),
#                                         eps=e,
#                                         rho=rho * 0.5 * (
#                                                 Cx.mean() + Cy.mean()),
#                                         init=T)
#         couplingM = couplingM.cpu().numpy()
#         log = None

#         """Performs the entropic Gromov-Wasserstein alignment.

#         This function utilizes Sinkhorn's algorithm to iteratively solve the entropic Gromov-Wasserstein problem.
#         The algorithm terminates when the error between iterations is less than the provided tolerance or
#         when the maximum number of iterations is reached.

#         Args:
#             device (str):   The device to be used for computation, either "cpu" or "cuda".
#             epsilon (float):    Regularization term for Gromov-Wasserstein alignment.
#             T (Any):    Initial plan for Gromov-Wasserstein alignment.
#             tol (float, optional):  Stop threshold on error. Defaults to 1e-9.
#             verbose (bool, optional): Print information along iterations. Defaults to False.

#         Returns:
#             log (Dict[str, Any]):  A dictionary containing the optimization results.
#         """

#         # all the variable are placed on "device" here.
#         self.back_end.device = device
#         C1, C2, p, q, T = self.back_end(self.source_dist, self.target_dist, self.p, self.q, T)
#         constC, hC1, hC2, fC2t = ot.gromov.init_matrix_semirelaxed(C1, C2, p, loss_fun="square_loss")
#         ones_p = self.back_end.nx.ones(p.shape, type_as=p)

#         def df(G):
#             qG = self.back_end.nx.sum(G, 0)
#             marginal_product = self.back_end.nx.outer(ones_p, self.back_end.nx.dot(qG, fC2t))
#             return ot.gromov.gwggrad(constC + marginal_product, hC1, hC2, G, self.back_end.nx)

#         cpt = 0
#         err = 1e15

#         logv = {"err": []}

#         while (err > tol and cpt < self.max_iter):
#             Tprev = T
#             # compute the kernel
#             K = T * self.back_end.nx.exp(- df(T) / eps)
#             scaling = p / self.back_end.nx.sum(K, 1)
#             T = self.back_end.nx.reshape(scaling, (-1, 1)) * K
#             if cpt % 10 == 0:
#                 err = self.back_end.nx.norm(T - Tprev)
#                 logv["err"].append(err)

#                 if verbose:
#                     if cpt % 200 == 0:
#                         print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
#                     print("{:5d}|{:8e}|".format(cpt, err))
#             cpt += 1

#         qT = self.back_end.nx.sum(T, 0)
#         marginal_product = self.back_end.nx.outer(ones_p, self.back_end.nx.dot(qT, fC2t))

#         logv['gw_dist'] = ot.gromov.gwloss(constC + marginal_product, hC1, hC2, T, self.back_end.nx)
#         logv["ot"] = T

#         # original part
#         if self.back_end.check_zeros(logv["ot"]):
#             logv["gw_dist"] = float("nan")
#             logv["acc"] = float("nan")

#         else:
#             pred = self.back_end.nx.argmax(logv["ot"], 1)
#             correct = (pred == self.back_end.nx.arange(len(logv["ot"]), type_as=logv["ot"])).sum()
#             logv["acc"] = correct / len(logv["ot"])

#         return logv

# def log_ugw_sinkhorn(a, dx, b, dy, init=None, eps=1.0,
#                      rho=float("Inf"), rho2=None,
#                      nits_plan=3000, tol_plan=1e-6,
#                      nits_sinkhorn=3000, tol_sinkhorn=1e-6,
#                      two_outputs=False):
#     """Solves the regularized UGW problem, keeps only one plan as output.
#     the algorithm is run as much as possible in log-scale.

#     Parameters
#     ----------
#     a: torch.Tensor of size [Batch, size_X]
#     Input measure of the first mm-space.

#     dx: torch.Tensor of size [Batch, size_X, size_X]
#     Input metric of the first mm-space.

#     b: torch.Tensor of size [Batch, size_Y]
#     Input measure of the second mm-space.

#     dy: torch.Tensor of size [Batch, size_Y, size_Y]
#     Input metric of the second mm-space.

#     init: torch.Tensor of size [Batch, size_X, size_Y]
#     Transport plan at initialization. Defaults to None and initializes
#     with tensor plan.

#     eps: float
#     Strength of entropic regularization.

#     rho: float
#     Strength of penalty on the first marginal of pi.

#     rho2: float
#     Strength of penalty on the first marginal of pi. If set to None it is
#     equal to rho.

#     nits_plan: int
#     Maximum number of iterations to update the plan pi.

#     tol_plan: float
#     Tolerance on convergence of plan.

#     nits_sinkhorn: int
#     Maximum number of iterations to update Sinkhorn potentials in inner loop.

#     tol_sinkhorn: float
#     Tolerance on convergence of Sinkhorn potentials.

#     two_outputs: bool
#     If set to True, outputs the two plans of the alternate minimization of UGW.

#     Returns
#     ----------
#     pi: torch.Tensor of size [Batch, size_X, size_Y]
#     Transport plan
#      which is a stationary point of UGW. The output is not
#     in log-scale.
#     """
#     if rho2 is None:
#         rho2 = rho

#     # Initialize plan and local cost
#     logpi = (init_plan(a, b, init=init) + 1e-30).log()
#     logpi_prev = torch.zeros_like(logpi)

#     up, vp = None, None
#     for i in range(nits_plan):
#         logpi_prev = logpi.clone()
#         lcost = compute_local_cost(logpi.exp(), a, dx, b, dy, eps, rho, rho2)
#         logmp = logpi.logsumexp(dim=(0, 1))
#         up, vp, logpi = log_sinkhorn(
#             lcost, up, vp, a, b, logmp.exp() + 1e-10, eps, rho, rho2,
#             nits_sinkhorn, tol_sinkhorn
#         )
#         if torch.any(torch.isnan(logpi)):
#             raise Exception(
#                 f"Solver got NaN plan with params (eps, rho, rho2) "
#                 f" = {eps, rho, rho2}. Try increasing argument eps."
#             )
#         logpi = (
#                 0.5 * (logmp - logpi.logsumexp(dim=(0, 1)))
#                 + logpi
#         )
#         if (logpi - logpi_prev).abs().max().item() < tol_plan:
#             break

#     if two_outputs:
#         return logpi.exp(), logpi_prev.exp()
#     return logpi.exp()
