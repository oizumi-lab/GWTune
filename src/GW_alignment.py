# %%
# Standard Library
import copy
import itertools
import os
import random
import sys
from copy import copy

# Third Party Library
import matplotlib.pyplot as plt
import numpy as np
import optuna
import ot
import pymysql
import seaborn as sns
from joblib import parallel_backend
from ot.backend import get_backend
from ot.bregman import sinkhorn
from ot.gromov import gwggrad, gwloss, init_matrix
from ot.lp import emd, emd_1d
from ot.optim import cg
from ot.utils import UndefinedParameter, check_random_state, dist, list_to_array, unif
from scipy.spatial import distance

sys.path.append("../")

# First Party Library
# from src.utils.utils_functions import (
#     calc_correct_rate_same_dim,
#     calc_correct_rate_top_n,
# )
# from src.utils.gw_functions import initialize_matrix
from processing.gw_funcs import initialize_matrix

# %%


def my_entropic_gromov_wasserstein(
    C1,
    C2,
    p,
    q,
    loss_fun,
    epsilon,
    T,
    max_iter=1000,
    tol=1e-9,
    verbose=False,
    log=False,
    stopping_rounds=None,
    trial=None,
):
    """
    日報を書く欄にする
    2023/3/2
    ここまでやりました。(佐々木)

    2023/3/3
    Early Stoppingの実装
    - 引数にstopping_roundsの追加

    """
    C1, C2, p, q = list_to_array(C1, C2, p, q)
    nx = get_backend(C1, C2, p, q)

    # add T as an input
    # T = nx.outer(p, q)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    cpt = 0
    err = 1
    err_count = 0
    err_flg = True

    if log:
        log = {"err": []}

    while err > tol and cpt < max_iter and err_flg:

        Tprev = T

        # compute the gradient
        tens = gwggrad(constC, hC1, hC2, T)

        T = sinkhorn(p, q, tens, epsilon, method="sinkhorn", numItermax=1000)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err_prev = copy(err)
            err = nx.norm(T - Tprev)

            if log:
                log["err"].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(cpt, err))

            if trial:
                trial.report(gwloss(constC, hC1, hC2, T), cpt // 10)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if stopping_rounds:
                if (err_prev <= err) or (np.abs(err_prev - err) <= 1e-10):
                    err_count += 1
                else:
                    err_count = 0

                if stopping_rounds < err_count:
                    if verbose:
                        print("Early stopping")
                    err_flg = False
        cpt += 1

    if log:
        log["gw_dist"] = gwloss(constC, hC1, hC2, T)
        return T, log
    else:
        return T


#%%
def gw_alignment(X, Y, epsilon, random_init=False, stopping_rounds=5):
    """
    2023/3/3 初期値の決め方を修正(阿部)
    """
    n = X.shape[0]
    if random_init:
        T = initialize_matrix(n)
        p, q = ot.unif(n), ot.unif(n)
        gw, log = my_entropic_gromov_wasserstein(
            C1=X,
            C2=Y,
            p=p,
            q=q,
            T=T,
            epsilon=epsilon,
            loss_fun="square_loss",
            verbose=True,
            log=True,
            stopping_rounds=5,
        )
    else:
        p = ot.unif(n)
        q = ot.unif(n)
        gw, log = ot.gromov.entropic_gromov_wasserstein(
            X, Y, p, q, "square_loss", epsilon=epsilon, log=True, verbose=True
        )

    plt.figure(figsize=(5, 5))
    sns.heatmap(gw, square=True)
    plt.show()
    return gw, log


# %%


class Create_initial_plans:
    def __init__(self, n_init_plan, n_column):
        self.n_column = n_column
        self.n_init_plan = n_init_plan

    def rand_plans(self):
        rand_mat = np.random.rand(self.n_init_plan, self.n_column, self.n_column)
        row_sums = rand_mat.sum(axis=2)
        rand_mat /= row_sums[:, :, np.newaxis] * self.n_column
        return rand_mat

    def unif_plans(self):
        unif_mat = np.random.uniform(
            0, 1, (self.n_init_plan, self.n_column, self.n_column)
        )
        row_sums = unif_mat.sum(axis=2)
        unif_mat /= row_sums[:, :, np.newaxis] * self.n_column
        return unif_mat

    def permutation_plans(self):
        """
        対角行列のランダムなpermutation
        """
        perm_vec_list = list(itertools.permutations([i for i in range(self.n_column)]))
        n_perm = len(perm_vec_list)
        diag_mat = np.diag(np.ones((self.n_column,)) / self.n_column)
        perm_mat = np.zeros((self.n_init_plan, self.n_column, self.n_column))
        rand_order = random.sample([i for i in range(n_perm)], n_perm)
        for i in range(self.n_init_plan):
            perm_mat[i] = diag_mat[:, perm_vec_list[rand_order[i]]]
        return perm_mat

    def shifted_plans(self, n_shift):
        diag_mat = np.diag(np.ones((self.n_column,)) / self.n_column)
        shifted_mat = diag_mat[:, [k - n_shift for k in range(self.n_column)]]
        return shifted_mat


# %%


class GW_alignment:
    def __init__(
        self,
        RDM_1,
        RDM_2,
        n_init_plan,
        epsilons=None,
        epsilon_range=None,
        DATABASE_URL=None,
        study_name=None,
        init_diag=True,
    ):
        """GW alignment

        Args:
            RDM_1 (array-like): Representational simmilarity matrix
            RDM_2 (array-like): Representational simmilarity matrix
            n_init_plan : the number of initial plans
            epsilons : list of epsilons. It is needed when you don't use optuna
            epsilon_range (list): search range of epsilons
            DATABASE_URL (string) : SQLサーバーのディレクトリ
            study_name (string) : optuna.studyのラベル
            init_diag (bool, optional): If True, initial plans contain the diagonal plan for each epsilons. Defaults to True.

        Returns:
            Pi_list (array-like): List of transportation matrices
        """
        self.RDM_1 = RDM_1
        self.RDM_2 = RDM_2
        self.epsilon_range = epsilon_range
        self.epsilons = epsilons
        self.n_init_plan = n_init_plan
        self.init_diag = init_diag
        self.study_name = study_name
        self.DATABASE_URL = DATABASE_URL

        self.min_gw = dict()
        self.min_gwd = dict()

        pass

    def _GW_1to1(self, trial):
        epsilon = trial.suggest_float(
            "epsilon", self.epsilon_range[0], self.epsilon_range[1]
        )
        create_init_plans = Create_initial_plans(self.n_init_plan, self.RDM_1.shape[0])
        init_plans = create_init_plans.rand_plans()
        if self.init_diag:
            init_plans[0] = create_init_plans.shifted_plans(0)
        gwd = 0
        n_iter = 0
        # エラーの場合はイテレート(最大3回)
        while gwd == 0 and n_iter < 3:
            gws = list()
            gwds = list()
            for i in range(self.n_init_plan):
                T = init_plans[i]
                p, q = np.sum(T, axis=1), np.sum(T, axis=0)
                gw, log = my_entropic_gromov_wasserstein(
                    C1=self.RDM_1,
                    C2=self.RDM_2,
                    p=p,
                    q=q,
                    T=T,
                    epsilon=epsilon,
                    loss_fun="square_loss",
                    verbose=False,
                    log=True,
                )
                gwd = log["gw_dist"]
                gws.append(gw)
                gwds.append(gwd)
            idx = np.argmin(gwds)
            gw = gws[idx]
            gwd = gwds[idx]
            n_iter += 1
            if n_iter == 3:
                gwd = np.nan

        self.min_gw[f"{trial.number}"] = gw
        self.min_gwd[f"{trial.number}"] = gwd

        return gwd

    def optimize(self, n_trials, n_jobs, parallel=True):
        """最適化計算

        Args:
            n_trials (int): trialの数
            n_jobs (int): 使うコアの数
            parallel (bool, optional): 並列化を行うかどうか. Defaults to True.

        Returns:
            min_gw : transportation matrix
            min_gwd : minimum gw distance
            min_epsilon : 最適なepsilon
            correct_rate : 最適化された時の正解率
        """
        # Create study
        study = optuna.create_study(
            study_name=self.study_name, storage=self.DATABASE_URL, load_if_exists=True
        )
        # Reset study
        optuna.delete_study(study_name=self.study_name, storage=self.DATABASE_URL)
        # Create study once again
        study = optuna.create_study(
            study_name=self.study_name, storage=self.DATABASE_URL, load_if_exists=True
        )

        if parallel == False:
            study.optimize(self._GW_1to1, n_trials=n_trials, n_jobs=n_jobs)

        # make parallel
        else:
            # Overrides `prefer="threads"` to use multi-processing.
            with parallel_backend("multiprocessing"):
                study.optimize(self._GW_1to1, n_trials=n_trials, n_jobs=n_jobs)

        study = optuna.load_study(study_name=self.study_name, storage=self.DATABASE_URL)
        best_trial = study.best_trial

        min_gw = self.min_gw[f"{best_trial.number}"]
        min_gwd = self.min_gwd[f"{best_trial.number}"]
        min_epsilon = min_epsilon = best_trial.params["epsilon"]
        correct_rate = calc_correct_rate_same_dim(min_gw)

        return min_gw, min_gwd, min_epsilon, correct_rate

    def GW_for_epsilons_and_init_plans(self):
        """
        optimization without optuna
        """
        gws = list()
        gwds = list()
        for epsilon in self.epsilons:
            create_init_plans = Create_initial_plans(
                self.n_init_plan, self.RDM_1.shape[0]
            )
            init_plans = create_init_plans.rand_plans()
            if self.init_diag:
                init_plans[0] = create_init_plans.shifted_plans(0)
            for i in range(self.n_init_plan):
                T = init_plans[i]
                p, q = np.sum(T, axis=1), np.sum(T, axis=0)
                gw, log = my_entropic_gromov_wasserstein(
                    C1=self.RDM_1,
                    C2=self.RDM_2,
                    p=p,
                    q=q,
                    T=T,
                    epsilon=epsilon,
                    loss_fun="square_loss",
                    verbose=False,
                    log=True,
                )
                gwd = log["gw_dist"]
                if gwd == 0:
                    gwd == 1e5
                gws.append(gw)
                gwds.append(gwd)
        idx = np.argmin(gwds)
        gw = gws[idx]
        gwd = gwds[idx]
        min_epsilon = self.epsilons[idx // self.n_init_plan]
        correct_rate = calc_correct_rate_same_dim(gw)

        return gw, gwd, min_epsilon, correct_rate


# myentropic の1を変更
# %%
class Entropic_GW:
    def __init__(self, X1, X2, save_path):
        self.X1 = X1
        self.X2 = X2
        self.save_path = save_path

        # gw alignmentに関わるparameter
        self.max_iter = 1000
        self.stopping_rounds = None
        self.n_iter = 1
        self.punishment = float("nan")

        # hyperparameter
        self.epsilon = (1.0e-5, 1.0e-3)
        self.initialize = ["random", "permutation", "uniform"]

        # optuna parameter
        self.min_resource = 3
        self.max_resource = (self.max_iter // 10) * self.n_iter
        self.reduction_factor = 3

    def setting(self, params, hyperparams, optuna_params):
        for p in [params, hyperparams, optuna_params]:
            for key, value in p.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def gw_alignment(
        self,
        T,
        epsilon,
        trial,
        max_iter=300,
        stopping_rounds=5,
        verbose=False,
        log=True,
    ):
        n = self.X1.shape[0]
        p, q = ot.unif(n), ot.unif(n)
        gw, log = my_entropic_gromov_wasserstein(
            C1=self.X1,
            C2=self.X2,
            p=p,
            q=q,
            T=T,
            max_iter=max_iter,
            epsilon=epsilon,
            loss_fun="square_loss",
            verbose=verbose,
            log=log,
            trial=trial,
            stopping_rounds=stopping_rounds,
        )
        return gw, log

    def make_initial_T(self, initialize):
        n = self.X1.shape[0]
        if initialize == "random":
            T = initialize_matrix(n)
        elif initialize == "permutation":
            ts = np.zeros(n)
            ts[0] = 1 / n
            T = initialize_matrix(n, ts=ts)
        elif initialize == "beta":
            ts = np.random.beta(2, 5, n)
            ts = ts / (n * np.sum(ts))
            T = initialize_matrix(n, ts=ts)
        elif initialize == "uniform":
            T = np.outer(ot.unif(n), ot.unif(n))
        return T

    def __call__(self, trial):
        # hyperparameter
        ep_lower, ep_upper = self.epsilon
        epsilon = trial.suggest_float("epsilon", ep_lower, ep_upper)
        # initialize = 'uniform'
        initialize = trial.suggest_categorical("initialize", self.initialize)

        Ts, gws, gwds = [], [], []

        for _ in range(self.n_iter):
            # make initial T
            T = self.make_initial_T(initialize)
            gw, log = self.gw_alignment(
                T, epsilon, trial, self.max_iter, stopping_rounds=self.stopping_rounds
            )

            gwd = log["gw_dist"]
            if (gwd == 0.0) or (np.sum(gw) < 0.5):
                continue
            Ts.append(T)
            gws.append(gw)
            gwds.append(gwd)

        if len(gwds) == 0:
            raise optuna.TrialPruned()
            return self.punishment
            # return float('nan')  # make trial failure
        # 最小のgwdを発見する
        idx = np.argmin(gwds)
        T, gw, gwd = Ts[idx], gws[idx], gwds[idx]

        # データの保存
        np.save(self.save_path + "/T/{}_T".format(trial.number), T)
        np.save(self.save_path + "/gw/{}_gw".format(trial.number), gw)
        np.save(self.save_path + "/gwd/{}_gwd".format(trial.number), gwd)

        return gwd

    def run_study_sqlite(self, filename, study_name, concurrency=20, num_trial=1000):
        save_file_name = filename + ".db"
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            storage="sqlite:///" + self.path + "/" + save_file_name,
            load_if_exists=True,
        )
        # max_cpu count
        max_cpu = os.cpu_count()
        if not concurrency <= max_cpu:
            raise ValueError("concurrency > max_cpu")
        # 並列化
        # study.optimize(self, n_trials=num_trial, n_jobs=concurrency)
        with parallel_backend("multiprocessing", n_jobs=concurrency):
            study.optimize(self, n_trials=num_trial)

    def run_study_mysql(self, database_uri, study_name, concurrency=20, num_trial=1000):
        study = optuna.create_study(
            direction="minimize",
            storage=database_uri,
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=self.min_resource,
                max_resource=self.max_resource,
                reduction_factor=self.reduction_factor,
            ),
            load_if_exists=True,
        )

        # max_cpu count
        max_cpu = os.cpu_count()
        if not concurrency <= max_cpu:
            raise ValueError("concurrency > max_cpu")

        # 並列化
        with parallel_backend("multiprocessing", n_jobs=concurrency):
            study.optimize(self, n_trials=num_trial)

        self.study = study


# %%
