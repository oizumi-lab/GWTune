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
