# %%
import numpy as np
import copy
import itertools
import random
import optuna
import pymysql
from scipy.spatial import distance
from joblib import parallel_backend

import ot
from ot.bregman import sinkhorn
from ot.utils import dist, UndefinedParameter, list_to_array
from ot.optim import cg
from ot.lp import emd_1d, emd
from ot.utils import check_random_state, unif
from ot.backend import get_backend
from ot.gromov import init_matrix, gwggrad, gwloss

from src.utils.utils_functions import calc_correct_rate_same_dim, calc_correct_rate_top_n

# %%


def my_entropic_gromov_wasserstein(C1, C2, p, q, loss_fun, epsilon, T,
                                   max_iter=1000, tol=1e-9, verbose=False, log=False):
    """
    日報を書く欄にする
    2023/3/2
    ここまでやりました。(佐々木)

    """
    C1, C2, p, q = list_to_array(C1, C2, p, q)
    nx = get_backend(C1, C2, p, q)

    # add T as an input
    # T = nx.outer(p, q)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    cpt = 0
    err = 1

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Tprev = T

        # compute the gradient
        tens = gwggrad(constC, hC1, hC2, T)

        T = sinkhorn(p, q, tens, epsilon, method='sinkhorn', numItermax=1000)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        log['gw_dist'] = gwloss(constC, hC1, hC2, T)
        return T, log
    else:
        return T

# %%


def my_entropic_gromov_wasserstein2(C1, C2, p, q, loss_fun, epsilon, T,
                                    max_iter=1000, tol=1e-9, verbose=False, log=False):
    r"""
    Returns the gromov-wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`

    The function solves the following optimization problem:

    .. math::
        \mathbf{GW} = \mathop{\arg\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} - \epsilon(H(\mathbf{T}))

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices
    - `H`: entropy

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p :  array-like, shape (ns,)
        Distribution in the source space
    q :  array-like, shape (nt,)
        Distribution in the target space
    loss_fun :  string
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.

    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two spaces

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    C1, C2, p, q = list_to_array(C1, C2, p, q)
    nx = get_backend(C1, C2, p, q)

    # add T as an input
    # T = nx.outer(p, q)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    cpt = 0
    err = 1
    if log:
        log = {'err': []}

    err_count = 0
    min_err = 1
    patience = 10
    err_flg = True
    while (err > tol and cpt < max_iter and err_flg):

        Tprev = T

        # compute the gradient
        tens = gwggrad(constC, hC1, hC2, T)

        T = sinkhorn(p, q, tens, epsilon, method='sinkhorn', numItermax=1000)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err_prev = copy.deepcopy(err)
            err = nx.norm(T - Tprev)
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

            if err_prev <= err:
                err_count += 1
                if err_count > patience:
                    print('Early Stopping')
                    err_flg = False
            else:
                err_count = 0
        cpt += 1

    if log:
        log['gw_dist'] = gwloss(constC, hC1, hC2, T)
        return T, log
    else:
        return T

# %%


class Create_initial_plans():
    def __init__(self, n_init_plan, n_column):
        self.n_column = n_column
        self.n_init_plan = n_init_plan

    def rand_plans(self):
        rand_mat = np.random.rand(
            self.n_init_plan, self.n_column, self.n_column)
        row_sums = rand_mat.sum(axis=2)
        rand_mat /= row_sums[:, :, np.newaxis] * self.n_column
        return rand_mat

    def unif_plans(self):
        unif_mat = np.random.uniform(
            0, 1, (self.n_init_plan, self.n_column, self.n_column))
        row_sums = unif_mat.sum(axis=2)
        unif_mat /= row_sums[:, :, np.newaxis] * self.n_column
        return unif_mat

    def permutation_plans(self):
        """
        対角行列のランダムなpermutation
        """
        perm_vec_list = list(itertools.permutations(
            [i for i in range(self.n_column)]))
        n_perm = len(perm_vec_list)
        diag_mat = np.diag(np.ones((self.n_column,))/self.n_column)
        perm_mat = np.zeros((self.n_init_plan, self.n_column, self.n_column))
        rand_order = random.sample([i for i in range(n_perm)], n_perm)
        for i in range(self.n_init_plan):
            perm_mat[i] = diag_mat[:, perm_vec_list[rand_order[i]]]
        return perm_mat

    def shifted_plans(self, n_shift):
        diag_mat = np.diag(np.ones((self.n_column,))/self.n_column)
        shifted_mat = diag_mat[:, [k - n_shift for k in range(self.n_column)]]
        return shifted_mat

# %%


class GW_alignment():
    def __init__(self, RDM_1, RDM_2, n_init_plan, epsilons=None, epsilon_range=None, DATABASE_URL=None, study_name=None, init_diag=True):
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
            'epsilon', self.epsilon_range[0], self.epsilon_range[1])
        create_init_plans = Create_initial_plans(
            self.n_init_plan, self.RDM_1.shape[0])
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
                    C1=self.RDM_1, C2=self.RDM_2, p=p, q=q, T=T, epsilon=epsilon, loss_fun="square_loss", verbose=False, log=True)
                gwd = log['gw_dist']
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
            study_name=self.study_name, storage=self.DATABASE_URL, load_if_exists=True)
        # Reset study
        optuna.delete_study(study_name=self.study_name,
                            storage=self.DATABASE_URL)
        # Create study once again
        study = optuna.create_study(
            study_name=self.study_name, storage=self.DATABASE_URL, load_if_exists=True)

        if parallel == False:
            study.optimize(self._GW_1to1, n_trials=n_trials, n_jobs=n_jobs)

        # make parallel
        else:
            # Overrides `prefer="threads"` to use multi-processing.
            with parallel_backend('multiprocessing'):
                study.optimize(self._GW_1to1, n_trials=n_trials, n_jobs=n_jobs)

        study = optuna.load_study(
            study_name=self.study_name, storage=self.DATABASE_URL)
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
                self.n_init_plan, self.RDM_1.shape[0])
            init_plans = create_init_plans.rand_plans()
            if self.init_diag:
                init_plans[0] = create_init_plans.shifted_plans(0)
            for i in range(self.n_init_plan):
                T = init_plans[i]
                p, q = np.sum(T, axis=1), np.sum(T, axis=0)
                gw, log = my_entropic_gromov_wasserstein(
                    C1=self.RDM_1, C2=self.RDM_2, p=p, q=q, T=T, epsilon=epsilon, loss_fun="square_loss", verbose=False, log=True)
                gwd = log['gw_dist']
                if gwd == 0:
                    gwd == 1e+5
                gws.append(gw)
                gwds.append(gwd)
        idx = np.argmin(gwds)
        gw = gws[idx]
        gwd = gwds[idx]
        min_epsilon = self.epsilons[idx//self.n_init_plan]
        correct_rate = calc_correct_rate_same_dim(gw)

        return gw, gwd, min_epsilon, correct_rate


# %%
