# %%
import os, sys, gc
import jax.numpy as jnp
import jax
import numpy as np
import pandas as pd
import torch
import ot
import matplotlib.pyplot as plt
import optuna
from joblib import parallel_backend
from copy import copy
import warnings
# warnings.simplefilter("ignore")
import os


# %%
from src.utils.backend import Backend

# %%
class GW_Alignment():
    def __init__(self, pred_dist, target_dist, p, q, device='cpu', to_types='torch', speed_test=False, gpu_queue = None):
        """
        2023/3/6 大泉先生

        1. epsilonに関して
        epsilon: １つ
        epsilonの範囲を決める：サーチ方法 optuna, 単純なgrid (samplerの種類, optuna)

        2. 初期値に関して
        初期値1つ固定: diagonal, uniform outer(p,q), 乱数
        初期値ランダムで複数: 乱数
        """

        self.device = device
        self.to_types = to_types
        self.gpu_queue = gpu_queue

        be = Backend(self.device, self.to_types) # potのnxに書き換えるべき。
        self.pred_dist, self.target_dist, self.p, self.q = be.change_data(pred_dist, target_dist, p, q)

        self.size = len(self.pred_dist)

        self.speed_test = speed_test

        self.save_path = '../data/gw_alignment'

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # hyperparameter
        self.epsilon = (1e-4, 1e-2)
        self.initialize = ['uniform', 'random', 'permutation', 'diag']
        self.init_mat_builder = InitMatrixForGW_Alignment(self.size)

        '''
        以下、阿部さんが使う各種設定。NumpyでGW計算に使うものと思われる。
        '''
        # gw alignmentに関わるparameter
        self.max_iter = 1000
        self.stopping_rounds = None
        self.n_iter = 1
        self.punishment = float("nan")

        # optuna parameter
        self.min_resource = 3
        self.max_resource = (self.max_iter // 10) * self.n_iter
        self.reduction_factor = 3

    def entropic_GW(self, device, epsilon, T = None, log = True, verbose = False):
        max_iter = 1000
        tol = 1e-9

        C1, C2, p, q = self.pred_dist.to(device), self.target_dist.to(device), self.p.to(device), self.q.to(device)

        nx = ot.backend.get_backend(C1, C2, p, q)

        # add T as an input
        if T is None:
            T = nx.outer(p, q)

        constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun = "square_loss")
        cpt = 0
        err = 1

        if log:
            log = {'err': []}

        while (err > tol and cpt < max_iter):
            Tprev = T
            # compute the gradient
            tens = ot.gromov.gwggrad(constC, hC1, hC2, T)
            T = ot.bregman.sinkhorn(p, q, tens, epsilon, method = 'sinkhorn')

            if cpt % 10 == 0:
                # we can speed up the process by checking for the error only all the 10th iterations
                err = nx.norm(T - Tprev)
                if log:
                    log['err'].append(err)
                if verbose:
                    if cpt % 200 == 0:
                        print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(cpt, err))
            cpt += 1

        if log:
            log['gw_dist'] = ot.gromov.gwloss(constC, hC1, hC2, T)
            return T, log

        else:
            return T

    def _choose_init_plans(self, init_plans_list):
        """
        ここから、初期値の条件を1個または複数個選択することができる。
        選択はself.initializeの中にあるものの中から。
        選択したい条件が1つであっても、リストで入力をすること。

        Args:
            init_plans_list (list) : 初期値の条件を1個または複数個入れたリスト。

        Raises:
            ValueError: 選択したい条件が1つであっても、リストで入力をすること。

        Returns:
            list : 選択希望の条件のリスト。
        """

        if type(init_plans_list) != list:
            raise ValueError('variable named "init_plans_list" is not list!')

        else:
            return [v for v in self.initialize if v in init_plans_list]

    def __call__(self, trial, init_plans_list):

        '''
        0.  define the "gpu_queue" here. This will be used when the memory of dataset was too much large for a single GPU board, and so on.
        '''

        if self.gpu_queue is None:
            device = self.device

        else:
            gpu_id = self.gpu_queue.get()
            device = 'cuda:' + str(gpu_id)



        '''
        1.  define hyperparameter (eps, T)
        '''
        ep_lower, ep_upper = self.epsilon
        eps = trial.suggest_float("eps", ep_lower, ep_upper, log = True)

        seed = trial.suggest_int("seed", 0, 9, 1)

        init_mat_types = self._choose_init_plans(init_plans_list) # リストを入力して、実行可能な方法のみをリストにして返す。
        init_mat_plan = trial.suggest_categorical("initialize", init_mat_types) # 上記のリストから、1つの方法を取り出す(optunaがうまく選択してくれる)。
        init_mat = self.init_mat_builder.make_initial_T(init_mat_plan) # epsの値全部を計算する際、randomは何回も計算していいけど、diag, uniformは一回だけでいいので、うまく切り分けよう。
        init_mat = torch.from_numpy(init_mat).float().to(device)

        trial.set_user_attr('size', self.size)

        '''
        2.  Compute GW alignment with hyperparameters defined above.
        '''

        gw, logv = self.entropic_GW(device, eps, T = init_mat)


        '''
        3.  count the accuracy of alignment and save the results if computation was finished in the right way.
            If not, set the result of accuracy and gw_loss as float('nan'), respectively. This will be used as a handy marker as bad results to be removed in the evaluation analysis.

        '''

        if torch.count_nonzero(gw).item() != 0:
            gw_loss = logv['gw_dist'].item()

            _, pred = torch.max(gw, 1)
            acc = pred.eq(torch.arange(len(gw)).to(device)).sum() / len(gw)

            torch.save(gw, self.save_path + '/GW({} pictures, epsilon={}, trial = {}).pt'.format(self.size, round(eps, 6, trial.number)))

        else:
            gw_loss = float('nan')
            acc = float('nan')

        '''
        4. delete unnecessary memory for next computation. If not, memory error would happen especially when using CUDA.
        '''

        del gw, logv
        torch.cuda.empty_cache()
        gc.collect()

        '''
        5.  "gpu_queue" can manage the GPU boards for the next computation.
        '''
        if self.gpu_queue is not None:
            self.gpu_queue.put(gpu_id)

        return gw_loss, acc


# %%
class InitMatrixForGW_Alignment():
    def __init__(self, matrix_size):
        self.matrix_size = matrix_size
        pass

    def make_initial_T(self, initialize):
        """
        To do : 説明を書く
        """
        # np.random.seed(seed)
        if initialize == 'random':
            T = self.make_random_initplan()
            return T

        elif initialize == 'permutation':
            ts = np.zeros(self.matrix_size)
            ts[0] = 1 / self.matrix_size
            T = self.initialize_matrix(self.matrix_size, ts=ts)
            return T

        elif initialize == 'any_permutation':
            T = self.initialize_matrix(self.matrix_size)
            return T

        elif initialize == 'beta':
            ts = np.random.beta(2, 5, self.matrix_size)
            ts = ts / (self.matrix_size * np.sum(ts))
            T = self.initialize_matrix(self.matrix_size, ts=ts)
            return T

        elif initialize == 'uniform':
            T = np.outer(ot.unif(self.matrix_size), ot.unif(self.matrix_size))
            return T

        elif initialize == 'diag':
            T = np.diag(np.ones(self.matrix_size) / self.matrix_size)
            return T

        else:
            raise ValueError('Not defined initialize matrix.')

    def randOrderedMatrix(self):
        """
        各行・各列に重複なしに[0,n]のindexを持つmatrixを作成

        Parameters
            n : int 行列のサイズ
        Returns
            np.ndarray 重複なしのindexを要素に持つmatrix
        """
        matrix = np.zeros((self.matrix_size, self.matrix_size))
        rows = np.tile(np.arange(0, self.matrix_size), 2)

        for i in range(self.matrix_size):
            matrix[i, :] = rows[i : i + self.matrix_size]

        r = np.random.choice(self.matrix_size, self.matrix_size, replace=False)
        c = np.random.choice(self.matrix_size, self.matrix_size, replace=False)
        matrix = matrix[r, :]
        matrix = matrix[:, c]
        return matrix.astype(int)

    def initialize_matrix(self, ts=None):
        """
        gw alignmentのための行列初期化

        Parameters
            n : int 行列のサイズ
        Returns
            np.ndarray 初期値
        """
        matrix = self.randOrderedMatrix(self.matrix_size)
        if ts is None:
            ts = np.random.uniform(0, 1, self.matrix_size)
            ts = ts / (self.matrix_size * np.sum(ts))

        T = np.array([ts[idx] for idx in matrix])
        return T

    def make_random_initplan(self):
        """
        大泉先生が作ったもの。
        """
        # make random initial transportation plan (N x N matrix)
        T = np.random.rand(self.matrix_size, self.matrix_size) # create a random matrix of size n x n
        rep = 100 # number of repetitions
        for _ in range(rep):
            # normalize each row so that the sum is 1
            p = T.sum(axis=1, keepdims=True)
            T = T / p
            # normalize each column so that the sum is 1
            q = T.sum(axis=0, keepdims=True)
            T = T / q
        T = T / self.matrix_size
        return T
