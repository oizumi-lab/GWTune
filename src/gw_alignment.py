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
sys.path.append("../")

from utils.utils import Backend

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
        
        be = Backend(self.device, self.to_types)
        self.pred_dist, self.target_dist, self.p, self.q = be.change_data(pred_dist, target_dist, p, q)
        
        self.size = len(self.pred_dist)
        
        self.speed_test = speed_test
        
        self.save_path = '../data/gw_alignment'
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
        # gw alignmentに関わるparameter
        self.max_iter = 1000
        self.stopping_rounds = None
        self.n_iter = 1
        self.punishment = float("nan")

        # hyperparameter
        self.epsilon = (1e-4, 1e-2)
        self.initialize = ["uniform"] #"random", "permutation",

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

    def __call__(self, trial):
        
        if self.gpu_queue is None:
            device = self.device
        
        else:
            gpu_id = self.gpu_queue.get()
            device = 'cuda:' + str(gpu_id) 
        
        
        ep_lower, ep_upper = self.epsilon
        eps = trial.suggest_float("eps", ep_lower, ep_upper, log = True)
        
        initialize = trial.suggest_categorical("initialize", self.initialize)
        
        # size = trial.

        init_plans = Init_Matrix_For_GW_Alignment(self.size)
        init_mat = init_plans.make_initial_T(initialize)

        init_mat = torch.from_numpy(init_mat).float().to(device)
        
        gw, logv = self.entropic_GW(device, eps, T = init_mat)
        
        if torch.count_nonzero(gw).item() != 0:
            gw_loss = logv['gw_dist'].item()
            
            _, pred = torch.max(gw, 1)
            acc = pred.eq(torch.arange(len(gw)).to(device)).sum() / len(gw)

            torch.save(gw, self.save_path + '/GW({} pictures, epsilon={}).pt'.format(gw.shape[0], round(eps, 6)))
            
        else:
            gw_loss = 1e6
            acc = -1
        
        del gw, logv
        torch.cuda.empty_cache()
        gc.collect()
        
        if self.gpu_queue is not None:
            self.gpu_queue.put(gpu_id)

        return gw_loss, acc


# %%
class Init_Matrix_For_GW_Alignment():
    def __init__(self, matrix_size):
        self.matrix_size = matrix_size
        pass

    def make_initial_T(self, initialize):
        
        if initialize == "random":
            T = self.initialize_matrix(self.matrix_size)
            return T
        
        elif initialize == "permutation":
            ts = np.zeros(self.matrix_size)
            ts[0] = 1 / self.matrix_size
            T = self.initialize_matrix(self.matrix_size, ts=ts)
            return T
        
        elif initialize == "beta":
            ts = np.random.beta(2, 5, self.matrix_size)
            ts = ts / (self.matrix_size * np.sum(ts))
            T = self.initialize_matrix(self.matrix_size, ts=ts)
            return T
        
        elif initialize == "uniform":
            T = np.outer(ot.unif(self.matrix_size), ot.unif(self.matrix_size))
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
    

