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
import warnings
import copy
# warnings.simplefilter("ignore")
import os

# %%
from src.utils.backend import Backend
from src.utils.init_matrix import InitMatrix

# %%
class GW_Alignment():
    def __init__(self, pred_dist, target_dist, p, q, device='cpu', to_types='torch', speed_test=False, gpu_queue = None, save_path = None):
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

        self.save_path = '../data/gw_alignment' if save_path is None else save_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # gw alignmentに関わるparameter
        self.max_iter = 1000
        self.stopping_rounds = None
        self.n_iter = 1
        self.punishment = float("nan")

        # hyperparameter
        self.initialize = ['uniform', 'random', 'permutation', 'diag']
        self.init_mat_builder = InitMatrix(self.size)
        self.n_iter = 100

        # optuna parameter
        self.min_resource = 3
        self.max_resource = (self.max_iter // 10) * self.n_iter
        self.reduction_factor = 3


    def entropic_gw(self, device, epsilon, T = None, max_iter = 1000, tol = 1e-9, log = True, verbose = False, trial = None):

        if self.to_types == 'torch':
            C1, C2, p, q = self.pred_dist.to(device), self.target_dist.to(device), self.p.to(device), self.q.to(device)
        else:
            C1, C2, p, q = self.pred_dist, self.target_dist, self.p, self.q
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
                err_prev = copy.copy(err)
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

    def mat_gw(self, device):
        """
        2023/3/13(阿部)
        gwd計算のための行列初期化。entropic_GWの最初と全く同じ

        Args:
            init_plans_list (list) : 初期値の条件を1個または複数個入れたリスト。

        Raises:
            ValueError: 選択したい条件が1つであっても、リストで入力をすること。

        Returns:
            list : 選択希望の条件のリスト。
        """

        if self.to_types == 'torch':
            C1, C2, p, q = self.pred_dist.to(device), self.target_dist.to(device), self.p.to(device), self.q.to(device)
        else:
            C1, C2, p, q = self.pred_dist, self.target_dist, self.p, self.q
        nx = ot.backend.get_backend(C1, C2, p, q)

        constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun = "square_loss")
        return constC, hC1, hC2


    def iter_entropic_gw(self, device, eps, init_mat_plan, trial):
        """
        n_iter回くりかえす関数

        """
        min_gwd = float('inf')
        for i,seed in enumerate(np.random.randint(self.n_iter)):
            np.random.seed(seed)
            init_mat = self.init_mat_builder.make_initial_T(init_mat_plan)
            gw, logv = self.entropic_gw(device, eps, T = init_mat)
            gwd = logv['gw_dist']
            if gwd < min_gwd:
                min_gwd = gwd
                best_gw = gw
                best_init_mat = init_mat
                best_logv = logv

            constC, hC1, hC2 = self.mat_gw(device)
            trial.report(ot.gromov.gwloss(constC, hC1, hC2, gw), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best_gw, best_logv



    def __call__(self, trial, init_plans_list, eps_list):

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
        ep_lower, ep_upper = eps_list

        if len(eps_list) == 2:
            eps = trial.suggest_float("eps", ep_lower, ep_upper, log = True)
        elif len(eps_list) == 3:
            ep_lower, ep_upper, ep_step = eps_list
            eps = trial.suggest_float("eps", ep_lower, ep_upper, ep_step)
        else:
            raise ValueError("The eps_list doesn't match.")

        init_mat_plan = trial.suggest_categorical("initialize", init_plans_list) # 上記のリストから、1つの方法を取り出す(optunaがうまく選択してくれる)。
        init_mat = self.init_mat_builder.make_initial_T(init_mat_plan) # epsの値全部を計算する際、randomは何回も計算していいけど、diag, uniformは一回だけでいいので、うまく切り分けよう。
        init_mat = torch.from_numpy(init_mat).float().to(device)

        '''
        randomの時にprunerを設定する場合は、if init_mat_plan == "random": pruner を、self.entropi_GWのなかにあるwhileループにいれたら良い。
        '''

        seed = trial.suggest_init('seed', 0, 100)

        trial.set_user_attr('size', self.size)

        '''
        2.  Compute GW alignment with hyperparameters defined above.
        '''
        if init_mat_plan in ['uniform', 'diag']:
            gw, logv = self.entropic_gw(device, eps, T = init_mat)
        elif init_mat_plan in ['random', 'permutation']:
            gw, logv = self.iter_entropic_gw(device, eps, init_mat_plan, trial)
        else:
            raise ValueError('Not defined initialize matrix.')

        '''



        '''
        3.  count the accuracy of alignment and save the results if computation was finished in the right way.
            If not, set the result of accuracy and gw_loss as float('nan'), respectively. This will be used as a handy marker as bad results to be removed in the evaluation analysis.
        '''

        if torch.count_nonzero(gw).item() != 0:
            gw_loss = logv['gw_dist'].item()

            _, pred = torch.max(gw, 1)
            acc = pred.eq(torch.arange(len(gw)).to(device)).sum() / len(gw)

            torch.save(gw, self.save_path + '/GW({} pictures, epsilon={}).pt'.format(self.size, round(eps, 6)))
            acc = acc.item()
        else:
            gw_loss = float('nan')
            acc = float('nan')

        trial.set_user_attr('acc', acc)
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

        return gw_loss
