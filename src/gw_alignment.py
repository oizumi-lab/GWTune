# %%
import os, sys, gc
import jax
import jax.numpy as jnp
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
from utils.backend import make_backend
from utils.init_matrix import InitMatrix

# %%
class GW_Alignment():
    def __init__(self, pred_dist, target_dist, p, q, max_iter = 1000, device='cpu', to_types='torch', speed_test=False, gpu_queue = None, save_path = None):
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

        self.nx = make_backend(self.to_types)
        self.pred_dist, self.target_dist, self.p, self.q = self.nx.change_data(pred_dist, target_dist, p, q)
        # be = Backend(self.device, self.to_types) # potのnxに書き換えるべき。
        # self.pred_dist, self.target_dist, self.p, self.q = be.change_data(pred_dist, target_dist, p, q)

        self.size = len(self.pred_dist)

        self.speed_test = speed_test

        self.save_path = '../data/gw_alignment' if save_path is None else save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # gw alignmentに関わるparameter
        self.max_iter = max_iter

        # hyperparameter
        self.initialize = ['uniform', 'random', 'permutation', 'diag']
        self.init_mat_builder = InitMatrix(self.size)

        # pruner parameter
        self.n_iter = 100
        # MedianPruner
        self.n_startup_trials = 5
        self.n_warmup_steps = 5
        # HyperbandPruner
        self.min_resource = 5
        self.reduction_factor = 2 # self.max_resource = self.n_iter


    def set_params(self, vars):
        '''
        2023/3/14 阿部

        インスタンス変数を外部から変更する関数
        '''
        for key, value in vars.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def entropic_gw(self, device, epsilon, T = None, max_iter = 1000, tol = 1e-9, log = True, verbose = False, trial = None):
        C1, C2, p, q = self.nx.to(self.pred_dist, device), self.nx.to(self.target_dist, device), self.nx.to(self.p, device), self.nx.to(self.q, device)

        T = self.nx.from_numpy(T)
        T = self.nx.to(T, device, dtype = 'float')

        # add T as an input
        if T is None:
            T = self.nx.outer(p, q)

        constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun = "square_loss")

        # constC, hC1, hC2, nx = self.mat_gw(device)

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
                err = self.nx.norm(T - Tprev)
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

    def gw_alignment_help(self,init_mat_plan, device, eps, seed):
        init_mat = self.init_mat_builder.make_initial_T(init_mat_plan, seed)
        gw, logv = self.entropic_gw(device, eps, T = init_mat, max_iter = self.max_iter)
        gw_loss = logv['gw_dist']

        if self.nx.array_equal(gw, self.nx.zeros(gw.shape)):
            gw_success = False
        else:
            gw_success = True

        return gw, logv, gw_loss, init_mat, gw_success

    def exit_torch(self, *args):
        l = []
        for v in args:
            if isinstance(v, torch.Tensor):
                v = v.item()
            l.append(v)
        return  l

    def __call__(self, trial, file_path, init_plans_list, eps_list, eps_log=True):
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
            eps = trial.suggest_float("eps", ep_lower, ep_upper, log = eps_log)
        elif len(eps_list) == 3:
            ep_lower, ep_upper, ep_step = eps_list
            eps = trial.suggest_float("eps", ep_lower, ep_upper, ep_step)
        else:
            raise ValueError("The eps_list doesn't match.")

        init_mat_plan = trial.suggest_categorical("initialize", init_plans_list) # 上記のリストから、1つの方法を取り出す(optunaがうまく選択してくれる)。
        # init_matをdeviceに上げる作業はentropic_gw中で行うことにしました。(2023/3/14 阿部)

        '''
        randomの時にprunerを設定する場合は、if init_mat_plan == "random": pruner を、self.entropi_GWのなかにあるwhileループにいれたら良い。
        '''
        trial.set_user_attr('size', self.size)

        '''
        2.  Compute GW alignment with hyperparameters defined above.
        '''
        if init_mat_plan in ['uniform', 'diag']:
            gw, logv, gw_loss, init_mat, gw_success = self.gw_alignment_help(init_mat_plan, device, eps, seed=42)
            gw_loss = self.nx.item(gw_loss)
            if not gw_success:
                gw_loss = float('nan')
                acc = float('nan')
                raise optuna.TrialPruned(f"Failed with parameters: {{'eps': {eps}, 'initialize': '{init_mat_plan}'}}")

        elif init_mat_plan in ['random', 'permutation']:
            gw_loss = float('inf')
            for i, seed in enumerate(np.random.randint(0, 100000, self.n_iter)):
                # current_init_mat = self.init_mat_builder.make_initial_T(init_mat_plan, seed)
                # current_gw, current_logv = self.entropic_gw(device, eps, T = current_init_mat, max_iter = self.max_iter)
                c_gw, c_logv, c_gw_loss, c_init_mat, c_gw_success = self.gw_alignment_help(init_mat_plan, device, eps, seed)
                c_gw_loss = self.nx.item(c_gw_loss)
                if not c_gw_success: # gw alignmentが失敗したならinf
                    c_gw_loss = float('inf')

                if c_gw_loss < gw_loss: # gw_lossの更新
                    gw_loss = c_gw_loss
                    gw = c_gw
                    init_mat = c_init_mat
                    logv = c_logv
                    best_seed = seed

                trial.report(gw_loss, i) # 最小値を報告
                if trial.should_prune():
                    raise optuna.TrialPruned(f"Trial was pruned at iteration {i} with parameters: {{'eps': {eps}, 'initialize': '{init_mat_plan}'}}")
            # trialが全て失敗したら
            if gw_loss == float('inf'):
                gw_loss = float('nan')
                acc = float('nan')
                raise optuna.TrialPruned(f"All iteration was failed with parameters: {{'eps': {eps}, 'initialize': '{init_mat_plan}'}}")
            # seedの保存
            trial.set_user_attr('seed', int(best_seed))
        else:
            raise ValueError('Not defined initialize matrix.')
        '''
        3.  count the accuracy of alignment and save the results if computation was finished in the right way.
            If not, set the result of accuracy and gw_loss as float('nan'), respectively. This will be used as a handy marker as bad results to be removed in the evaluation analysis.
        '''

        # if nx.array_equal(gw, nx.zeros(gw.shape)): # gwが0行列ならnanを返す
        #     gw_loss = float('nan')
        #     acc = float('nan')
        #     raise optuna.TrialPruned("Failed.")
        pred = self.nx.argmax(gw, 1)
        correct = (pred == self.nx.arange(len(gw), type_as = gw)).sum()
        acc = correct / len(gw)

        # save data
        self.nx.save(file_path + f'/gw_{trial.number}', gw)
        self.nx.save(file_path + f'/init_mat_{trial.number}', init_mat)

        acc = self.nx.item(acc)
        # jaxの保存方法を作成してください

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

# %%
