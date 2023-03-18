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
from tqdm.auto import tqdm

# %%
from utils.backend import Backend
from utils.init_matrix import InitMatrix
from utils.gw_optimizer import RunOptuna

class GW_Alignment():
    def __init__(self, pred_dist, target_dist, p, q, max_iter = 1000, device='cpu', to_types='torch', main_save_path = None, gpu_queue = None):
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
        self.size = len(p)

        self.main_save_path = '../results/gw_alignment' if main_save_path is None else main_save_path
        if not os.path.exists(self.main_save_path):
            os.makedirs(self.main_save_path)
        
        self.main_compute = MainGromovWasserstainComputation(pred_dist, target_dist, p, q, self.device, self.to_types, max_iter = max_iter, gpu_queue = gpu_queue)

    def set_params(self, vars): # この関数の使用目的がoptimizer側にあるので、ここには定義しないほうがいいです。
        '''
        2023/3/14 阿部

        インスタンス変数を外部から変更する関数
        '''
        for key, value in vars.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
    def define_eps_range(self, trial, eps_list, eps_log):
        """
        2023.3.16 佐々木 作成
        
        epsの範囲を指定する関数。
        Args:
            trial (_type_): _description_
            eps_list (_type_): _description_
            eps_log (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        ep_lower, ep_upper = eps_list

        if len(eps_list) == 2:
            eps = trial.suggest_float("eps", ep_lower, ep_upper, log = eps_log)
        elif len(eps_list) == 3:
            ep_lower, ep_upper, ep_step = eps_list
            eps = trial.suggest_float("eps", ep_lower, ep_upper, ep_step)
        else:
            raise ValueError("The eps_list doesn't match.")
        
        return trial, eps

    def __call__(self, trial, init_plans_list, eps_list, eps_log=True):
        '''
        0.  define the "gpu_queue" here. 
            This will be used when the memory of dataset was too much large for a single GPU board, and so on.
        '''

        if self.to_types != 'numpy':
            if self.gpu_queue is None:
                device = self.device

            else:
                gpu_id = self.gpu_queue.get()
                device = 'cuda:' + str(gpu_id % 4)
        
        else:
            device = 'cpu'
        
        '''
        1.  define hyperparameter (eps, T)
        '''
        
        trial, eps = self.define_eps_range(trial, eps_list, eps_log)

        init_mat_plan = trial.suggest_categorical("initialize", init_plans_list) # init_matをdeviceに上げる作業はentropic_gw中で行うことにしました。(2023/3/14 阿部)
        
        trial.set_user_attr('size', self.size)
        
        file_path = self.main_save_path + '/' + init_mat_plan # ここのパス設定はoptimizer.py側でも使う可能性があるので、変更の可能性あり。
        
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok = True)

        '''
        2.  Compute GW alignment with hyperparameters defined above.
        '''

        gw, logv, gw_loss, acc, init_mat, trial = self.main_compute.compute_GW_with_init_plans(trial, eps, init_mat_plan, device, gpu_id = gpu_id)
        
        '''
        3.  count the accuracy of alignment and save the results if computation was finished in the right way.
            If not, set the result of accuracy and gw_loss as float('nan'), respectively. This will be used as a handy marker as bad results to be removed in the evaluation analysis.
        '''

        self.main_compute.backend.save_computed_results(gw, init_mat, file_path, trial.number)
        
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


class MainGromovWasserstainComputation():
    def __init__(self, pred_dist, target_dist, p, q, device, to_types, max_iter = 1000, gpu_queue = None) -> None:
        self.device = device
        self.to_types = to_types
        self.size = len(p)
        
        self.backend = Backend(device, to_types)
        self.pred_dist, self.target_dist, self.p, self.q = self.backend(pred_dist, target_dist, p, q) # 特殊メソッドのcallに変更した (2023.3.16 佐々木)
        
        # hyperparameter
        self.initialize = ['uniform', 'random', 'permutation', 'diag']
        self.init_mat_builder = InitMatrix(self.size, self.backend)
        
        # gw alignmentに関わるparameter
        self.max_iter = max_iter

        # pruner parameter
        self.n_iter = 100
        # MedianPruner
        self.n_startup_trials = 5
        self.n_warmup_steps = 5
        # HyperbandPruner
        self.min_resource = 5
        self.reduction_factor = 2
        
        # Multi-GPUの時に使う。
        self.gpu_queue = gpu_queue

    def entropic_gw(self, device, epsilon, T, max_iter = 1000, tol = 1e-9, trial = None, log = True, verbose = False):
        '''
        2023.3.16 佐々木
        backendに実装した "change_device" で、全型を想定した変数のdevice切り替えを行う。
        numpyに関しては、CPUの指定をしていないとエラーが返ってくるようにしているだけ。
        torch, jaxに関しては、GPUボードの番号指定が、これでできる。
        # # add T as an input
        # if T is None:
        #     T = self.backend.nx.outer(p, q) # このif文の状況を想定している場面がないので、消してもいいのでは？？ (2023.3.16 佐々木)
        '''
        C1, C2, p, q, T = self.backend.change_device(device, self.pred_dist, self.target_dist, self.p, self.q, T)
         
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
                # err_prev = copy.copy(err)　#ここは使われていないようなので、一旦コメントアウトしました (2023.3.16 佐々木)
                err = self.backend.nx.norm(T - Tprev)
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

    def gw_alignment_computation(self, init_mat_plan, eps, max_iter, device, trial = None, num_iter = None, seed = 42, gpu_id = None):
        """
        gw_alignmentの計算を行う。

        Args:
            init_mat_plan (_type_): _description_
            eps (_type_): _description_
            max_iter (_type_): _description_
            device (_type_): _description_
            trial (_type_, optional): _description_. Defaults to None.
            num_iter (int, forループのiter): randomやpermutationの時のforループの回数を受け取る. Defaults to None.
            seed (int, optional): _description_. Defaults to 42.

        Raises:
            optuna.TrialPruned: _description_

        Returns:
            _type_: _description_
        """
        
        '''
        2023.3.17 佐々木
        gw_alignmentの計算を行う。ここのメソッドは変更しない方がいいと思う。
        外部で、特定のhyper parametersでのgw_alignmentの計算結果だけを抽出したい時にも使えるため。
        '''
        
        init_mat = self.init_mat_builder.make_initial_T(init_mat_plan, seed)
        gw, logv = self.entropic_gw(device, eps, init_mat, max_iter = max_iter, trial = trial)
        gw_loss = logv['gw_dist']
        
        if self.backend.check_zeros(gw):
            gw_loss = float('nan')
            acc = float('nan')
            
        else:
            pred = self.backend.nx.argmax(gw, 1)
            correct = (pred == self.backend.nx.arange(len(gw), type_as = gw)).sum()
            acc = correct / len(gw)
        
        if trial is not None:
            gw_loss, acc = self.backend.get_item_from_torch_or_jax(gw_loss, acc)
            trial.set_user_attr('acc', acc)
            trial.set_user_attr('num_iter', num_iter)
            trial.report(gw_loss, num_iter)
            
            if trial.should_prune():
                if self.gpu_queue is not None:
                    self.gpu_queue.put(gpu_id)
                
                raise optuna.TrialPruned(f"Trial was pruned at iteration {trial.number}, {num_iter} with parameters: {{'eps': {eps}, 'initialize': '{init_mat_plan}'}}")
        
        return gw, logv, gw_loss, acc, init_mat, trial
    
    def compute_GW_with_init_plans(self, trial, eps, init_mat_plan, device, gpu_id = None):
        """
        2023.3.17 佐々木
        uniform, diagでも、prunerを使うこともできるが、いまのところはコメントアウトしている。
        どちらにも使えるようにする場合は、少しだけ手直しが必要。
        """
        
        if init_mat_plan in ['uniform', 'diag']:
            # best_gw_loss = float('inf')
            gw, logv, gw_loss, acc, init_mat, trial = self.gw_alignment_computation(init_mat_plan, eps, self.max_iter, device, trial = trial, gpu_id = gpu_id)
            # gw, logv, gw_loss, acc, init_mat = self.save_best_results(trial, init_mat_plan, best_gw_loss, gw, logv, gw_loss, acc, init_mat)
            
            return gw, logv, gw_loss, acc, init_mat, trial
            
        elif init_mat_plan in ['random', 'permutation']:
            best_gw_loss = float('inf')
            for i, seed in enumerate(tqdm(np.random.randint(0, 100000, self.n_iter))):
                c_gw, c_logv, c_gw_loss, c_acc, c_init_mat, trial = self.gw_alignment_computation(init_mat_plan, eps, self.max_iter, device, trial = trial, num_iter = i, seed = seed, gpu_id = gpu_id)
                
                if c_gw_loss < best_gw_loss:
                    # 以下のコードはuniform, diagでも使い回せる内容なので、うまく書き直したい。(2023.3.18, 佐々木)
                    best_gw, best_logv, best_gw_loss, best_acc, best_init_mat = c_gw, c_logv, c_gw_loss, c_acc, c_init_mat
                    
                    if init_mat_plan in ['random', 'permutation']: 
                        best_seed = seed
                        trial.set_user_attr('best_seed', int(best_seed)) # ここはint型に変換しないと、謎のエラーが出る (2023.3.18 佐々木)。
                    
                # if i % 10 == 0:
                #     print(i, c_gw_loss, best_gw_loss)
                    
            if best_gw_loss == float('inf'):
                return c_gw, c_logv, c_gw_loss, c_acc, c_init_mat, trial
            
            else:
                return best_gw, best_logv, best_gw_loss, best_acc, best_init_mat, trial
                
        
        # else:
        #     raise ValueError('Not defined initialize matrix.')




# %%
if __name__ == '__main__':
    
    os.chdir(os.path.dirname(__file__))
    
    path1 = '../data/model1.pt'
    path2 = '../data/model2.pt'
    unittest_save_path = '../results/unittest/gw_alignment'
    
    model1 = torch.load(path1)
    model2 = torch.load(path2)
    p = ot.unif(len(model1))
    q = ot.unif(len(model2))
    
    init_mat_types = ['random'] 
    eps_list = [1e-4, 1e-2]
    eps_log = True
    
    # dataset = GW_Alignment(model1, model2, p, q, max_iter = 1000, device = 'cuda', save_path = unittest_save_path)
    # study.optimize(lambda trial: dataset(trial, init_mat_types, eps_list), n_trials = 20, n_jobs = 10)
    
    # %%
    # 以下はMulti-GPUで計算を回す場合。こちらの方が非常に早い。
    from multiprocessing import Manager
    from joblib import parallel_backend
    
    n_gpu = 4
    with Manager() as manager:
        gpu_queue = manager.Queue()
        
        for i in range(n_gpu):
            gpu_queue.put(i)
            
        dataset = GW_Alignment(model1, model2, p, q, max_iter = 1000, device = 'cuda', save_path = unittest_save_path, gpu_queue = gpu_queue)

        study = optuna.create_study(direction = "minimize",
                                    study_name = "test",
                                    sampler = optuna.samplers.TPESampler(seed = 42),
                                    pruner = optuna.pruners.MedianPruner(),
                                    storage = 'sqlite:///' + unittest_save_path + '/' + init_mat_types[0] + '.db', #この辺のパス設定は一度議論した方がいいかも。
                                    load_if_exists = True)

        with parallel_backend("multiprocessing", n_jobs = n_gpu):
            study.optimize(lambda trial: dataset(trial, init_mat_types, eps_list), n_trials = 20, n_jobs = n_gpu)
    
    # %%
    # study = optuna.create_study(direction = "minimize",
    #                             study_name = "test",
    #                             storage = 'sqlite:///' + unittest_save_path + '/' + init_mat_types[0] + '.db', #この辺のパス設定は一度議論した方がいいかも。
    #                             load_if_exists = True)
    
    df = study.trials_dataframe()
    print(df)
# %%
    