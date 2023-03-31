#%%
import os, sys, gc, math
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
# warnings.simplefilter("ignore")
import seaborn as sns
import matplotlib.style as mplstyle
from tqdm.auto import tqdm
#nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# %%
from utils.backend import Backend
from utils.init_matrix import InitMatrix
from utils.gw_optimizer import load_optimizer

# %%
class GW_Alignment():
    def __init__(self, pred_dist, target_dist, p, q, save_path, max_iter = 1000, n_iter = 100, device='cpu', to_types='torch', gpu_queue = None):
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

        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.n_iter = n_iter

        self.main_compute = MainGromovWasserstainComputation(pred_dist, target_dist, p, q, self.device, self.to_types, max_iter = max_iter, n_iter = n_iter, gpu_queue = gpu_queue)

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
        if self.gpu_queue is None:
            gpu_id = None
            device = self.device

        else:
            gpu_id = self.gpu_queue.get()
            device = 'cuda:' + str(gpu_id % 4)

        if self.to_types == 'numpy':
            assert device == 'cpu'

        '''
        1.  define hyperparameter (eps, T)
        '''

        trial, eps = self.define_eps_range(trial, eps_list, eps_log)

        init_mat_plan = trial.suggest_categorical("initialize", init_plans_list) # init_matをdeviceに上げる作業はentropic_gw中で行うことにしました。(2023/3/14 阿部)

        trial.set_user_attr('size', self.size)

        file_path = self.save_path + '/' + init_mat_plan # ここのパス設定はoptimizer.py側でも使う可能性があるので、変更の可能性あり。

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

    def load_graph(self, study):
        best_trial = study.best_trial
        eps = best_trial.params['eps']
        init_plan = best_trial.params['initialize']
        acc = best_trial.user_attrs['acc']
        size = best_trial.user_attrs['size']
        number = best_trial.number

        if self.to_types == 'torch':
            gw = torch.load(self.save_path + '/' + init_plan + f'/gw_{best_trial.number}.pt')
        elif self.to_types == 'numpy':
            gw = np.load(self.save_path + '/' + init_plan + f'/gw_{best_trial.number}.npy')
        # gw = torch.load(self.file_path + '/GW({} pictures, epsilon = {}, trial = {}).pt'.format(size, round(eps, 6), number))
        self.plot_coupling(gw, eps, acc)

    def make_eval_graph(self, study):
        df_test = study.trials_dataframe()
        success_test = df_test[df_test['values_0'] != float('nan')]

        plt.figure()
        plt.title('The evaluation of GW results for random pictures')
        plt.scatter(success_test['values_1'], np.log(success_test['values_0']), label = 'init diag plan ('+str(self.train_size)+')', c = 'C0')
        plt.xlabel('accuracy')
        plt.ylabel('log(GWD)')
        plt.legend()
        plt.show()

    def plot_coupling(self, T, epsilon, acc):
        mplstyle.use('fast')
        N = T.shape[0]
        plt.figure(figsize=(8,6))
        if self.to_types == 'torch':
            T = T.to('cpu').numpy()
        sns.heatmap(T)

        plt.title('GW results ({} pictures, eps={}, acc.= {})'.format(N, round(epsilon, 6), round(acc, 4)))
        plt.tight_layout()
        plt.show()


class MainGromovWasserstainComputation():
    def __init__(self, pred_dist, target_dist, p, q, device, to_types, max_iter = 1000, n_iter = 100, gpu_queue = None) -> None:
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

        # 初期値のiteration回数, かつ hyperbandのparameter
        self.n_iter = n_iter

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

    def gw_alignment_computation(self, init_mat_plan, eps, max_iter, device, trial = None, seed = 42):
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

        return gw, logv, gw_loss, acc, init_mat

    def _save_results(self, gw_loss, acc, trial, init_mat_plan, num_iter = None, seed = None):

        gw_loss, acc = self.backend.get_item_from_torch_or_jax(gw_loss, acc)

        if init_mat_plan in ['random', 'permutation']:
            trial.set_user_attr('best_acc', acc)
            trial.set_user_attr('best_iter', num_iter)
            trial.set_user_attr('best_seed', int(seed)) # ここはint型に変換しないと、謎のエラーが出る (2023.3.18 佐々木)。
        else:
            trial.set_user_attr('acc', acc)

        return trial

    def _check_pruner_should_work(self, gw_loss, trial, init_mat_plan, eps, num_iter = None, gpu_id = None):
        """
        2023.3.28 佐々木
        全条件において、prunerを動かすメソッド。

        Args:
            gw_loss (_type_): _description_
            trial (_type_): _description_
            init_mat_plan (_type_): _description_
            eps (_type_): _description_
            num_iter (_type_, optional): _description_. Defaults to None.
            gpu_id (_type_, optional): _description_. Defaults to None.

        Raises:
            optuna.TrialPruned: _description_
        """
        
        if init_mat_plan in ['uniform', 'diag'] and math.isnan(gw_loss): # math.isnan()を使わないとnanの判定ができない。 
            # このifブロックがなくても、diag, uniformのprunerは正しく動作する。
            # ただ、tutorialの挙動を見ていると、これがあった方が良さそう。(2023.3.28 佐々木)
            self._gpu_queue_put(gpu_id)
            raise optuna.TrialPruned(f"Trial for '{init_mat_plan}' was pruned with parameters: {{'eps': {eps}}}")

        if num_iter is None: # uniform, diagにおいて、nanにならなかったがprunerが動くときのためのifブロック。
            num_iter = trial.number
        
        trial.report(gw_loss, num_iter)
        
        if trial.should_prune():
            self._gpu_queue_put(gpu_id)
            raise optuna.TrialPruned(f"Trial was pruned at iteration {num_iter} with parameters: {{'eps': {eps}, 'initialize': '{init_mat_plan}'}}")

    def _gpu_queue_put(self, gpu_id) -> None:
        if self.gpu_queue is not None:
            self.gpu_queue.put(gpu_id)

    def compute_GW_with_init_plans(self, trial, eps, init_mat_plan, device, gpu_id = None):
        """
        2023.3.17 佐々木
        uniform, diagでも、prunerを使うこともできるが、いまのところはコメントアウトしている。
        どちらにも使えるようにする場合は、ある程度の手直しが必要。
        
        2023.3.28 佐々木
        全条件において、正しくprunerを動かすメソッドを作成。
        各条件ごとへの拡張性を考慮すると、prunerの挙動は一本化しておく方が絶対にいい。
        """

        if init_mat_plan in ['uniform', 'diag']:
            gw, logv, gw_loss, acc, init_mat = self.gw_alignment_computation(init_mat_plan, eps, self.max_iter, device)
            trial = self._save_results(gw_loss, acc, trial, init_mat_plan)
            self._check_pruner_should_work(gw_loss, trial, init_mat_plan, eps, gpu_id = gpu_id)
            return gw, logv, gw_loss, acc, init_mat, trial

        elif init_mat_plan in ['random', 'permutation']:
            best_gw_loss = float('inf')
            for i, seed in enumerate(tqdm(np.random.randint(0, 100000, self.n_iter))):
                c_gw, c_logv, c_gw_loss, c_acc, c_init_mat = self.gw_alignment_computation(init_mat_plan, eps, self.max_iter, device, seed = seed)

                if c_gw_loss < best_gw_loss:
                    best_gw, best_logv, best_gw_loss, best_acc, best_init_mat = c_gw, c_logv, c_gw_loss, c_acc, c_init_mat
                    trial = self._save_results(best_gw_loss, best_acc, trial, init_mat_plan, num_iter = i, seed = seed)

                self._check_pruner_should_work(c_gw_loss, trial, init_mat_plan, eps, num_iter = i, gpu_id = gpu_id)

            if best_gw_loss == float('inf'):
                self._gpu_queue_put(gpu_id)
                raise optuna.TrialPruned(f"All iteration was failed with parameters: {{'eps': {eps}, 'initialize': '{init_mat_plan}'}}")

            else:
                return best_gw, best_logv, best_gw_loss, best_acc, best_init_mat, trial

        else:
            raise ValueError('Not defined initialize matrix.')


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

    init_mat_types = ['diag']
    eps_list = [1e-4, 1e-2]
    eps_log = True

    dataset = GW_Alignment(model1, model2, p, q, max_iter = 1000, n_iter = 100, device = 'cuda', save_path = unittest_save_path)
    # study.optimize(lambda trial: dataset(trial, init_mat_types, eps_list), n_trials = 20, n_jobs = 10)

    # %%
    # 以下はMulti-GPUで計算を回す場合。こちらの方が非常に早い。
    # from multiprocessing import Manager
    # from joblib import parallel_backend

    # n_gpu = 4
    # with Manager() as manager:
    #     gpu_queue = manager.Queue()

    #     for i in range(n_gpu):
    #         gpu_queue.put(i)

    #     dataset = GW_Alignment(model1, model2, p, q, unittest_save_path, max_iter = 1000, n_iter = 100, device = 'cuda', gpu_queue = gpu_queue)

    #     study = optuna.create_study(direction = "minimize",
    #                                 study_name = "test",
    #                                 sampler = optuna.samplers.TPESampler(seed = 42),
    #                                 pruner = optuna.pruners.MedianPruner(),
    #                                 storage = 'sqlite:///' + unittest_save_path + '/' + init_mat_types[0] + '.db', #この辺のパス設定は一度議論した方がいいかも。
    #                                 load_if_exists = True)

    #     with parallel_backend("multiprocessing", n_jobs = n_gpu):
    #         study.optimize(lambda trial: dataset(trial, init_mat_types, eps_list), n_trials = 20, n_jobs = n_gpu)
    
    # df = study.trials_dataframe()
    # print(df)
    # # %%
    # df.dropna()

    # %%
    from concurrent.futures import ThreadPoolExecutor

    study = optuna.create_study(direction = "minimize",
                        study_name = "test",
                        sampler = optuna.samplers.TPESampler(seed = 42),
                        pruner = optuna.pruners.MedianPruner(),
                        storage = 'sqlite:///' + unittest_save_path + '/' + init_mat_types[0] + '.db', #この辺のパス設定は一度議論した方がいいかも。
                        load_if_exists = True)

    def multi_run(dataset, seed):
        load_study = optuna.load_study(study_name = "test",
                                       sampler = optuna.samplers.TPESampler(seed = seed),
                                       pruner = optuna.pruners.MedianPruner(),
                                       storage = 'sqlite:///' + unittest_save_path + '/' + init_mat_types[0] + '.db'
                                       )

        load_study.optimize(lambda trial: dataset(trial, init_mat_types, eps_list), n_trials = 5, n_jobs = 1)


    processes = []
    
    n_jobs = 4
    seed = 42
    
    with ThreadPoolExecutor(n_jobs) as pool:
        for i in range(n_jobs):
            pool.submit(multi_run, dataset, seed + i)

#%%

