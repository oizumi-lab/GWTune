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
        """
        if len(eps_list) == 2:
            ep_lower, ep_upper = eps_list
            eps = trial.suggest_float("eps", ep_lower, ep_upper, log = eps_log)
        elif len(eps_list) == 3:
            ep_lower, ep_upper, ep_step = eps_list
            eps = trial.suggest_float("eps", ep_lower, ep_upper, step = ep_step)
        else:
            raise ValueError("The eps_list doesn't match.")

        return trial, eps

    def check_eps(self, eps_list, eps_log):

        if type(eps_list) != list:
            raise ValueError('variable named "eps_list" is not list!')

        else:
            if len(eps_list) == 2:
                pass
            if len(eps_list) == 3:
                if eps_log:
                    warnings.warn('You cannot use "eps_log" and "eps_step" at the same time, in such case "eps_log = False". \n If you want to use "eps_log = True", set "eps_list = [eps_lower, eps_upper]".', UserWarning)
                    eps_log = False

            else:
                ValueError('Not defined initialize matrix.')

            return eps_list, eps_log


    def __call__(self, trial, device, init_plans_list, eps_list, eps_log=True):
        '''
        0.  define the "gpu_queue" here.
            This will be used when the memory of dataset was too much large for a single GPU board, and so on.
        '''
        # if self.gpu_queue is None:
        #     gpu_id = None
        #     device = self.device

        # else:
        #     gpu_id = self.gpu_queue.get()
        #     device = 'cuda:' + str(gpu_id % 4)

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
        gw, logv, gw_loss, acc, init_mat, trial = self.main_compute.compute_GW_with_init_plans(trial, eps, init_mat_plan, device, gpu_id = None)

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
        # if self.gpu_queue is not None:
        #     self.gpu_queue.put(gpu_id)

        return gw_loss

    def load_graph(self, study):
        best_trial = study.best_trial
        eps = best_trial.params['eps']
        init_plan = best_trial.params['initialize']

        if init_plan in ['uniform', 'diag']:
            acc = best_trial.user_attrs['acc']
        else:
            acc = best_trial.user_attrs['best_acc']

        size = best_trial.user_attrs['size']
        number = best_trial.number

        if self.to_types == 'torch':
            gw = torch.load(self.save_path + '/' + init_plan + f'/gw_{number}.pt')
        elif self.to_types == 'numpy':
            gw = np.load(self.save_path + '/' + init_plan + f'/gw_{number}.npy')

        self.plot_coupling(gw, eps, acc)

    def make_eval_graph(self, study):
        df_test = study.trials_dataframe()
        success_test = df_test[df_test['values_0'] != float('nan')]

        plt.figure()
        plt.title('The evaluation of GW results for random pictures')
        plt.scatter(success_test['values_1'], np.log(success_test['values_0']), label = 'init diag plan ('+str(self.size)+')', c = 'C0')
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
        # self.initialize = ['uniform', 'random', 'permutation', 'diag']
        self.init_mat_builder = InitMatrix(matrix_size = self.size, backend = self.backend)

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

    def multi_run(dataset, seed, device, num_trials = 40):
        load_study = optuna.load_study(study_name = "test",
                                       sampler = optuna.samplers.TPESampler(seed = seed),
                                       pruner = optuna.pruners.MedianPruner(),
                                       storage = 'sqlite:///' + unittest_save_path + '/' + init_mat_types[0] + '.db')

        load_study.optimize(lambda trial: dataset(trial, init_mat_types, eps_list, device), n_trials = num_trials, n_jobs = 1)

    n_trials = 20
    n_jobs = 10
    seed = 42

    with ThreadPoolExecutor(n_jobs) as pool:
        for i in range(n_jobs):
            device = 'cuda:0'# + str(i % 4)
            dataset = GW_Alignment(model1, model2, p, q, max_iter = 1000, n_iter = 100, device = device, save_path = unittest_save_path)
            pool.submit(multi_run, dataset, seed + i, device, num_trials = n_trials // n_jobs)

    #%%
    study = optuna.load_study(study_name = "test",
                              sampler = optuna.samplers.TPESampler(seed = seed),
                              pruner = optuna.pruners.MedianPruner(),
                              storage = 'sqlite:///' + unittest_save_path + '/' + init_mat_types[0] + '.db')

    df = study.trials_dataframe()
    print(df)
    # %%
    df.dropna()


    # %%
    """
    [I 2023-04-01 00:05:16,942] A new study created in RDB with name: test
    /home/masaru-sasaki/.pyenv/versions/mambaforge-22.9.0-3/lib/python3.10/site-packages/ot/bregman.py:492: UserWarning: Warning: numerical errors at iteration 0
    warnings.warn('Warning: numerical errors at iteration %d' % ii)
    [I 2023-04-01 00:05:18,148] Trial 1 pruned. Trial for 'diag' was pruned with parameters: {'eps': 0.00016864634905345633}
    [I 2023-04-01 00:05:18,518] Trial 8 pruned. Trial for 'diag' was pruned with parameters: {'eps': 0.0001698670453505509}
    [I 2023-04-01 00:05:18,524] Trial 9 pruned. Trial for 'diag' was pruned with parameters: {'eps': 0.00010838783508283825}
    [I 2023-04-01 00:06:14,865] Trial 12 finished with value: 0.013235032558441162 and parameters: {'eps': 0.0060694108171990896, 'initialize': 'diag'}. Best is trial 12 with value: 0.013235032558441162.
    [I 2023-04-01 00:06:19,481] Trial 6 finished with value: 0.013266123831272125 and parameters: {'eps': 0.009506551974984317, 'initialize': 'diag'}. Best is trial 12 with value: 0.013235032558441162.
    [I 2023-04-01 00:06:20,045] Trial 10 finished with value: 0.013262338005006313 and parameters: {'eps': 0.00889131893580497, 'initialize': 'diag'}. Best is trial 12 with value: 0.013235032558441162.
    [I 2023-04-01 00:06:21,871] Trial 4 finished with value: 0.013180121779441833 and parameters: {'eps': 0.003695427629909335, 'initialize': 'diag'}. Best is trial 4 with value: 0.013180121779441833.
    [I 2023-04-01 00:06:22,595] Trial 7 finished with value: 0.013094939291477203 and parameters: {'eps': 0.002246274520879033, 'initialize': 'diag'}. Best is trial 7 with value: 0.013094939291477203.
    [I 2023-04-01 00:06:22,762] Trial 15 pruned. Trial for 'diag' was pruned with parameters: {'eps': 0.00012286391913479718}
    [I 2023-04-01 00:06:30,498] Trial 3 finished with value: 0.013209367170929909 and parameters: {'eps': 0.00467395253077256, 'initialize': 'diag'}. Best is trial 7 with value: 0.013094939291477203.
    [I 2023-04-01 00:06:45,984] Trial 11 finished with value: 0.013026578351855278 and parameters: {'eps': 0.0016524680777017765, 'initialize': 'diag'}. Best is trial 11 with value: 0.013026578351855278.
    [I 2023-04-01 00:06:51,954] Trial 2 finished with value: 0.012748004868626595 and parameters: {'eps': 0.0009754461323261458, 'initialize': 'diag'}. Best is trial 2 with value: 0.012748004868626595.
    [I 2023-04-01 00:07:05,149] Trial 14 finished with value: 0.013054275885224342 and parameters: {'eps': 0.0018606616740103096, 'initialize': 'diag'}. Best is trial 2 with value: 0.012748004868626595.
    [I 2023-04-01 00:07:15,443] Trial 13 finished with value: 0.012949157506227493 and parameters: {'eps': 0.0012562887007755699, 'initialize': 'diag'}. Best is trial 2 with value: 0.012748004868626595.
    [I 2023-04-01 00:07:18,461] Trial 16 finished with value: 0.013045035302639008 and parameters: {'eps': 0.0017871695144801546, 'initialize': 'diag'}. Best is trial 2 with value: 0.012748004868626595.
    [I 2023-04-01 00:07:28,586] Trial 0 finished with value: 0.012348032556474209 and parameters: {'eps': 0.0005611516415334506, 'initialize': 'diag'}. Best is trial 0 with value: 0.012348032556474209.
    [I 2023-04-01 00:07:39,551] Trial 17 finished with value: 0.012670058757066727 and parameters: {'eps': 0.0008807412123296194, 'initialize': 'diag'}. Best is trial 0 with value: 0.012348032556474209.
    [I 2023-04-01 00:07:56,682] Trial 5 finished with value: 0.0120679447427392 and parameters: {'eps': 0.00039987928942942635, 'initialize': 'diag'}. Best is trial 5 with value: 0.0120679447427392.
    [I 2023-04-01 00:08:10,249] Trial 18 finished with value: 0.012284882366657257 and parameters: {'eps': 0.0005220207331711442, 'initialize': 'diag'}. Best is trial 5 with value: 0.0120679447427392.
    [I 2023-04-01 00:08:24,811] Trial 19 finished with value: 0.012181275524199009 and parameters: {'eps': 0.000459804725256059, 'initialize': 'diag'}. Best is trial 5 with value: 0.0120679447427392.
    """

    """
    [I 2023-04-01 00:11:08,739] A new study created in RDB with name: test
    /home/masaru-sasaki/.pyenv/versions/mambaforge-22.9.0-3/lib/python3.10/site-packages/ot/bregman.py:492: UserWarning: Warning: numerical errors at iteration 0
    warnings.warn('Warning: numerical errors at iteration %d' % ii)
    [I 2023-04-01 00:11:09,777] Trial 1 pruned. Trial for 'diag' was pruned with parameters: {'eps': 0.0001698670453505509}
    [I 2023-04-01 00:11:10,151] Trial 8 pruned. Trial for 'diag' was pruned with parameters: {'eps': 0.00016864634905345633}
    [I 2023-04-01 00:11:10,214] Trial 5 pruned. Trial for 'diag' was pruned with parameters: {'eps': 0.00010838783508283825}
    [I 2023-04-01 00:12:06,391] Trial 12 finished with value: 0.013235032558441162 and parameters: {'eps': 0.0060694108171990896, 'initialize': 'diag'}. Best is trial 12 with value: 0.013235032558441162.
    [I 2023-04-01 00:12:11,041] Trial 3 finished with value: 0.013266123831272125 and parameters: {'eps': 0.009506551974984317, 'initialize': 'diag'}. Best is trial 12 with value: 0.013235032558441162.
    [I 2023-04-01 00:12:11,608] Trial 11 finished with value: 0.013262338005006313 and parameters: {'eps': 0.00889131893580497, 'initialize': 'diag'}. Best is trial 12 with value: 0.013235032558441162.
    [I 2023-04-01 00:12:13,571] Trial 4 finished with value: 0.013180121779441833 and parameters: {'eps': 0.003695427629909335, 'initialize': 'diag'}. Best is trial 4 with value: 0.013180121779441833.
    [I 2023-04-01 00:12:14,527] Trial 7 finished with value: 0.013094939291477203 and parameters: {'eps': 0.002246274520879033, 'initialize': 'diag'}. Best is trial 7 with value: 0.013094939291477203.
    [I 2023-04-01 00:12:14,731] Trial 15 pruned. Trial for 'diag' was pruned with parameters: {'eps': 0.00012286391913479718}
    [I 2023-04-01 00:12:22,171] Trial 2 finished with value: 0.013209367170929909 and parameters: {'eps': 0.00467395253077256, 'initialize': 'diag'}. Best is trial 7 with value: 0.013094939291477203.
    [I 2023-04-01 00:12:37,812] Trial 10 finished with value: 0.013026578351855278 and parameters: {'eps': 0.0016524680777017765, 'initialize': 'diag'}. Best is trial 10 with value: 0.013026578351855278.
    [I 2023-04-01 00:12:43,798] Trial 6 finished with value: 0.012748004868626595 and parameters: {'eps': 0.0009754461323261458, 'initialize': 'diag'}. Best is trial 6 with value: 0.012748004868626595.
    [I 2023-04-01 00:12:57,259] Trial 14 finished with value: 0.013054275885224342 and parameters: {'eps': 0.0018606616740103096, 'initialize': 'diag'}. Best is trial 6 with value: 0.012748004868626595.
    [I 2023-04-01 00:13:07,250] Trial 13 finished with value: 0.012949157506227493 and parameters: {'eps': 0.0012562887007755699, 'initialize': 'diag'}. Best is trial 6 with value: 0.012748004868626595.
    [I 2023-04-01 00:13:09,679] Trial 16 finished with value: 0.013045035302639008 and parameters: {'eps': 0.0017871695144801546, 'initialize': 'diag'}. Best is trial 6 with value: 0.012748004868626595.
    [I 2023-04-01 00:13:20,035] Trial 0 finished with value: 0.012348032556474209 and parameters: {'eps': 0.0005611516415334506, 'initialize': 'diag'}. Best is trial 0 with value: 0.012348032556474209.
    [I 2023-04-01 00:13:31,620] Trial 17 finished with value: 0.012670058757066727 and parameters: {'eps': 0.0008807412123296194, 'initialize': 'diag'}. Best is trial 0 with value: 0.012348032556474209.
    [I 2023-04-01 00:13:48,982] Trial 9 finished with value: 0.0120679447427392 and parameters: {'eps': 0.00039987928942942635, 'initialize': 'diag'}. Best is trial 9 with value: 0.0120679447427392.
    [I 2023-04-01 00:14:03,002] Trial 18 finished with value: 0.012284882366657257 and parameters: {'eps': 0.0005220207331711442, 'initialize': 'diag'}. Best is trial 9 with value: 0.0120679447427392.
    [I 2023-04-01 00:14:17,417] Trial 19 finished with value: 0.012181275524199009 and parameters: {'eps': 0.000459804725256059, 'initialize': 'diag'}. Best is trial 9 with value: 0.0120679447427392.
    """
