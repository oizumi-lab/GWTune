# %%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# %%
import time
import numpy as np
import pandas as pd
import torch
import ot
import matplotlib.pyplot as plt
import functools

# nvidia-smi --query-compute-apps=timestamp,pid,name,used_memory --format=csv # GPUをだれが使用しているのかを確認できるコマンド。

# %%
from src.gw_alignment import GW_Alignment
from src.utils.adjust_distribution import Adjust_Distribution
from src.utils.gw_optimizer import load_optimizer

# %%
class Test():
    def __init__(self, path_model1, path_model2, device, to_types) -> None:
        self.path_model1 = path_model1
        self.path_model2 = path_model2
        self.main_save_path = '../results/gw_alignment_test'
        
        self.device = device
        self.to_types = to_types

        self.model1, self.model2, self.p, self.q = self.load_sample_data()

    def load_sample_data(self):
        """
        ただ、sample dataをloadするだけの関数。

        Returns:
            _type_: _description_
        """
        model1 = torch.load(self.path_model1)
        model2 = torch.load(self.path_model2)
        p = ot.unif(len(model1))
        q = ot.unif(len(model2))
        return model1, model2, p, q

    def main_test(self, filename):
        save_path = self.main_save_path + '/' + filename
        adjust_filename = filename + '_adjust'
        
        # まずは、histogramの調整の計算を行う。
        adjust = self.adjustment_test(save_path, fix_method = 'both')
        opt_adjust = self.optimizer(adjust_filename, save_path, n_jobs = 4, num_trial = 1000)
        
        forced_run = False
        study_adjust = opt_adjust.run_study(adjust, gpu_board = 'cuda:0', forced_run = forced_run)
        
        adjust.make_graph(study_adjust)
        
        model1_best_yj, model2_best_yj = adjust.best_models(study_adjust)
        
        # histogramを調整後にalignmentの計算を行う
        init_plans_list = ['random']
        eps_list = [1e-4, 1e-3]
        eps_log = True
        n_iter = 20
        
        test_gw = GW_Alignment(model1_best_yj, model2_best_yj, self.p, self.q, save_path, max_iter = 1000, n_iter = n_iter, device = self.device, to_types = self.to_types, gpu_queue = None)
        
        # 1. 初期値の選択。実装済みの初期値条件の抽出をgw_optimizer.pyからinit_matrix.pyに移動しました。
        init_plans = test_gw.main_compute.init_mat_builder.implemented_init_plans(init_plans_list)

        # 2. 最適化関数の定義。事前に、functools.partialで、必要なhyper parametersの条件を渡しておく。
        gw_objective = functools.partial(test_gw, init_plans_list = init_plans, eps_list = eps_list, eps_log = eps_log)
        
        # 3. 最適化を実行。run_studyに渡す関数は、alignmentとhistogramの両方ともを揃えるようにしました。
        opt_gw = self.optimizer(filename, save_path, n_jobs = 8, num_trial = 40, n_iter = n_iter)
        study = opt_gw.run_study(gw_objective, gpu_board = 'cuda')
        test_gw.load_graph(study)
        
        return study
        
        
    def optimizer(self, filename, save_path, n_jobs = 4, num_trial = 100, n_iter = 100, delete_study = False):
        
        # 分散計算のために使うRDBを指定
        sql_name = 'sqlite'
        storage = "sqlite:///" + save_path +  '/' + filename + '.db'
        
        # sql_name = 'mysql'
        # storage = 'mysql+pymysql://root:olabGPU61@localhost/GW_MethodsTest'

        # 使用するsamplerを指定
        # 現状使えるのは['random', 'grid', 'tpe']
        sampler_name = 'tpe'

        # 使用するprunerを指定
        # 現状使えるのは['median', 'hyperband', 'nop']
        # median pruner (ある時点でスコアが過去の中央値を下回っていた場合にpruning)
        #     n_startup_trials : この数字の数だけtrialが終わるまではprunerを作動させない
        #     n_warmup_steps   : 各trialについてこのステップ以下ではprunerを作動させない
        # hyperband pruner (pruning判定の期間がだんだん伸びていく代わりに基準がだんだん厳しくなっていく)
        #     min_resource     : 各trialについてこのステップ以下ではprunerを作動させない
        #     reduction_factor : どれくらいの間隔でpruningをcheckするか。値が小さいほど細かい間隔でpruning checkが入る。2~6程度.
        
        pruner_name = 'median'
        pruner_params = {'n_startup_trials': 1, 'n_warmup_steps': 2, 'min_resource': 2, 'reduction_factor' : 3}
        
        opt = load_optimizer(save_path,
                             n_jobs = n_jobs,
                             num_trial = num_trial,
                             to_types = self.to_types,
                             method = 'optuna',
                             sampler_name = sampler_name,
                             pruner_name = pruner_name,
                             pruner_params = pruner_params,
                             n_iter = n_iter,
                             filename = filename,
                             sql_name = sql_name,
                             storage = storage,
                             delete_study = delete_study)
        
        return opt

    def adjustment_test(self, save_path, fix_method = 'both'):
        """
        2023.3.29 佐々木
        yeo-johnson変換を使った場合のhistogramの調整(マッチング)を行う。
        """
        dataset = Adjust_Distribution(self.model1, self.model2, save_path, fix_method = fix_method, device = self.device, to_types = self.to_types, gpu_queue = None)
        return dataset
    
        
    
    

# %%
if __name__ == '__main__':
    path1 = '../data/model1.pt'
    path2 = '../data/model2.pt'
    device = 'cuda'
    to_types = 'torch'

    tgw = Test(path1, path2, device, to_types)
    # diagと['random', 'uniform']ではfilenameを分けてください
    
    filename = 'test'
    tgw.main_test(filename)
# %%
