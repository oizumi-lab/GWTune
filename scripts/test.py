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

# nvidia-smi --query-compute-apps=timestamp,pid,name,used_memory --format=csv # GPUをだれが使用しているのかを確認できるコマンド。

# %%
from src.gw_alignment import GW_Alignment
from src.utils import gw_optimizer

# %%
class Test():
    def __init__(self, path_model1, path_model2) -> None:
        self.path_model1 = path_model1
        self.path_model2 = path_model2

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

    def optimizer_test(self, filename, device, to_types):
        test_gw = GW_Alignment(self.model1, self.model2, self.p, self.q, max_iter = 30, device = device, to_types = to_types, gpu_queue = None)

        pruner_params = {'n_iter':5, 'n_startup_trials':1, 'n_warmup_steps':0, 'min_resource':5, 'reduction_factor' : 2}
        test_gw.set_params(pruner_params)

        opt = gw_optimizer.Optimizer(test_gw.save_path)

        study = opt.optimizer(test_gw,
                              to_types = test_gw.to_types,
                              method = 'optuna',
                              init_plans_list = ['diag'],
                              eps_list = [1e-4, 1e-2],
                              eps_log = True,
                              sampler_name = 'random',
                              pruner_name = 'median',
                              filename = filename,
                              sql_name = 'mysql',
                              storage = "mysql+pymysql://root@localhost/TestGW_Methods",
                              n_jobs = 10,
                              num_trial = 10,
                              delete_study = False,
                              )

        return study

    def adjustment_test(self, filename, device, to_types):
        test_gw = GW_Alignment(self.model1, self.model2, self.p, self.q, device = device, to_types = to_types, filename = filename, gpu_queue = None)

        opt = gw_optimizer.Optimizer(test_gw.save_path)

        study = opt.optimizer(test_gw, 
                              method = 'optuna', 
                              init_plans_list = ['diag'], 
                              eps_list = [1e-4, 1e-2], 
                              sampler_name = 'grid_search', 
                              filename = filename, 
                              n_jobs = 10, 
                              num_trial = 50)

        return study

# %%
if __name__ == '__main__':
    path1 = '../data/model1.pt'
    path2 = '../data/model2.pt'

    tgw = Test(path1, path2)
    # diagと['random', 'uniform']ではfilenameを分けてください
    filename = 'test'
    device = 'cpu'
    to_types = 'torch'
    tgw.optimizer_test(filename, device, to_types)
# %%
