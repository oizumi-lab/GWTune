# %%
import time
import numpy as np
import pandas as pd
import torch
import ot
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../')
os.getcwd()

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

    def optimizer_test(self, device, to_types):
        test_gw = GW_Alignment(self.model1, self.model2, self.p, self.q, device=device, to_types=to_types, speed_test=True)
        
        opt = gw_optimizer.Optimizer(test_gw.save_path)
        
        study = opt.optimizer(test_gw, method = 'optuna', init_plans_list = ['diag'], sampler_name = 'random', filename = 'test', n_jobs = 10, num_trial = 50)
        
        return study


# %%
if __name__ == '__main__':
    path1 = '../data/model1.pt'
    path2 = '../data/model2.pt'

    tgw = Test(path1, path2)
    device = 'cuda'
    to_types = 'torch'
    tgw.optimizer_test(device, to_types)


# %%