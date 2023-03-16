# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import ot
import matplotlib.style as mplstyle
import optuna
from joblib import parallel_backend

from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.colors as colors
from scipy import stats
import warnings

import scipy as sp
warnings.simplefilter("ignore")
# optuna.logging.set_verbosity(optuna.logging.WARNING)


# %%
class Adjust_Distribution():
    def __init__(self, model1, model2, adjust_path, gpu_queue = None, random_dataset = True):
        self.model1 = model1
        self.model2 = model2
        self.adjust_path = adjust_path
        self.size = len(model1)
        
        self.gpu_queue = gpu_queue
        self.random_dataset = random_dataset
    
        if not os.path.exists(self.adjust_path):
            os.makedirs(self.adjust_path)
    
    def make_limit(self):
        # m1 = self.model1.detach().clone()
        # m2 = self.model2.detach().clone()
        
        # m1 = m1.fill_diagonal_(float('nan'))
        # m2 = m2.fill_diagonal_(float('nan'))
        
        # lim_min = min(m1[~torch.isnan(m1)].min(), m2[~torch.isnan(m2)].min())
        # lim_max = max(m1[~torch.isnan(m1)].max(), m2[~torch.isnan(m2)].max())
        
        m2 = self.model2.detach().clone()
        m2 = m2.fill_diagonal_(float('nan'))
        
        lim_min = m2[~torch.isnan(m2)].min()
        lim_max = m2[~torch.isnan(m2)].max()

        return lim_max.item(), lim_min.item()
    
    def make_histogram(self, data):
    
        lim_max, lim_min = self.make_limit()
        bin = 100
        
        hist = torch.histc(data, bins = bin, min = lim_min, max = lim_max)
        
        mm1 = torch.tensor([0], device = hist.device)
        mm2 = torch.tensor([0], device = hist.device)
        hist = torch.cat((mm1, hist, mm2), dim = 0)
        
        m2 = data.detach().clone()
        m2 = m2.fill_diagonal_(float('nan'))
        m2 = m2[~torch.isnan(m2)]
        
        data_min = m2.min()
        data_max = m2.max()
        
        if data_max > lim_max:
            _, counts = torch.unique(m2[m2 > lim_max], return_counts = True)
            
            hist[-1] += counts.sum()

        if data_min < lim_min:
            _, counts = torch.unique(m2[m2 < lim_min], return_counts = True)
            
            hist[0] += counts.sum()
        
        return hist
    
    def __call__(self, trial):
        
        if self.gpu_queue is None:
            de = self.model1.device
        
        else:
            gpu_id = self.gpu_queue.get()
            de = 'cuda:' + str(gpu_id) 
        
        
        # alpha1 = trial.suggest_float("alpha1", 1e-6, 1e1, log = True)
        # lam1 = trial.suggest_float("lam1", 1e-6, 1e1, log = True)
        
        alpha2 = trial.suggest_float("alpha2", 1e-6, 1e1, log = True)
        lam2 = trial.suggest_float("lam2", 1e-6, 1e1, log = True)
        
        lam1 = 1.0
        alpha1 = 1.0
        
        model1_norm = self.model_normalize(self.model1.to(de), lam1, alpha1) 
        model2_norm = self.model_normalize(self.model2.to(de), lam2, alpha2)
        
        ot_cost = self.L2_wasserstain(model1_norm, model2_norm)
        
        if self.gpu_queue is not None:
            self.gpu_queue.put(gpu_id)
    
        return ot_cost
        
    def model_normalize(self, data, lam, alpha):
        data = alpha * ((torch.pow(1 + data, lam) - 1) / lam)
        return data

    def L2_wasserstain(self, m1, m2):
        
        # lim_max, lim_min = self.make_limit()
        # bins = 100
        
        # hist1 = torch.histc(m1, bins = bins, min = lim_min, max = lim_max)
        # hist2 = torch.histc(m2, bins = bins, min = lim_min, max = lim_max)
            
        hist1 = self.make_histogram(m1)
        hist2 = self.make_histogram(m2)
        
        h1_prob = hist1 / hist1.sum()
        h2_prob = hist2 / hist2.sum()
    
        ind1 = torch.arange(len(hist1)).float()
        ind2 = torch.arange(len(hist2)).float()
        
        cost_matrix = torch.cdist(ind1.unsqueeze(dim=1), ind2.unsqueeze(dim=1), p = 1).to(hist1.device)
        
        # res = ot.emd2(h1_prob, h2_prob, cost_matrix, center_dual = False)
        # res = ot.sinkhorn2(h1_prob, h2_prob, cost_matrix, reg = 1)
        # res = torch.sum(torch.mm(WassDists, cost_matrix))
            
        if res == 0:
            res = float('nan')
            
        
        return res

    def best_parameters(self, study):
        
        best_trial = study.best_trial
        
        # a1 = best_trial.params["alpha1"]
        # lam1 = best_trial.params["lam1"]
        
        a2 = best_trial.params["alpha2"]
        lam2 = best_trial.params["lam2"]
        
        a1 = 1
        lam1 = 1
        
        return a1, lam1, a2, lam2

    def output_result(self, study):
        a1, lam1, a2, lam2 = self.best_parameters(study)
        model1_norm = self.model_normalize(self.model1, lam1, a1) 
        model2_norm = self.model_normalize(self.model2, lam2, a2)
        return model1_norm, model2_norm
    
# %%
if __name__ == '__main__':
    model1 = torch.load('../../data/model1.pt')#.to('cuda')
    model2 = torch.load('../../data/model2.pt')#.to('cuda')
    tt = Adjust_Distribution(model1, model2, '../../results/adjust_histogram')
    
    

    
