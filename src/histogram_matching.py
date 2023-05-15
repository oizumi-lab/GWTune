# Standard Library
import itertools
import os
import warnings
import functools

warnings.simplefilter("ignore")

# Third Party Library
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.colors as colors
# %config InlineBackend.figure_formats = {'png', 'retina'} # for notebook?
plt.rcParams["font.size"] = 14

import torch
import optuna
import ot

from scipy.special import comb
from sklearn.metrics import confusion_matrix, accuracy_score

# optuna.logging.set_verbosity(optuna.logging.WARNING)
# nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
from utils.backend import Backend

# %%
class HistogramMatching():
    def __init__(self, source, target, to_types = 'torch'):
        self.source = source
        self.target = target
  
        self.to_types = to_types

    def sort_for_scaling(self, data):
        # x = sp.spatial.distance.squareform(data)
        x = data.flatten()
        x_sorted = np.sort(x)
        x_inverse_idx = np.argsort(x).argsort()
        return x, x_sorted, x_inverse_idx

    def pointwise_matching(self, X, Y):
        x, x_sorted, x_inverse_idx = self.sort_for_scaling(X)
        y, y_sorted, y_inverse_idx = self.sort_for_scaling(Y)
    
        y_t = x_sorted[y_inverse_idx]
        Y_t = y_t.reshape(Y.shape) #= sp.spatial.distance.squareform(y_t)
        return Y_t
        

# %%
class Adjust_Distribution():
    def __init__(self, model1, model2, adjust_path, fix_method = 'target', to_types = 'torch'):
        self.adjust_path = adjust_path
        self.model1, self.model2 = model1, model2
        
        self.size = len(model1)
        self.to_types = to_types
        
        self.fix_method = fix_method
    
        if not os.path.exists(self.adjust_path):
            os.makedirs(self.adjust_path)
    
    def _extract_min_and_max(self, data):
        data = data.detach().clone()
        data = data.fill_diagonal_(float('nan'))
        
        lim_min = data[~torch.isnan(data)].min()
        lim_max = data[~torch.isnan(data)].max()
        
        return lim_min.item(), lim_max.item()

    def make_limit(self):
        if self.fix_method == 'source':
            lim_min, lim_max = self._extract_min_and_max(self.model1)

        elif self.fix_method == 'target':
            lim_min, lim_max = self._extract_min_and_max(self.model2)
            
        elif self.fix_method == 'both':
            lim_min1, lim_max1 = self._extract_min_and_max(self.model1)
            lim_min2, lim_max2 = self._extract_min_and_max(self.model2)
            
            lim_min = min(lim_min1, lim_min2)
            lim_max = max(lim_max1, lim_max2)
        
        else:
            raise ValueError('Please choose the fix method')

        return lim_min, lim_max
    
    def make_histogram(self, data):
    
        lim_min, lim_max = self.make_limit()
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
        
        # sinkhornを動かすコマンド。
        dist = ot.dist(h1_prob.unsqueeze(dim=1), h2_prob.unsqueeze(dim=1))
        res = ot.sinkhorn2(h1_prob, h2_prob, dist, reg = 1)
        
        # 以下は、EMDを動かす際のコマンド。
        # ind1 = self.backend.nx.arange(len(hist1), type_as = hist1).float()
        # ind2 = self.backend.nx.arange(len(hist2), type_as = hist2).float()
        # cost_matrix = torch.cdist(ind1.unsqueeze(dim=1), ind2.unsqueeze(dim=1), p = 1).to(hist1.device)
        # res = ot.emd2(h1_prob, h2_prob, cost_matrix)
        
        return res
    
    def __call__(self, trial, device):
        
        if self.to_types == 'numpy':
            assert device == 'cpu'
        
        self.backend = Backend(device, self.to_types)
        self.model1, self.model2 = self.backend(self.model1, self.model2)
        
        if self.fix_method == 'source':
            alpha = trial.suggest_float("alpha", 1e-6, 1e1, log = True)
            lam = trial.suggest_float("lam", 1e-6, 1e1, log = True)
        
            model1_norm = self.model_normalize(self.model1, 1.0, 1.0) 
            model2_norm = self.model_normalize(self.model2, lam, alpha)
        
        elif self.fix_method == 'target':
            alpha = trial.suggest_float("alpha", 1e-6, 1e1, log = True)
            lam = trial.suggest_float("lam", 1e-6, 1e1, log = True)
            
            model1_norm = self.model_normalize(self.model1, lam, alpha) 
            model2_norm = self.model_normalize(self.model2, 1.0, 1.0)
            
        elif self.fix_method == 'both':
            alpha1 = trial.suggest_float("alpha1", 1e-6, 1e1, log = True)
            lam1   = trial.suggest_float("lam1", 1e-6, 1e1, log = True)
            
            alpha2 = trial.suggest_float("alpha2", 1e-6, 1e1, log = True)
            lam2   = trial.suggest_float("lam2", 1e-6, 1e1, log = True)
            
            model1_norm = self.model_normalize(self.model1, lam1, alpha1) 
            model2_norm = self.model_normalize(self.model2, lam2, alpha2)
        
        else:
            raise ValueError('Please choose the fix method')
        
        
        ot_cost = self.L2_wasserstain(model1_norm, model2_norm)
        
        if ot_cost == 0:
            ot_cost = float('nan')
        
        trial.report(ot_cost, trial.number)
        
        if trial.should_prune():    
            raise optuna.TrialPruned(f"Trial was pruned at iteration {trial.number}")
        
        return ot_cost
    
    def best_parameters(self, study):
        best_trial = study.best_trial
        
        if self.fix_method == 'source':
            a1 = 1
            lam1 = 1
            
            a2 = best_trial.params["alpha"]
            lam2 = best_trial.params["lam"]
        
        elif self.fix_method == 'target':
            a1 = best_trial.params["alpha"]
            lam1 = best_trial.params["lam"]
            
            a2 = 1
            lam2 = 1
        
        elif self.fix_method == 'both':
            a1 = best_trial.params["alpha1"]
            lam1 = best_trial.params["lam1"]
            
            a2 = best_trial.params["alpha2"]
            lam2 = best_trial.params["lam2"]
        
        return a1, lam1, a2, lam2

    def best_models(self, study):
        a1, lam1, a2, lam2 = self.best_parameters(study)
        model1_norm = self.model_normalize(self.model1, lam1, a1) 
        model2_norm = self.model_normalize(self.model2, lam2, a2)
        return model1_norm, model2_norm
    
    def make_graph(self, study):
        
        a1, lam1, a2, lam2 = self.best_parameters(study)
        model1_norm, model2_norm = self.best_models(study)
        
        plt.figure()
        plt.subplot(121)

        model_numpy1 = model1_norm.to('cpu').numpy()
        # np.fill_diagonal(model_numpy1, np.nan)
        
        model_numpy2 = model2_norm.to('cpu').numpy()
        # np.fill_diagonal(model_numpy2, np.nan)
        mappable = ScalarMappable(cmap=plt.cm.jet, norm = colors.Normalize(vmin = 0, vmax = 1))
        
        plt.title('source (a:{:.3g}, lam:{:.3g})'.format(a1, lam1))
        plt.imshow(model_numpy1, cmap=plt.cm.jet)
        plt.colorbar(orientation='horizontal')

        plt.subplot(122)
        plt.title('target (a:{:.3g}, lam:{:.3g})'.format(a2, lam2))
        plt.imshow(model_numpy2, cmap=plt.cm.jet)
        plt.colorbar(orientation='horizontal')

        plt.tight_layout()
        plt.show()
        
        lim_min, lim_max = self.make_limit()
        # bins = 100
        
        # hist1 = torch.histc(model1_norm, bins = bins, min = lim_min, max = lim_max)
        # hist2 = torch.histc(model2_norm, bins = bins, min = lim_min, max = lim_max)
        
        hist1 = self.make_histogram(model1_norm)
        hist2 = self.make_histogram(model2_norm)
        
        x = np.linspace(lim_min, lim_max, len(hist1))
        plt.figure()
        plt.title('histogram')
        plt.bar(x, hist1.to('cpu').numpy(), label = 'source', alpha = 0.6, width = 9e-3)
        plt.bar(x, hist2.to('cpu').numpy(), label = 'target', alpha = 0.6, width = 9e-3)
    
        plt.legend()
        plt.xlabel('dis-similarity value')
        plt.ylabel('count')
        plt.tight_layout()
        plt.show()
        
        
        corr = sp.stats.pearsonr(model1_norm[~torch.isnan(model1_norm)].to('cpu').numpy(),
                                 model2_norm[~torch.isnan(model2_norm)].to('cpu').numpy())
    
        print('peason\'s r (without diag element) =', corr[0])
    
    def eval_study(self, study):
        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_parallel_coordinate(study).show()
        # optuna.visualization.plot_contour(study).show()

# %%
if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    model1 = torch.load('../../data/model1.pt')
    model2 = torch.load('../../data/model2.pt')
    unittest_save_path = '../../results/unittest/adjust_histogram'
    fix_method = 'both'
    device = 'cuda'
    
    # %%
    tt = Adjust_Distribution(model1, model2, unittest_save_path, fix_method = fix_method, device = device) 
    # %%
    study = optuna.create_study(direction = 'minimize',
                                study_name = 'unit_test('+fix_method+')',
                                sampler = optuna.samplers.RandomSampler(seed = 42),
                                pruner = optuna.pruners.MedianPruner(),
                                storage = 'sqlite:///' + unittest_save_path + '/unit_test('+fix_method+').db',
                                load_if_exists = True)

    test_adjust = functools.partial(tt, device = device)
    study.optimize(test_adjust, n_trials = 1500)
    
    # %%
    tt.make_graph(study)
    tt.eval_study(study)  
