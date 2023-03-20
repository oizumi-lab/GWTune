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

from backend import Backend

# %%
class Adjust_Distribution():
    def __init__(self, model1, model2, adjust_path, fix_method = 'pred', device = 'cpu', to_types = 'torch', gpu_queue = None):
        self.adjust_path = adjust_path
        self.size = len(model1)
        self.device = device
        self.to_types = to_types
        
        self.fix_method = fix_method
        
        self.gpu_queue = gpu_queue
        
        self.backend = Backend(device, to_types)
        self.model1, self.model2 = self.backend(model1, model2)
    
        if not os.path.exists(self.adjust_path):
            os.makedirs(self.adjust_path)
    
    def make_limit(self):
        if self.fix_method == 'target':
            m1 = self.model2.detach().clone()
            m1 = m1.fill_diagonal_(float('nan'))
            
            lim_min = m1[~torch.isnan(m1)].min()
            lim_max = m1[~torch.isnan(m1)].max()


        elif self.fix_method == 'target':
            m2 = self.model2.detach().clone()
            m2 = m2.fill_diagonal_(float('nan'))
            
            lim_min = m2[~torch.isnan(m2)].min()
            lim_max = m2[~torch.isnan(m2)].max()
        
        elif self.fix_method == 'both':
            m1 = self.model1.detach().clone()
            m2 = self.model2.detach().clone()
            
            m1 = m1.fill_diagonal_(float('nan'))
            m2 = m2.fill_diagonal_(float('nan'))
            
            lim_min = min(m1[~torch.isnan(m1)].min(), m2[~torch.isnan(m2)].min())
            lim_max = max(m1[~torch.isnan(m1)].max(), m2[~torch.isnan(m2)].max())
        
        else:
            raise ValueError('Please choose the fix method')

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
            gpu_id = None
            de = self.device
        else:
            gpu_id = self.gpu_queue.get()
            de = 'cuda:' + str(gpu_id % 4) 
        
        if self.to_types == 'numpy':
            assert self.gpu_queue is None
        
        self.model1, self.model2 = self.backend.change_device(de, self.model1, self.model2)
        
        alpha = trial.suggest_float("alpha", 1e-6, 1e1, log = True)
        lam = trial.suggest_float("lam", 1e-6, 1e1, log = True)
        
        if self.fix_method == 'pred':
            model1_norm = self.model_normalize(self.model1, lam, alpha) 
            model2_norm = self.model_normalize(self.model2, 1.0, 1.0)
        
        elif self.fix_method == 'target':
            model1_norm = self.model_normalize(self.model1, 1.0, 1.0) 
            model2_norm = self.model_normalize(self.model2, lam, alpha)
        
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
        
        if res == 0:
            res = float('nan')
        
        trial.report(res, trial.number)
        
        if trial.should_prune():
            if self.gpu_queue is not None:
                self.gpu_queue.put(gpu_id)
            raise optuna.TrialPruned(f"Trial was pruned at iteration {trial.number}")
        
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
    
        ind1 = self.backend.nx.arange(len(hist1), type_as = hist1)
        ind2 = self.backend.nx.arange(len(hist2), type_as = hist2)
        
        cost_matrix = torch.cdist(ind1.unsqueeze(dim=1), ind2.unsqueeze(dim=1), p = 1).to(hist1.device)
        
        res = ot.emd2(h1_prob, h2_prob, cost_matrix, center_dual = False)
        # res = ot.sinkhorn2(h1_prob, h2_prob, cost_matrix, reg = 1)
        
        return res
    
    def best_parameters(self, study):
        a1, lam1, a2, lam2 = 0
        return a1, lam1, a2, lam2
    
    def make_graph(self, study):
        
        a1, lam1, a2, lam2 = self.best_parameters(study)
        model1_norm = self.model_normalize(self.model1, lam1, a1) 
        model2_norm = self.model_normalize(self.model2, lam2, a2)
    
        plt.figure()
        plt.subplot(121)

        model_numpy1 = model1_norm.to('cpu').numpy()
        # np.fill_diagonal(model_numpy1, np.nan)
        
        model_numpy2 = model2_norm.to('cpu').numpy()
        # np.fill_diagonal(model_numpy2, np.nan)
        mappable = ScalarMappable(cmap=plt.cm.jet, norm = colors.Normalize(vmin=0, vmax=1))
        
        plt.title('model1 (a:{:.3g}, lam:{:.3g})'.format(a1, lam1))
        plt.imshow(model_numpy1, cmap=plt.cm.jet)
        plt.colorbar(orientation='horizontal')

        plt.subplot(122)
        plt.title('model2 (a:{:.3g}, lam:{:.3g})'.format(a2, lam2))
        plt.imshow(model_numpy2, cmap=plt.cm.jet)
        plt.colorbar(orientation='horizontal')

        plt.tight_layout()
        plt.show()
        
        lim_max, lim_min = self.make_limit()
        # bins = 100
        
        # hist1 = torch.histc(model1_norm, bins = bins, min = lim_min, max = lim_max)
        # hist2 = torch.histc(model2_norm, bins = bins, min = lim_min, max = lim_max)
        
        hist1 = self.make_histogram(model1_norm)
        hist2 = self.make_histogram(model2_norm)
        
        x = np.linspace(lim_min, lim_max, len(hist1))
        plt.figure()
        plt.title('histogram')
        plt.bar(x, hist1.to('cpu').numpy(), label = 'model1 : AlexNet', alpha = 0.6, width = 9e-3)
        plt.bar(x, hist2.to('cpu').numpy(), label = 'model2 : VGG19', alpha = 0.6, width = 9e-3)
    
        plt.legend()
        plt.xlabel('dis-similarity value')
        plt.ylabel('count')
        plt.tight_layout()
        plt.show()
        
        
        corr = stats.pearsonr(model1_norm[~torch.isnan(model1_norm)].to('cpu').numpy(),
                              model2_norm[~torch.isnan(model2_norm)].to('cpu').numpy())
    
        print('peason\'s r (without diag element) =', corr[0])

    def make_eval_graph(self, study):
        df_test = study.trials_dataframe()
        success_test = df_test[df_test['values_1'] != -1]
        success_test = success_test[success_test['values_0'] > 1e-7]
        
        plt.figure()
        plt.title('The evaluation of GW results for random pictures')
        plt.scatter(success_test['values_1'], np.log(success_test['values_0']), label = 'init diag plan ('+str(self.train_size)+')', c = 'C0')
        plt.xlabel('accuracy')
        plt.ylabel('log(GWD)')
        plt.legend()
        plt.show()
    
    def eval_study(self, study):
        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_parallel_coordinate(study).show()
        # optuna.visualization.plot_contour(study).show()

# %%
if __name__ == '__main__':
    import optuna
    
    os.chdir(os.path.dirname(__file__))
    model1 = torch.load('../../data/model1.pt')#.to('cuda')
    model2 = torch.load('../../data/model2.pt')#.to('cuda')
    unittest_save_path = '../../results/unittest/adjust_histogram'
    tt = Adjust_Distribution(model1, model2, unittest_save_path)
    
    fix_method = 'pred'

    study = optuna.create_study(direction = "minimize",
                                study_name = "test",
                                sampler = optuna.samplers.TPESampler(seed = 42),
                                pruner = optuna.pruners.MedianPruner(),
                                storage = 'sqlite:///' + unittest_save_path + '/test.db', #この辺のパス設定は一度議論した方がいいかも。
                                load_if_exists = True)

    study.optimize(lambda trial: tt(trial, fix_method), n_trials = 20, n_jobs = 10)
    
