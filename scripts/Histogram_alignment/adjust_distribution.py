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

# %%
from src.mydataset import GWD_Dataset
from src.utils import torch_fix_seed
warnings.simplefilter("ignore")
# optuna.logging.set_verbosity(optuna.logging.WARNING)

torch_fix_seed()

# %%
class Adjust_Distribution():
    def __init__(self, model1, model2, gpu_queue = None, random_dataset = True):
        self.model1 = model1
        self.model2 = model2
        self.gpu_queue = gpu_queue
        self.random_dataset = random_dataset
        
        self.adjust_path = "./GWD/Test/adjust"
        self.test_path = "./GWD/Test/"
        
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
        
        res = ot.emd2(h1_prob, h2_prob, cost_matrix, center_dual = False)
        # res = ot.sinkhorn2(h1_prob, h2_prob, cost_matrix, reg = 1)
        
        # res = torch.sum(torch.mm(WassDists, cost_matrix))
            
        if res == 0:
            res = float('nan')
            
        
        return res

    def run_study(self, filename, n_gpu = 10, num_trial = 5000):
        
        save_file_name = "adjust (" + filename + ").db"
        
        # search_space = {
        #     'alpha1': np.logspace(-1, 1, num = num_alpha1),
        #     'lam1': np.logspace(-6, 0, num = num_lam1),
        # }
        
        # if not os.path.exists(self.adjust_path  + "/" + save_file_name):
        study = optuna.create_study(direction = "minimize",
                                    study_name = 'adjust',
                                    # sampler = optuna.samplers.GridSampler(search_space),
                                    sampler = optuna.samplers.RandomSampler(seed = 42),
                                    # sampler = optuna.samplers.TPESampler(seed = 42),
                                    # storage = "sqlite:///" + self.adjust_path + "/" + save_file_name,
                                    load_if_exists = False)
        
        torch_fix_seed()
        # with parallel_backend("multiprocessing", n_jobs = n_gpu):
        study.optimize(self, n_trials = num_trial)

        # else:
        #     study = optuna.create_study(direction = "minimize",
        #                                 study_name = 'adjust',
        #                                 storage = "sqlite:///" + self.adjust_path + "/" + save_file_name,
        #                                 load_if_exists = True)
        
        return study
    
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
        plt.title('histgram')
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
    test = GWD_Dataset(train_size = 10000, random_dataset = True, device = 'cuda:3')
    model1, model2, p, q, filename = test.extract(sort_dataset=True)
    
    # %%
    tt = Adjust_Distribution(model1, model2)
    
    # ss = tt.model_normalize(model1, 0.00041, 4.5)
    
    # tt.make_histogram(ss)
    
    # tt.L2_wasserstain(ss, model2)
 
    
    # %%
    filename = 'run_test'
    study = tt.run_study(filename)
    # %%
    tt.make_graph(study)
    # %%
    tt.eval_study(study)
    # %%
    
    
