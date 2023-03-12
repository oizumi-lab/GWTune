# %%
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import optuna
from joblib import parallel_backend
import matplotlib.style as mplstyle
import seaborn as sns

class Optimizer:
    def __init__(self, save_path) -> None:
        self.save_path = save_path
        pass
    
    def optimizer(self, 
                  dataset, 
                  method = 'optuna',
                  init_plans_list = ['diag'],
                  eps_list = [1e-4, 1e-2],
                  sampler_name = 'random',
                  filename = 'test',
                  n_jobs = 10,
                  num_trial = 50):
        """_summary_

        Args:
            dataset (_type_): _description_
            method (str, optional): _description_. Defaults to 'optuna'.
            init_plans_list (list, optional): _description_. Defaults to ['diag'].
            sampler_name (str, optional): _description_. Defaults to 'random'.
            filename (str, optional): _description_. Defaults to 'test'.
            n_jobs (int, optional): _description_. Defaults to 10.
            num_trial (int, optional): _description_. Defaults to 50.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if method == 'optuna':
            Opt = RunOptuna(self.save_path, filename, sampler_name, init_plans_list, eps_list, n_jobs, num_trial)
        else:
            raise ValueError('no implemented method.')

        study = Opt.run_study(dataset)

        Opt.load_graph(study)

        return study

class RunOptuna():
    def __init__(self, save_path, filename, sampler_name, init_plans_list, eps_list, n_jobs, num_trial):
        self.save_path = save_path
        self.filename = filename
        self.sampler_name = sampler_name
        self.init_plans_list = init_plans_list
        self.eps_list = eps_list
        self.n_jobs = n_jobs
        self.num_trial = num_trial


    def run_study(self, dataset):
        if not os.path.exists(self.save_path + '/' + self.filename):
            study = optuna.create_study(direction = "minimize",
                                        study_name = self.filename,
                                        sampler = self.choose_sampler(),
                                        storage = "sqlite:///" + self.save_path + "/" + self.filename + '.db',
                                        load_if_exists = True)

            with parallel_backend("multiprocessing", n_jobs = self.n_jobs):
                study.optimize(lambda trial: dataset(trial, self.init_plans_list, self.eps_list), n_trials = self.num_trial, n_jobs = self.n_jobs)

        else:
            study = optuna.create_study(direction = "minimize",
                                        study_name = self.filename,
                                        storage = "sqlite:///" + self.save_path + "/"  + self.filename + '.db',
                                        load_if_exists = True)
        return study

    def choose_sampler(self):

        if self.sampler_name == 'random':
            sampler = optuna.samplers.RandomSampler(seed = 42)

        elif self.sampler_name == 'grid_search':
            sampler = optuna.samplers.GridSampler(seed = 42)

        else:
            raise ValueError('not implemented sampler yet.')

        return sampler

    def load_graph(self, study):
        best_trial = study.best_trial
        eps = best_trial.params['eps']
        acc = best_trial.user_attrs['acc']
        size = best_trial.user_attrs['size']
        number = best_trial.number

        gw = torch.load(self.save_path + '/GW({} pictures, epsilon = {}, trial = {}).pt'.format(size, round(eps, 6), number))

        self.plot_coupling(gw, eps, acc)

    def make_eval_graph(self, study):
        df_test = study.trials_dataframe()
        success_test = df_test[df_test['values_1'] != float('nan')]
        success_test = success_test[success_test['values_0'] != float('nan')]
        
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
        sns.heatmap(T.to('cpu').numpy())

        plt.title('GW results ({} pictures, eps={}, acc.= {})'.format(N, round(epsilon, 6), round(acc, 4)))
        plt.tight_layout()
        plt.show()
