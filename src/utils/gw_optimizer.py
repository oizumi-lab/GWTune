# %%
import os
import numpy as np
import torch
import pymysql
import matplotlib.pyplot as plt
import optuna
from joblib import parallel_backend
import matplotlib.style as mplstyle
import seaborn as sns


# %%
def load_optimizer(save_path, n_jobs = 10, num_trial = 50,
                   to_types = 'torch', method = 'optuna', 
                   init_plans_list = ['diag'], eps_list = [1e-4, 1e-2], eps_log = True, 
                   sampler_name = 'random', pruner_name = 'median', 
                   filename = 'test', sql_name = 'sqlite', storage = None,
                   delete_study = False):
    
    """
    (usage example)
    >>> dataset = mydataset()
    >>> opt = load_optimizer(save_path)
    >>> study = Opt.run_study(dataset)

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    # make file_path
    file_path = save_path + '/' + filename
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    # make db path
    if sql_name == 'sqlite':
        storage = "sqlite:///" + file_path +  '/' + filename + '.db',
    
    elif sql_name == 'mysql':
        if storage == None:
            raise ValueError('mysql path was not set.')
    else:
        raise ValueError('no implemented SQL.')

    if method == 'optuna':
        Opt = RunOptuna(file_path, to_types, storage, filename, sampler_name, pruner_name, sql_name, init_plans_list, eps_list, eps_log, n_jobs, num_trial, delete_study)
    else:
        raise ValueError('no implemented method.')

    return Opt


class RunOptuna():
    def __init__(self, 
                 file_path, 
                 to_types, 
                 storage, 
                 filename, 
                 sampler_name, 
                 pruner_name,
                 sql_name,
                 init_plans_list,
                 eps_list, 
                 eps_log,
                 n_jobs,
                 num_trial,
                 delete_study
                 ):
        
        self.file_path = file_path
        self.to_types = to_types
        self.storage = storage
        self.filename = filename
        self.sampler_name = sampler_name
        self.pruner_name = pruner_name
        self.sql_name = sql_name
        self.init_plans_list = init_plans_list
        self.eps_list = eps_list
        self.eps_log = eps_log
        self.n_jobs = n_jobs
        self.num_trial = num_trial
        self.delete_study = delete_study

        self.initialize = ['uniform', 'random', 'permutation', 'diag'] # 実装済みの方法の名前を入れる。
        self.init_mat_types = self._choose_init_plans(self.init_plans_list) # リストを入力して、実行可能な方法のみをリストにして返す。

    def run_study(self, dataset):

        if self.delete_study:
            while True:
                confirmation = input(f"This code will delete the study named '{self.filename}'.\nDo you want to execute the code? (y/n)")
                if confirmation == 'y':
                    try:
                        optuna.delete_study(storage = self.storage, study_name = self.filename)
                        print(f"delete the study '{self.filename}'!")
                        break
                    except:
                        print(f"study '{self.filename}' does not exist.")
                        break
                elif confirmation == 'n':
                    raise ValueError("If you don't want to delete study, use 'delete_study = False'.")
                else:
                    print("Invalid input. Please enter again.")

        study = optuna.create_study(direction = "minimize",
                                    study_name = self.filename,
                                    sampler = self.choose_sampler(),
                                    pruner = self.choose_pruner(dataset),
                                    storage = self.storage,
                                    load_if_exists = True)

        with parallel_backend("multiprocessing", n_jobs = self.n_jobs):
            study.optimize(lambda trial: dataset(trial, self.file_path, self.init_mat_types, self.eps_list, self.eps_log), n_trials = self.num_trial, n_jobs = self.n_jobs)

        return study


    def get_study(self, dataset):
        study = optuna.create_study(direction = "minimize",
                                        study_name = self.filename,
                                        storage = self.storage,
                                        load_if_exists = True)
        return study


    def _choose_init_plans(self, init_plans_list):
        """
        ここから、初期値の条件を1個または複数個選択することができる。
        選択はself.initializeの中にあるものの中から。
        選択したい条件が1つであっても、リストで入力をすること。

        Args:
            init_plans_list (list) : 初期値の条件を1個または複数個入れたリスト。

        Raises:
            ValueError: 選択したい条件が1つであっても、リストで入力をすること。

        Returns:
            list : 選択希望の条件のリスト。
        """

        if type(init_plans_list) != list:
            raise ValueError('variable named "init_plans_list" is not list!')

        else:
            return [v for v in self.initialize if v in init_plans_list]

    def choose_sampler(self):
        '''
        2023/3/15 阿部
        TPE Sampler追加
        '''
        if self.sampler_name == 'random':
            sampler = optuna.samplers.RandomSampler(seed = 42)

        elif self.sampler_name == 'grid_search':

            search_space = {
                "eps": np.logspace(-4, -2),
                "initialize": self.init_mat_types
            }

            sampler = optuna.samplers.GridSampler(search_space)

        elif self.sampler_name.lower() == 'tpe':
            sampler = optuna.samplers.TPESampler(constant_liar = True, multivariate = True, seed = 42) # 分散最適化のときはTrueにするのが良いらしい(阿部)

        else:
            raise ValueError('not implemented sampler yet.')

        return sampler

    def choose_pruner(self, dataset):
        '''
        2023/3/15 阿部
        Median PrunerとHyperbandPrunerを追加
        (RandomSampler, MedianPruner)か(TPESampler, HyperbandPruner)がbestらしい
        '''
        if self.pruner_name == 'median':
            pruner = optuna.pruners.MedianPruner(n_startup_trials = dataset.n_startup_trials, n_warmup_steps = dataset.n_warmup_steps)
        elif self.pruner_name.lower() == 'hyperband':
            pruner = optuna.pruners.HyperbandPruner(min_resource = dataset.min_resource, max_resource = dataset.n_iter, reduction_factor = dataset.reduction_factor)
        elif self.pruner_name.lower() == 'nop':
            pruner = optuna.pruners.NopPruner()
        else:
            raise ValueError('not implemented pruner yet.')
        return pruner

    
# %%
