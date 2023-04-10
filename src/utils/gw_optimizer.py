# %%
# Standard Library
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Union

# Third Party Library
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import optuna
import pymysql
import seaborn as sns
import torch
from joblib import parallel_backend


# %%
def load_optimizer(save_path, n_jobs = 4, num_trial = 20,
                   to_types = 'torch', method = 'optuna',
                   sampler_name = 'random', pruner_name = 'median',
                   pruner_params = None, n_iter = 10,
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
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # make db path
    if sql_name == 'sqlite':
        storage = "sqlite:///" + save_path +  '/' + filename + '.db'

    elif sql_name == 'mysql':
        if storage == None:
            raise ValueError('mysql path was not set.')
    else:
        raise ValueError('no implemented SQL.')

    if method == 'optuna':
        Opt = RunOptuna(save_path, to_types, storage, filename, sampler_name, pruner_name, pruner_params, n_iter, sql_name, n_jobs, num_trial, delete_study)
    else:
        raise ValueError('no implemented method.')

    return Opt


class RunOptuna():
    def __init__(self, save_path, to_types, storage, filename,               # optunaによる結果の保存先やファイル名の指定
                 sampler_name, pruner_name, pruner_params, n_iter, sql_name, # optunaにおける各種設定
                 n_jobs, num_trial, delete_study                             # optuna.studyに与えるパラメータ
                 ):

        # optunaによる結果の保存先やファイル名の指定
        self.save_path = save_path
        self.to_types = to_types
        self.storage = storage
        self.filename = filename

        # optunaにおける各種設定
        self.sampler_name = sampler_name
        self.pruner_name = pruner_name
        self.pruner_params = pruner_params
        self.sql_name = sql_name

        # optuna.studyに与えるパラメータ
        self.n_jobs = n_jobs
        self.num_trial = num_trial
        self.delete_study = delete_study

        # MedianPruner
        self.n_startup_trials = 5
        self.n_warmup_steps = 5

        # HyperbandPruner
        self.min_resource = 5
        self.reduction_factor = 2
        self.n_iter = n_iter

        if pruner_params is not None:
            self._set_params(pruner_params)

    def _set_params(self, vars_dic: dict) -> None:
        '''
        2023/3/14 阿部
        '''
        for key, value in vars_dic.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f'{key} is not a parameter of the pruner.')

    def _confirm_delete(self) -> None:
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

    def create_study(self, seed = 42):
        study = optuna.create_study(direction = "minimize",
                                    study_name = self.filename,
                                    sampler = self.choose_sampler(seed = seed),
                                    pruner = self.choose_pruner(),
                                    storage = self.storage,
                                    load_if_exists = True)
        return study


    def load_study(self, seed = 42):
        """
        2023.4.3 佐々木
        studyファイルの作成を行う関数。

        Returns:
            _type_: _description_
        """
        db_file_path = self.save_path + '/' + self.filename + '.db'

        if os.path.exists(db_file_path):
            study = optuna.load_study(study_name = self.filename,
                                      sampler = self.choose_sampler(seed = seed),
                                      pruner = self.choose_pruner(),
                                      storage = self.storage)

        else:
            raise ValueError('This db does not exist.')
        return study



    def run_study(self, objective, gpu_board = 'cuda:0', forced_run = True):
        """
        2023.3.29 佐々木

        objective = functools.partial(dataset, init_plans_list = init_mat_types, eps_list = self.eps_list)
        """
        # Substitute partial applicated init_mat_types
        self.init_mat_types = objective.keywords['init_plans_list']
        self.eps_space = self.define_eps_space(objective.keywords['eps_list'], objective.keywords['eps_log'])

        if self.delete_study:
            self._confirm_delete()

        if forced_run:
            self.create_study() #dbファイルがない場合、ここでloadをさせないとmulti_runが正しく動かなくなってしまう。

            def multi_run(objective, seed, num_trials, device):
                tt = functools.partial(objective, device = device)
                study = self.load_study(seed = seed)
                study.optimize(tt, n_trials = num_trials, n_jobs = 1)

            seed = 42

            with ThreadPoolExecutor(self.n_jobs) as pool:
                for i in range(self.n_jobs):
                    if gpu_board == 'multi':
                        device = 'cuda:' + str(i % 4)
                    elif 'cuda' in gpu_board:
                        device = gpu_board
                    elif gpu_board == 'cpu':
                        device = 'cpu'

                    pool.submit(multi_run, objective, seed + i, self.num_trial // self.n_jobs, device)

        study = self.load_study()

        return study

    def choose_sampler(self, seed = 42):
        '''
        2023/3/15 阿部
        TPE Sampler追加
        '''
        if self.sampler_name == 'random':
            sampler = optuna.samplers.RandomSampler(seed)

        elif self.sampler_name == 'grid':

            search_space = {
                "eps": self.eps_space,
                "initialize": self.init_mat_types
            }

            sampler = optuna.samplers.GridSampler(search_space)

        elif self.sampler_name.lower() == 'tpe':
            sampler = optuna.samplers.TPESampler(constant_liar = True, multivariate = True, seed = seed) # 分散最適化のときはTrueにするのが良いらしい(阿部)

        else:
            raise ValueError('not implemented sampler yet.')

        return sampler

    def choose_pruner(self):
        '''
        2023/3/15 阿部
        Median PrunerとHyperbandPrunerを追加
        (RandomSampler, MedianPruner)か(TPESampler, HyperbandPruner)がbestらしい
        '''
        if self.pruner_name == 'median':
            pruner = optuna.pruners.MedianPruner(n_startup_trials = self.n_startup_trials, n_warmup_steps = self.n_warmup_steps)
        elif self.pruner_name.lower() == 'hyperband':
            pruner = optuna.pruners.HyperbandPruner(min_resource = self.min_resource, max_resource = self.n_iter, reduction_factor = self.reduction_factor)
        elif self.pruner_name.lower() == 'nop':
            pruner = optuna.pruners.NopPruner()
        else:
            raise ValueError('not implemented pruner yet.')
        return pruner

    def define_eps_space(self, eps_list: list, eps_log: bool):
        '''
        2023/4/8 abe
        grid samplerにepsilonのrangeを渡す関数を追加
        '''
        if len(eps_list) == 2:
            ep_lower, ep_upper = eps_list
            if eps_log:
                eps_space = np.logspace(ep_lower, ep_upper)
            else:
                eps_space = np.linspace(ep_lower, ep_upper)

        elif len(eps_list) == 3:
            ep_lower, ep_upper, ep_step = eps_list
            eps_space = np.arange(ep_lower, ep_upper, ep_step)

        else:
            raise ValueError("The eps_list doesn't match.")

        return eps_space
