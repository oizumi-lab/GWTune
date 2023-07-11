# %%
# Standard Library
import functools
import os
import warnings

# Third Party Library
import numpy as np
import torch
import pymysql
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import optuna
from sqlalchemy_utils import create_database, database_exists


# %%
def load_optimizer(
    save_path=None,
    filename=None,
    storage=None,
    init_mat_plan="random",
    n_iter=10,
    num_trial=20,
    n_jobs=1,
    method="optuna",
    sampler_name="random",
    pruner_name="median",
    pruner_params=None,
):

    """
    (usage example)
    >>> dataset = mydataset()
    >>> opt = load_optimizer(filename)
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

    # create a database from the URL
    if not database_exists(storage):
        create_database(storage)

    if method == "optuna":
        Opt = RunOptuna(
            filename, 
            storage, 
            init_mat_plan,
            num_trial,
            n_iter, 
            n_jobs, 
            sampler_name, 
            pruner_name, 
            pruner_params, 
        )
    else:
        raise ValueError("no implemented method.")

    return Opt


class RunOptuna:
    def __init__(
        self,
        filename, 
        storage, 
        init_mat_plan,
        num_trial,
        n_iter, 
        n_jobs, 
        sampler_name, 
        pruner_name, 
        pruner_params, 
    ):

        # the path or file name to save the results.
        self.filename = filename
        self.storage = storage
        self.init_mat_plan = init_mat_plan
        
        # parameters for optuna.study
        self.num_trial = num_trial
        self.n_iter = n_iter
        self.n_jobs = n_jobs

        # setting of optuna
        self.sampler_name = sampler_name
        self.pruner_name = pruner_name
        self.pruner_params = pruner_params


        # parameters for optuna.study
        self.n_jobs = n_jobs
        self.num_trial = num_trial

        
        # parameters for optuna.study
        self.n_jobs = n_jobs
        self.num_trial = num_trial

        # MedianPruner
        self.n_startup_trials = 5
        self.n_warmup_steps = 5

        # HyperbandPruner
        self.min_resource = 5
        self.reduction_factor = 2

        if pruner_params is not None:
            self._set_params(pruner_params)

    def _set_params(self, vars_dic: dict) -> None:
        """
        2023/3/14 阿部
        """
        for key, value in vars_dic.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"{key} is not a parameter of the pruner.")

    def create_study(self, direction="minimize"):
        study = optuna.create_study(
            direction=direction,
            study_name=self.filename + "_" + self.init_mat_plan,
            storage=self.storage,
            load_if_exists=True,
        )
        return study

    def load_study(self, seed=42):
        """
        2023.4.3 佐々木
        studyファイルの作成を行う関数。

        Returns:
            _type_: _description_
        """
        study = optuna.load_study(
            study_name=self.filename + "_" + self.init_mat_plan,
            sampler=self.choose_sampler(seed=seed),
            pruner=self.choose_pruner(),
            storage=self.storage,
        )
        return study

    def run_study(self, objective, device, seed=42, **kwargs):
        """
        2023.3.29 佐々木
        """

        if self.sampler_name == "grid":
            assert kwargs.get("search_space") != None, "please define search space for grid search."
            self.search_space = kwargs.pop("search_space")

        else:
            if kwargs.get("search_space") is not None:
                warnings.warn("except for grid search, search space is ignored.", UserWarning)
            del kwargs["search_space"]

        objective = functools.partial(objective, **kwargs)

        # If there is no db file, multi_run will not work properly if you don't let it load here.
        # PyMySQL implementation will be here if necessary.

        try:
            study = self.load_study(seed=seed)
        except KeyError:
            print("Study for " + self.filename + "_" + self.init_mat_plan + " was not found, creating a new one...")
            self.create_study()
            study = self.load_study(seed=seed)
        
        objective_device = functools.partial(objective, device=device)

        if self.n_jobs > 1:
            warnings.filterwarnings("always")
            warnings.warn(
                "The parallel computation is done by the functions implemented in Optuna.\n \
                This doesn't always provide a benefit to speed up or to get a better results.",
                UserWarning,
            )
            warnings.filterwarnings("ignore")

        study.optimize(objective_device, self.num_trial, n_jobs=self.n_jobs)

        return study

    def choose_sampler(self, seed=42, constant_liar=False, multivariate=False):
        """
        2023/3/15 Abe
        added TPE Sampler
        """
        if self.sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed)

        elif self.sampler_name == "grid":
            sampler = optuna.samplers.GridSampler(self.search_space, seed=seed)

        elif self.sampler_name.lower() == "tpe":
            sampler = optuna.samplers.TPESampler(
                constant_liar=constant_liar,  # I heard it is better to set to True for distributed optimization (Abe)
                multivariate=multivariate,
                seed=seed,
            )

        else:
            raise ValueError("not implemented sampler yet.")

        return sampler

    def choose_pruner(self):
        """
        2023/3/15 abe
        Added Median Pruner and HyperbandPruner
        (RandomSampler, MedianPruner) or (TPESampler, HyperbandPruner) seems to be the best
        """
        if self.pruner_name == "median":
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.n_startup_trials, 
                n_warmup_steps=self.n_warmup_steps
            )
        elif self.pruner_name.lower() == "hyperband":
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=self.min_resource, 
                max_resource=self.n_iter, 
                reduction_factor=self.reduction_factor
            )
        elif self.pruner_name.lower() == "nop":
            pruner = optuna.pruners.NopPruner()
        else:
            raise ValueError("not implemented pruner yet.")
        return pruner

    def define_eps_space(self, eps_list: list, eps_log: bool, num_trial: int):
        """
        2023/4/8 abe
        Add function to pass epsilon range to grid sampler
        """
        if len(eps_list) == 2:
            ep_lower, ep_upper = eps_list
            if eps_log:
                eps_space = np.logspace(np.log10(ep_lower), np.log10(ep_upper), num=num_trial)
            else:
                eps_space = np.linspace(ep_lower, ep_upper, num=num_trial)

        elif len(eps_list) == 3:
            ep_lower, ep_upper, ep_step = eps_list
            eps_space = np.arange(ep_lower, ep_upper, ep_step)

        else:
            raise ValueError("The eps_list doesn't match.")

        return eps_space
