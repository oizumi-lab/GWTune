# %%
# Standard Library
import functools
import os
from concurrent.futures import ThreadPoolExecutor
import warnings

# Third Party Library
import numpy as np
import torch
import torch.multiprocessing as mp
import pymysql
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import optuna
import pymysql
import seaborn as sns
import torch
from joblib import parallel_backend


# %%
def load_optimizer(
    save_path,
    n_jobs=1,
    num_trial=20,
    to_types="torch",
    method="optuna",
    sampler_name="random",
    pruner_name="median",
    pruner_params=None,
    n_iter=10,
    filename="test",
    storage=None,
    delete_study=False,
):

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

    if method == "optuna":
        Opt = RunOptuna(
            save_path,
            to_types,
            storage,
            filename,
            sampler_name,
            pruner_name,
            pruner_params,
            n_iter,
            n_jobs,
            num_trial,
            delete_study,
        )
    else:
        raise ValueError("no implemented method.")

    return Opt


class RunOptuna:
    def __init__(
        self,
        save_path,
        to_types,
        storage,
        filename,  # optunaによる結果の保存先やファイル名の指定
        sampler_name,
        pruner_name,
        pruner_params,
        n_iter,
        n_jobs,  # optunaにおける各種設定
        num_trial,
        delete_study,  # optuna.studyに与えるパラメータ
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
        """
        2023/3/14 阿部
        """
        for key, value in vars_dic.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"{key} is not a parameter of the pruner.")

    def _confirm_delete(self) -> None:
        while True:
            confirmation = input(
                f"This code will delete the study named '{self.filename}'.\nDo you want to execute the code? (y/n)"
            )
            if confirmation == "y":
                try:
                    optuna.delete_study(storage=self.storage, study_name=self.filename)
                    print(f"delete the study '{self.filename}'!")
                    break
                except:
                    print(f"study '{self.filename}' does not exist.")
                    break
            elif confirmation == "n":
                raise ValueError("If you don't want to delete study, use 'delete_study = False'.")
            else:
                print("Invalid input. Please enter again.")

    def create_study(self):
        study = optuna.create_study(
            direction="minimize", study_name=self.filename, storage=self.storage, load_if_exists=True
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
            study_name=self.filename,
            sampler=self.choose_sampler(seed=seed),
            pruner=self.choose_pruner(),
            storage=self.storage,
        )
        return study

    def _run_study(self, objective, device="cpu", seed=42, forced_run=True):
        """_summary_

        Args:
            objective (_type_): _description_
            device (str, optional): _description_. Defaults to "cpu".
            seed (int, optional): _description_. Defaults to 42.
            forced_run (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        if self.delete_study:
            self._confirm_delete()

        if forced_run:
            self.create_study()  # dbファイルがない場合、ここでloadをさせないとmulti_runが正しく動かなくなってしまう。

            def _local_runner(objective, num_trials, device, seed, worker_id=0):
                tt = functools.partial(objective, device=device)
                study = self.load_study(seed=seed + worker_id)
                study.optimize(tt, n_trials=num_trials)

            if self.sampler_name.lower() == "tpe":
                assert self.n_jobs == 1, "TPE-Sampler does not work in a proper way if n_jobs > 1."
                _local_runner(objective, self.num_trial, device, seed)

            elif self.sampler_name.lower() == "random" or self.sampler_name.lower() == "grid":
                if self.n_jobs == 1:
                    _local_runner(objective, self.num_trial, device, seed)

                elif self.n_jobs == -1:
                    raise ValueError(
                        "Do not use n_jobs = -1 in this library, please use 'os.cpu_count()' instead of it."
                    )

                elif self.n_jobs > 1:
                    if self.to_types == "numpy":
                        warnings.warn(
                            "parallel computation may be slower than single computation for numpy...", UserWarning
                        )
                    worker_arr = np.array_split(np.arange(self.num_trial), self.n_jobs)

                    with ThreadPoolExecutor(self.n_jobs) as pool:
                        for i in range(self.n_jobs):
                            if device == "multi":
                                device = "cuda:" + str(i % 4)

                            worker_trial = len(worker_arr[i])
                            pool.submit(_local_runner, objective, worker_trial, device, seed, worker_id=i)

        study = self.load_study()

        return study

    def run_study(self, objective, device, forced_run=True, **kwargs):
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

        study = self._run_study(objective, device=device, forced_run=forced_run)

        return study

    def choose_sampler(self, seed=42):
        """
        2023/3/15 阿部
        TPE Sampler追加
        """
        if self.sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed)

        elif self.sampler_name == "grid":
            sampler = optuna.samplers.GridSampler(self.search_space, seed=seed)

        elif self.sampler_name.lower() == "tpe":
            sampler = optuna.samplers.TPESampler(
                constant_liar=True, multivariate=True, seed=seed
            )  # 分散最適化のときはTrueにするのが良いらしい(阿部)

        else:
            raise ValueError("not implemented sampler yet.")

        return sampler

    def choose_pruner(self):
        """
        2023/3/15 abe
        Median PrunerとHyperbandPrunerを追加
        (RandomSampler, MedianPruner)か(TPESampler, HyperbandPruner)がbestらしい
        """
        if self.pruner_name == "median":
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_warmup_steps
            )
        elif self.pruner_name.lower() == "hyperband":
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=self.min_resource, max_resource=self.n_iter, reduction_factor=self.reduction_factor
            )
        elif self.pruner_name.lower() == "nop":
            pruner = optuna.pruners.NopPruner()
        else:
            raise ValueError("not implemented pruner yet.")
        return pruner

    def define_eps_space(self, eps_list: list, eps_log: bool, num_trial: int):
        """
        2023/4/8 abe
        grid samplerにepsilonのrangeを渡す関数を追加
        """
        if len(eps_list) == 2:
            ep_lower, ep_upper = eps_list
            if eps_log:
                eps_space = np.logspace(
                    np.log10(ep_lower), np.log10(ep_upper), num=num_trial
                )  # defaultだと50個の分割になる(numpyのtutorialより)。
            else:
                eps_space = np.linspace(ep_lower, ep_upper, num=num_trial)

        elif len(eps_list) == 3:
            ep_lower, ep_upper, ep_step = eps_list
            eps_space = np.arange(ep_lower, ep_upper, ep_step)

        else:
            raise ValueError("The eps_list doesn't match.")

        return eps_space
