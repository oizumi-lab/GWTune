# %%
import optuna
import torch
import torch.multiprocessing as mp
import numpy as np
torch.manual_seed(42)
from concurrent.futures import ThreadPoolExecutor


def objective(trial, test_arr):
    x = trial.suggest_float("x", -10, 10)    
    return sum(test_arr * x)
 
study = optuna.create_study(study_name="my_study", storage = "sqlite:///cuda_test.db", load_if_exists = True)


def multi_run(seed, i, n_trials):
    device = 'cuda:0'#+str(i)
    test_arr = torch.randn(10).to(device)

    # sampler = optuna.samplers.RandomSampler(seed = seed + i)
    sampler = optuna.samplers.GridSampler(search_space, seed = seed + i)
    loaded_study = optuna.load_study(sampler = sampler, study_name="my_study", storage="sqlite:///cuda_test.db")
    loaded_study.optimize(lambda trial: objective(trial, test_arr), n_trials = n_trials)

processes = []
n_jobs = 2
n_trials = 4
seed = 42

search_space = {'x' : np.logspace(0, 1, num = n_trials)}

# %%
# プロセスを生成する
processes = []
for i in range(n_jobs):
    p = mp.Process(target=multi_run, args=(seed, i, n_trials // n_jobs))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

# %%
