# %%
import os
import optuna
import torch.multiprocessing as mp

import torch 


torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

test_arr = torch.randn(10).to('cuda')

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return sum(test_arr * x)
 
study = optuna.create_study(study_name="my_study", storage = "sqlite:///cuda_test.db", load_if_exists = True)


def multi_run(dataset, seed):
    sampler = optuna.samplers.RandomSampler(seed = seed)
    loaded_study = optuna.load_study(sampler = sampler, study_name="my_study", storage="sqlite:///cuda_test.db")
    loaded_study.optimize(dataset, n_trials = 10, n_jobs = 1)

processes = []

n_jobs = 4
seed = 42

# mp.set_start_method('spawn')

for i in range(n_jobs):
    p = mp.Process(target = multi_run, args=(objective, seed + i))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

# %%