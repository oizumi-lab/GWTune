# %%
import os
import optuna
import multiprocessing as mp

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return x ** 2
 
sampler = optuna.samplers.RandomSampler(seed = 42)
study = optuna.create_study(sampler=sampler, storage = "sqlite:///test.db")
 
n_jobs = 4
def multi_run(dataset):
    study.optimize(dataset, n_trials = 10)

processes = []

for i in range(n_jobs):
    p = mp.Process(target = multi_run, args=(objective, ))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
# %%
from joblib import parallel_backend
with parallel_backend("multiprocessing", n_jobs = 4):
    study.optimize(objective, n_trials = 10, n_jobs = 4)
# %%
