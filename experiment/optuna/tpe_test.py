# %%
import optuna
import torch
import torch.multiprocessing as mp
import numpy as np
torch.manual_seed(42)
from concurrent.futures import ThreadPoolExecutor

# %%
def objective(trial, test_arr):
    x = trial.suggest_float("x", -10, 10)    
    return sum(test_arr * x)
 
study = optuna.create_study(study_name="my_study", storage = "sqlite:///cuda_test.db", load_if_exists = True)


def multi_run(seed, i, n_trials):
    device = 'cuda:0'#+str(i)
    test_arr = torch.randn(10).to(device)
    sampler = optuna.samplers.TPESampler(seed = seed + i)
    loaded_study = optuna.load_study(sampler = sampler, study_name="my_study", storage="sqlite:///cuda_test.db")
    loaded_study.optimize(lambda trial: objective(trial, test_arr), n_trials = n_trials)

processes = []
n_jobs = 2
n_trials = 4
seed = 42

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

def batched_objective(xs: np.ndarray, ys: np.ndarray):
    return xs**2 + ys

batch_size = 10
study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())

for _ in range(3):

    # create batch
    trial_numbers = []
    x_batch = []
    y_batch = []
    for _ in range(batch_size):
        trial = study.ask()
        trial_numbers.append(trial.number)
        x_batch.append(trial.suggest_float("x", -10, 10))
        y_batch.append(trial.suggest_float("y", -10, 10))

    # evaluate batched objective
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    objectives = batched_objective(x_batch, y_batch)

    # finish all trials in the batch
    for trial_number, objective in zip(trial_numbers, objectives):
        study.tell(trial_number, objective)