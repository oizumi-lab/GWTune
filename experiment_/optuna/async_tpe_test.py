# %%
import optuna
import torch
import numpy as np
torch.manual_seed(42)
import random
import asyncio
from functools import partial

# 以下の二行があると、Jupyterでasyncio.run()が動くようになる。 
import nest_asyncio
nest_asyncio.apply()

# %%
optuna.create_study(study_name = "my_study", storage = "sqlite:///cuda_test.db", load_if_exists = True)

def objective(trial, device):
    x = trial.suggest_float('x', -10, 10)
    test_arr = torch.randn(10).to(device)
    return sum(test_arr * x)    

async def study_optimize(i):
    
    loop = asyncio.get_event_loop()
    
    device = 'cuda'
    sampler = optuna.samplers.TPESampler(seed = 42 + i)
    study = optuna.load_study(sampler = sampler, study_name = "my_study", storage = "sqlite:///cuda_test.db")
    func = partial(study.optimize, lambda trial: objective(trial, device), n_trials = 5)
    loop.run_in_executor(None, func)


def optimize():
    n_jobs = 8
    tasks = []
    for i in range(n_jobs):
        task = asyncio.create_task(study_optimize(i))
        tasks.append(task)
    
    loop = asyncio.get_event_loop()
    gather = asyncio.gather(*tasks)
    loop.run_until_complete(gather)

    # print('finish')
    

    
if __name__ == '__main__':
    # study = await optimize()
    # asyncio.run(optimize()) 
    optimize()
    
    # study = optuna.load_study(study_name = "my_study", storage = "sqlite:///cuda_test.db")
    # print(f'Best score: {study.best_value:.3f}')
    # print(f'Best parameters: {study.best_params}')
    # print(study.trials_dataframe())
# %%
