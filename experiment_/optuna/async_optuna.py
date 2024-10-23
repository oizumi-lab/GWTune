# %%
import optuna
import torch
import numpy as np
torch.manual_seed(42)

import asyncio
from functools import partial

# 以下の二行があると、Jupyterでasyncio.run()が動くようになる。 
import nest_asyncio
nest_asyncio.apply()

def objective(trial, device):
    x = trial.suggest_float('x', -10, 10)
    test_arr = torch.randn(10).to(device)
    return sum(test_arr * x)    

async def study_report(study, device):
    trial = study.ask()
    score = objective(trial, device)
    study.tell(trial, score)

def optimize():
    sampler = optuna.samplers.TPESampler(seed = 42)

    study = optuna.create_study(sampler = sampler, study_name = "my_study", storage = "sqlite:///cuda_test.db")

    # define the number of trials
    n_trials = 80

    # define the trials
    tasks = []
    for i in range(n_trials):
        device = 'cuda:' + str(i%4)
        task = asyncio.create_task(study_report(study, device))
        tasks.append(task)

    loop = asyncio.get_event_loop()
    gather = asyncio.gather(*tasks)
    loop.run_until_complete(gather)


    print(f'Best score: {study.best_value:.3f}')
    print(f'Best parameters: {study.best_params}')
    print(study.trials_dataframe())
    
    return study
    
if __name__ == '__main__':
    # study = await optimize()
    optimize()

# %%
