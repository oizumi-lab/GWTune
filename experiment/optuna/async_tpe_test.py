# %%
import optuna
import torch
import numpy as np
torch.manual_seed(42)
import random
import asyncio

# 以下の二行があると、Jupyterでasyncio.run()が動くようになる。 
import nest_asyncio
nest_asyncio.apply()

# %%
optuna.create_study(study_name = "my_study", storage = "sqlite:///cuda_test.db", load_if_exists = True)

def objective(trial, device):
    x = trial.suggest_float('x', -10, 10)
    test_arr = torch.randn(10).to(device)
    return sum(test_arr * x)    

async def study_report(study, device):
    trial = study.ask()
    score = objective(trial, device)
    study.tell(trial, score)

async def optimize():
    sampler = optuna.samplers.TPESampler(seed = 42)

    study = optuna.load_study(sampler = sampler, study_name = "my_study", storage = "sqlite:///cuda_test.db")

    # define the number of trials
    n_trials = 80

    # define the trials
    tasks = []
    for i in range(n_trials):
        device = 'cuda:' + str(i%4)
        task = asyncio.create_task(study_report(study, device))
        tasks.append(task)

    for task in tasks:
        await task

    print(f'Best score: {study.best_value:.3f}')
    print(f'Best parameters: {study.best_params}')
    print(study.trials_dataframe())
    
    return study
    
if __name__ == '__main__':
    # study = await optimize()
    asyncio.run(optimize()) 
# %%
