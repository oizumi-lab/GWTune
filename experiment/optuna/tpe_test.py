# %%
import optuna
import torch
import torch.multiprocessing as mp
import numpy as np
torch.manual_seed(42)
from concurrent.futures import ThreadPoolExecutor
import asyncio

# def batched_objective(xs: np.ndarray, ys: np.ndarray):
#     return xs**2 + ys

# batch_size = 10
# study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())

# for _ in range(3):

#     # create batch
#     trial_numbers = []
#     x_batch = []
#     y_batch = []
#     for _ in range(batch_size):
#         trial = study.ask()
#         trial_numbers.append(trial.number)
#         x_batch.append(trial.suggest_float("x", -10, 10))
#         y_batch.append(trial.suggest_float("y", -10, 10))

#     # evaluate batched objective
#     x_batch = np.array(x_batch)
#     y_batch = np.array(y_batch)
#     objectives = batched_objective(x_batch, y_batch)

#     # finish all trials in the batch
#     for trial_number, objective in zip(trial_numbers, objectives):
#         study.tell(trial_number, objective)

# # %%
# optuna.create_study(study_name = "my_study", storage = "sqlite:///cuda_test.db", load_if_exists = True)

# async def objective_func(x, test_arr):
#     return sum(test_arr * x)
 
# async def multi_run(seed, i):
#     loop = asyncio.get_event_loop()
    
#     device = 'cuda:0'#+str(i)
#     test_arr = torch.randn(10).to(device)
#     sampler = optuna.samplers.TPESampler(seed = seed + i)
    
#     study = optuna.load_study(sampler = sampler, study_name="my_study", storage="sqlite:///cuda_test.db")
    
#     trial = study.ask()
#     x = trial.suggest_float("x", -10, 10)
    
#     res = await loop.run_in_executor(None, objective_func, x, test_arr)
#     print(res)
    
#     study.tell(trial.number, res)
    
# # %%
# async def main():
#     processes = []
#     n_jobs = 2
#     n_trials = 4
#     seed = 42
#     array = range(n_jobs)

#     loop = asyncio.get_event_loop()
    
#     for i in array:
#         processes.append(multi_run(seed, i))

#     gather = asyncio.gather(*processes)
    
#     study = optuna.load_study(study_name="my_study", storage="sqlite:///cuda_test.db")
#     print(study.trials_dataframe())


# # %%
# asyncio.run(main())


# %%
async def async_func(input: int):
    return input * 2


async def await_demo():
    async_result = await async_func(2)
    async_result2 = await async_func(4)
    print(async_result, type(async_result))
    print(async_result2, type(async_result2))

# asyncio.run(await_demo())
await_demo()
# %%