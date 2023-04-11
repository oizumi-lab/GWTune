# %%
import optuna
import torch
import torch.multiprocessing as mp
from joblib import parallel_backend
torch.manual_seed(42)


print('---------')

def objective(trial, test_arr):
    x = trial.suggest_float("x", -10, 10)
    return sum(test_arr * x)
 
study = optuna.create_study(study_name="my_study", storage = "sqlite:///cuda_test.db", load_if_exists = True)


def multi_run(seed, i):
    # device = 'cuda:0'#+str(i)
    test_arr = torch.randn(10).to('cuda')
    sampler = optuna.samplers.RandomSampler(seed = seed + i)
    loaded_study = optuna.load_study(sampler = sampler, study_name="my_study", storage="sqlite:///cuda_test.db")
    
    loaded_study.optimize(lambda trial: objective(trial, test_arr), n_trials = 10, n_jobs = 1)

processes = []
n_jobs = 4
n_gpu = 4
seed = 42


# プロセスを生成する
processes = []
for i in range(n_jobs):
    p = mp.Process(target=multi_run, args=(seed, i))
    processes.append(p)

# プロセスを実行する
for p in processes:
    p.start()

# # プロセスが終了するのを待つ
for p in processes:
    p.join()
# %%
