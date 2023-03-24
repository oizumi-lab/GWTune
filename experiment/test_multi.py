# %%
import os
import optuna
import multiprocessing as mp
import torch 
import ot

tensor_a = torch.randn(10).to('cuda')
tensor_b = torch.randn(10).to('cuda')

def objective(trial):
    x = trial.suggest_float("x", 1e-2, 1, log = True)
    h1_prob = tensor_a / tensor_a.sum()
    h2_prob = tensor_b / tensor_b.sum()
    
    # sinkhornを動かすコマンド。
    dist = ot.dist(h1_prob.unsqueeze(dim=1), h2_prob.unsqueeze(dim=1))
    res = ot.sinkhorn2(h1_prob, h2_prob, dist, reg = x)
    return res

unittest_save_path = '../results/experiment/'
sampler = optuna.samplers.TPESampler(seed = 42)

os.makedirs(unittest_save_path, exist_ok=True)

study = optuna.create_study(sampler=sampler,
                            storage = 'sqlite:///' + unittest_save_path + '/test_multi.db',
                            load_if_exists = True)
    
# %%
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
