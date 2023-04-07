# %%
import optuna
import torch
import torch.multiprocessing as mp

torch.manual_seed(42)
test_arr = torch.randn(10).to('cuda')
print(test_arr)

print('---------')

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return sum(test_arr * x)
 
study = optuna.create_study(study_name="my_study", storage = "sqlite:///cuda_test.db", load_if_exists = True)


def multi_run(seed):
    sampler = optuna.samplers.TPESampler(seed = seed)
    loaded_study = optuna.load_study(sampler = sampler, study_name="my_study", storage="sqlite:///cuda_test.db")
    loaded_study.optimize(objective, n_trials = 10, n_jobs = 1)


processes = []

n_jobs = 4
seed = 42

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=n_jobs) as pool:
        pool.map(multi_run, range(n_jobs))


# %%
