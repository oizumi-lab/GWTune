# %%
import optuna
from concurrent.futures import ThreadPoolExecutor

# %%

def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_float("y", -100, 100)
    return x**2 + y


if __name__ == "__main__":
    study = optuna.create_study()
    with ThreadPoolExecutor(10) as pool:
        for i in range(10):
            pool.submit(study.optimize, objective, n_trials=10)

    print(f"Number of trials: {len(study.get_trials())}")
    print(f"Best params: {study.best_params}")
# %%
