# %%
import os
import optuna
import multiprocessing as mp

import torch 


torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

test_arr = torch.randn(10).to('cuda')

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return sum(test_arr ** x)
 
study = optuna.create_study(study_name="my_study", storage = "sqlite:///cuda_test.db", load_if_exists = True)


def multi_run(dataset, seed):
    sampler = optuna.samplers.RandomSampler(seed = seed)
    loaded_study = optuna.load_study(sampler = sampler, study_name="my_study", storage="sqlite:///cuda_test.db")
    loaded_study.optimize(dataset, n_trials = 10, n_jobs = 1)

processes = []

n_jobs = 4
seed = 42
for i in range(n_jobs):
    p = mp.Process(target = multi_run, args=(objective, seed + i))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

# %%
"""
この計算結果になるはず。順番は入れ替わるものの、乱数自体は完全に固定されている。
[I 2023-03-30 16:00:25,763] A new study created in RDB with name: my_study
[I 2023-03-30 16:00:25,888] Trial 1 finished with value: 59.27319474294445 and parameters: {'x': -7.698908672204421}. Best is trial 0 with value: 6.296072711533571.
[I 2023-03-30 16:00:25,888] Trial 0 finished with value: 6.296072711533571 and parameters: {'x': -2.50919762305275}. Best is trial 0 with value: 6.296072711533571.
[I 2023-03-30 16:00:25,918] Trial 3 finished with value: 95.65290412467884 and parameters: {'x': 9.780230269512003}. Best is trial 0 with value: 6.296072711533571.
[I 2023-03-30 16:00:25,949] Trial 4 finished with value: 81.25735440102869 and parameters: {'x': 9.014286128198322}. Best is trial 0 with value: 6.296072711533571.
[I 2023-03-30 16:00:25,969] Trial 2 finished with value: 44.84770580921156 and parameters: {'x': 6.6968429733129895}. Best is trial 0 with value: 6.296072711533571.
[I 2023-03-30 16:00:25,973] Trial 5 finished with value: 4.758203996161064 and parameters: {'x': 2.181330785589628}. Best is trial 5 with value: 4.758203996161064.
[I 2023-03-30 16:00:25,993] Trial 6 finished with value: 0.9818719832473821 and parameters: {'x': 0.9908945368945083}. Best is trial 6 with value: 0.9818719832473821.
[I 2023-03-30 16:00:26,015] Trial 7 finished with value: 21.52847561487744 and parameters: {'x': 4.639878836228101}. Best is trial 6 with value: 0.9818719832473821.
[I 2023-03-30 16:00:26,076] Trial 9 finished with value: 3.8933986016227684 and parameters: {'x': 1.973169683940732}. Best is trial 6 with value: 0.9818719832473821.
[I 2023-03-30 16:00:26,087] Trial 8 finished with value: 19.106112729088593 and parameters: {'x': -4.3710539608987435}. Best is trial 6 with value: 0.9818719832473821.
[I 2023-03-30 16:00:26,111] Trial 10 finished with value: 62.47444764849235 and parameters: {'x': -7.904077912602605}. Best is trial 6 with value: 0.9818719832473821.
[I 2023-03-30 16:00:26,113] Trial 11 finished with value: 53.76087405619173 and parameters: {'x': -7.332180716280234}. Best is trial 6 with value: 0.9818719832473821.
[I 2023-03-30 16:00:26,115] Trial 12 finished with value: 47.329270289227914 and parameters: {'x': -6.87962719115127}. Best is trial 6 with value: 0.9818719832473821.
[I 2023-03-30 16:00:26,159] Trial 15 finished with value: 26.917498107888868 and parameters: {'x': -5.1882076006930244}. Best is trial 6 with value: 0.9818719832473821.
[I 2023-03-30 16:00:26,163] Trial 13 finished with value: 71.47364437821363 and parameters: {'x': -8.454208678416546}. Best is trial 6 with value: 0.9818719832473821.
[I 2023-03-30 16:00:26,186] Trial 16 finished with value: 11.9523624103457 and parameters: {'x': -3.4572188837772044}. Best is trial 6 with value: 0.9818719832473821.
[I 2023-03-30 16:00:26,226] Trial 14 finished with value: 23.9395861105546 and parameters: {'x': 4.892809633590357}. Best is trial 6 with value: 0.9818719832473821.
[I 2023-03-30 16:00:26,226] Trial 17 finished with value: 1.233454751050239 and parameters: {'x': -1.110610080563939}. Best is trial 6 with value: 0.9818719832473821.
[I 2023-03-30 16:00:26,248] Trial 18 finished with value: 47.33590801548771 and parameters: {'x': -6.880109593275947}. Best is trial 6 with value: 0.9818719832473821.
[I 2023-03-30 16:00:26,286] Trial 19 finished with value: 51.59189496194162 and parameters: {'x': 7.182749818971953}. Best is trial 6 with value: 0.9818719832473821.
[I 2023-03-30 16:00:26,309] Trial 21 finished with value: 0.2957626035274406 and parameters: {'x': -0.5438406048902937}. Best is trial 21 with value: 0.2957626035274406.
[I 2023-03-30 16:00:26,309] Trial 20 finished with value: 78.11603753372253 and parameters: {'x': -8.83832775663601}. Best is trial 6 with value: 0.9818719832473821.
[I 2023-03-30 16:00:26,357] Trial 24 finished with value: 53.63398789383462 and parameters: {'x': 7.323522915498703}. Best is trial 21 with value: 0.2957626035274406.
[I 2023-03-30 16:00:26,362] Trial 23 finished with value: 81.53295346620772 and parameters: {'x': -9.029559981871083}. Best is trial 21 with value: 0.2957626035274406.
...
[I 2023-03-30 16:00:26,623] Trial 36 finished with value: 4.513113506805362 and parameters: {'x': -2.124408978234973}. Best is trial 21 with value: 0.2957626035274406.
[I 2023-03-30 16:00:26,648] Trial 37 finished with value: 3.3071161110033356 and parameters: {'x': -1.8185478027820263}. Best is trial 21 with value: 0.2957626035274406.
[I 2023-03-30 16:00:26,672] Trial 38 finished with value: 0.03922308627213185 and parameters: {'x': 0.19804819179212885}. Best is trial 38 with value: 0.03922308627213185.
[I 2023-03-30 16:00:26,697] Trial 39 finished with value: 17.66487163975829 and parameters: {'x': 4.20295986654147}. Best is trial 38 with value: 0.03922308627213185.
"""