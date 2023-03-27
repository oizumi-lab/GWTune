#%%
import os, sys, gc
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pymysql
import sqlalchemy
import torch
import ot
import time
import matplotlib.pyplot as plt
import optuna
from joblib import parallel_backend
import warnings
import copy
# warnings.simplefilter("ignore")
import os
import seaborn as sns
import matplotlib.style as mplstyle
from tqdm.auto import tqdm

from src.gw_alignment import GW_Alignment
from src.utils.gw_optimizer import load_optimizer, RunOptuna
%load_ext autoreload

#%%
# データダウンロード
path1 = '../data/model1.pt'
path2 = '../data/model2.pt'

model1 = torch.load(path1)
model2 = torch.load(path2)
p = ot.unif(len(model1))
q = ot.unif(len(model2))
#%%
# filename(study_name)を書く
# 保存結果としては
# ├── results
#     ├── gw_alignment
#         ├── {filename}(これがstudy_nameにもなる)
#             ├── {filename}.db
#             ├── diag
#             ├── uniform
#             └── random
#                 ├── gw_{trial_number}.pt
#                 ├── init_mat_{trial_number}.pt

filename = 'test'
save_path = '../results/gw_alignment/' + filename

# 使用する型とマシンを決定
device = 'cuda'
to_types = 'torch'

# 分散計算のために使うRDBを指定
sql_name = 'sqlite'
storage = "sqlite:///" + save_path +  '/' + filename + '.db'
# sql_name = 'mysql'
# storage = 'mysql+pymysql://root:olabGPU61@localhost/GW_MethodsTest'
# GW_MethodsTest
# チューニング対象のハイパーパラメーターの探索範囲を指定
init_plans_list = ['uniform', 'random', 'permutation']
eps_list = [1e-4, 1e-2]
eps_log = True

# init_mat_plan in ['random', 'permutation']のときに生成する初期値の数を指定
n_iter = 10

# 使用するsamplerを指定
# 現状使えるのは['random', 'grid', 'tpe']
sampler_name = 'tpe'

# 使用するprunerを指定
# 現状使えるのは['median', 'hyperband', 'nop']
# median pruner (ある時点でスコアが過去の中央値を下回っていた場合にpruning)
#     n_startup_trials : この数字の数だけtrialが終わるまではprunerを作動させない
#     n_warmup_steps   : 各trialについてこのステップ以下ではprunerを作動させない
# hyperband pruner (pruning判定の期間がだんだん伸びていく代わりに基準がだんだん厳しくなっていく)
#     min_resource     : 各trialについてこのステップ以下ではprunerを作動させない
#     reduction_factor : どれくらいの間隔でpruningをcheckするか。値が小さいほど細かい間隔でpruning checkが入る。2~6程度.

pruner_name = 'hyperband'
pruner_params = {'n_startup_trials': 1, 'n_warmup_steps': 2, 'min_resource': 2, 'reduction_factor' : 3}

delete_study = True
#%%
test_gw = GW_Alignment(model1, model2, p, q, save_path, max_iter = 100, n_iter = n_iter, device = device, to_types = to_types, gpu_queue = None)

opt = load_optimizer(test_gw.save_path,
                    n_jobs = 4,
                    num_trial = 100,
                    to_types = to_types,
                    method = 'optuna',
                    init_plans_list = init_plans_list,
                    eps_list = eps_list,
                    eps_log = eps_log,
                    sampler_name = sampler_name,
                    pruner_name = pruner_name,
                    pruner_params = pruner_params,
                    filename = filename,
                    sql_name = sql_name,
                    storage = storage,
                    delete_study = delete_study
)
#%%
# 最適化実行
start = time.time()
study = opt.run_study(test_gw)
processed_time = time.time() - start

#%%
# 最適化結果を確認
best_trial = study.best_trial
print(best_trial)

# %%
