#%%
import os, sys, gc
# ここがないと、srcのimportが通らなかったです。(2023.3.28 佐々木)
# ただ、jupyterで動かすのか、debuggerで動かすのかで、必要かどうかは変わる。
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

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
# warnings.simplefilter("ignore")
import seaborn as sns
import matplotlib.style as mplstyle
from tqdm.auto import tqdm
import functools

from src.gw_alignment import GW_Alignment
from src.utils.gw_optimizer import load_optimizer

import scipy.io
# %load_ext autoreload

#%%
data_path = '../data/faces_GROUP_interp.mat'
mat_dic = scipy.io.loadmat(data_path)

C1 = mat_dic["group_mean_ATTENDED"]
C2 = mat_dic["group_mean_UNATTENDED"]

im1 = plt.imshow(C1)
plt.colorbar(im1)
plt.title('ATTENDED')
plt.show()

im2 = plt.imshow(C2)
plt.colorbar(im2)
plt.title('UNATTENDED')
plt.show()

#%%
# C1 = torch.from_numpy(C1.astype(np.float32)).clone()
# C2 = torch.from_numpy(C2.astype(np.float32)).clone()

p = ot.unif(len(C1))
q = ot.unif(len(C2))
#%%
# filename(study_name)を書く
# 保存結果としては
# ├── results
#     ├── gw_alignment
#         ├── {filename}(これがstudy_nameにもなる)
#             ├── {filename}.db # この.dbはここではなくて、各条件ごとのフォルダーの想定でした (2023.3.28 佐々木)
#             ├── diag
#             ├── uniform
#             └── random
#                 ├── gw_{trial_number}.pt
#                 ├── init_mat_{trial_number}.pt
#                 └── {filename}.db # sqliteでの挙動なら、ここがわかりやすいかと。MySQLなら阿部さんの方がいいのだろうか・・・ (2023.3.28 佐々木)


filename = 'elise_data'
save_path = '../results/gw_alignment/' + filename

# 使用する型とマシンを決定
device = 'cuda'
to_types = 'torch'

# device = 'cpu'
# to_types = 'numpy'

# 分散計算のために使うRDBを指定
sql_name = 'sqlite'
storage = "sqlite:///" + save_path +  '/' + filename + '.db'
# sql_name = 'mysql'
# storage = 'mysql+pymysql://root:olabGPU61@localhost/GW_MethodsTest'
# GW_MethodsTest
# チューニング対象のハイパーパラメーターの探索範囲を指定
init_plans_list = ['random']#, 'permutation']
eps_list = [1e-2, 1e-1]
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

# pruner_name = 'hyperband'
# pruner_name = 'nop'
pruner_name = 'median'
pruner_params = {'n_startup_trials': 1, 'n_warmup_steps': 2, 'min_resource': 2, 'reduction_factor' : 3}

delete_study = False
#%%
test_gw = GW_Alignment(C1, C2, p, q, save_path, max_iter = 1000, n_iter = n_iter, device = device, to_types = to_types, gpu_queue = None)

opt = load_optimizer(save_path,
                     n_jobs = 4,
                     num_trial = 100,
                     to_types = to_types,
                     method = 'optuna',
                     sampler_name = sampler_name,
                     pruner_name = pruner_name,
                     pruner_params = pruner_params,
                     n_iter = n_iter,
                     filename = filename,
                     sql_name = sql_name,
                     storage = storage,
                     delete_study = delete_study
)
#%%
### 最適化実行 
'''
2023.4.3 佐々木
実装済みの初期値条件の抽出をgw_optimizer.pyからinit_matrix.pyに移動しました。
また、run_studyに渡す関数は、alignmentとhistogramの両方ともを揃えるようにしました。
事前に、functools.partialで、必要なhyper parametersの条件を渡しておく。
''' 
# 1. 初期値の選択。実装済みの初期値条件の抽出をgw_optimizer.pyからinit_matrix.pyに移動しました。
init_plans = test_gw.main_compute.init_mat_builder.implemented_init_plans(init_plans_list)

# 2. 最適化関数の定義。事前に、functools.partialで、必要なhyper parametersの条件を渡しておく。
gw_objective = functools.partial(test_gw, init_plans_list = init_plans, eps_list = eps_list, eps_log = eps_log)

# 3. 最適化を実行。run_studyに渡す関数は、alignmentとhistogramの両方ともを揃えるようにしました。
study = opt.run_study(gw_objective)

#%%
# 最適化結果を確認
best_trial = study.best_trial
print(best_trial)

# epsilon + GWD
print(best_trial.number)
print(best_trial.params)

# %%
gw_opt = np.load(save_path+f'/random/gw_{best_trial.number}.npy')
plt.imshow(gw_opt)
plt.show()
# %%
