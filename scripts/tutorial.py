#%%
import os, sys, gc
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
# %load_ext autoreload
import scipy.io
import pickle as pkl

os.chdir(os.path.dirname(__file__))

#%%
### load data
# you can choose the following data
# 'DNN': representations of 2000 imagenet images in AlexNet and VGG
# 'color': human similarity judgements of 93 colors for 5 paricipants groups
# 'face': human similarity judgements of 16 faces, attended vs unattended condition in the same participant
data_select = 'color'

if data_select == 'DNN':
    path1 = '../data/model1.pt'
    path2 = '../data/model2.pt'
    C1 = torch.load(path1)
    C2 = torch.load(path2)
elif data_select == 'color':
    data_path = '../data/num_groups_5_seed_0_fill_val_3.5.pickle'
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    sim_mat_list = data["group_ave_mat"]
    C1 = sim_mat_list[0]
    C2 = sim_mat_list[1]
elif data_select == 'face':
    data_path = '../data/faces_GROUP_interp.mat'
    mat_dic = scipy.io.loadmat(data_path)
    C1 = mat_dic["group_mean_ATTENDED"]
    C2 = mat_dic["group_mean_UNATTENDED"]

# may not be needed?
if type(C1) == np.ndarray:
    C1 = torch.from_numpy(C1.astype(np.float32)).clone()
    C2 = torch.from_numpy(C2.astype(np.float32)).clone()

im1 = plt.imshow(C1)
plt.colorbar(im1)
plt.title('Dissmilarity matrix #1')
plt.show()

im2 = plt.imshow(C2)
plt.colorbar(im2)
plt.title('Dissmilarity matrix #2')
plt.show()
#%%
# set the filename and foldername for saving optuna results
filename = 'test'
save_path = '../results/gw_alignment/' + filename

# Delete previous optimization results or not
delete_study = False

# set the device ('cuda' or 'cpu') and variable type ('torch' or 'numpy')
# device = 'cuda'
# to_types =    'torch'
device = 'cpu'
to_types = 'numpy'

# the number of jobs
n_jobs = 4

# 分散計算のために使うRDBを指定
sql_name = 'sqlite'
storage = "sqlite:///" + save_path +  '/' + filename + '.db'
# sql_name = 'mysql'
# storage = 'mysql+pymysql://root:olabGPU61@localhost/GW_MethodsTest'
# GW_MethodsTest

#%%
### Set the parameters for optimization
# initialization of transportation plan
# 'uniform': uniform matrix, 'random': random matrix, 'permutation': permutation matrix
init_plans_list = ['random']

# you can select multiple options
# init_plans_list = ['uniform', 'random']

# set the range of epsilon
# you should set the minimum value and maximum value
eps_list = [1e-2, 1e-1]
eps_log = False # use log scale if True

# set the number of trials, i.e., the number of epsilon values tested in optimization
num_trial = 4

# the number of random initial matrices for 'random' or 'permutation'] options
n_iter = 2

# the maximum number of iteration for GW optimization: default: 1000
max_iter = 100

# choose sampler 
# 'random': randomly select epsilon between the range of epsilon
# 'grid': grid search between the range of epsilon
# 'tpe': Bayesian sampling
sampler_name = 'tpe'

# choose pruner
# 'median': ある時点でスコアが過去の中央値を下回っていた場合にpruning
#     n_startup_trials : この数字の数だけtrialが終わるまではprunerを作動させない
#     n_warmup_steps   : 各trialについてこのステップ以下ではprunerを作動させない
# 'hyperband': pruning判定の期間がだんだん伸びていく代わりに基準がだんだん厳しくなっていく)
#     min_resource     : 各trialについてこのステップ以下ではprunerを作動させない
#     reduction_factor : どれくらいの間隔でpruningをcheckするか。値が小さいほど細かい間隔でpruning checkが入る。2~6程度.
# 'nop': no pruning
pruner_name = 'median'
pruner_params = {'n_startup_trials': 1, 'n_warmup_steps': 2, 'min_resource': 2, 'reduction_factor' : 3}

#%%
p = ot.unif(len(C1))
q = ot.unif(len(C2))
test_gw = GW_Alignment(C1, C2, p, q, save_path, max_iter = max_iter, n_iter = n_iter, device = device, to_types = to_types, gpu_queue = None)

opt = load_optimizer(save_path,
                     n_jobs = n_jobs,
                     num_trial = num_trial,
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

### optimization
# 1. 初期値の選択。実装済みの初期値条件の抽出をgw_optimizer.pyからinit_matrix.pyに移動しました。
init_plans = test_gw.main_compute.init_mat_builder.implemented_init_plans(init_plans_list)

# 2. 最適化関数の定義。事前に、functools.partialで、必要なhyper parametersの条件を渡しておく。
gw_objective = functools.partial(test_gw, init_plans_list = init_plans, eps_list = eps_list, eps_log = eps_log)

# run optimzation
study = opt.run_study(gw_objective, gpu_board = device)

#%%
print(study.trials_dataframe().sort_values('params_eps'))

#%%