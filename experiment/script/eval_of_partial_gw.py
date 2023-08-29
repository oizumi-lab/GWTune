#%%
# partial GWがどれだけ偉いか調べる
# Standard Library
import os
import pickle as pkl
import sys
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), '../'))

# Third Party Library
import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
# import pymysql
import scipy.io
import seaborn as sns
import torch
from sqlalchemy import URL, create_engine
from sqlalchemy_utils import create_database, database_exists, drop_database
#%%
# First Party Library
from src.gw_alignment import GW_Alignment
from src.utils.gw_optimizer import load_optimizer
from src.utils.init_matrix import InitMatrix
# os.chdir(os.path.dirname(__file__))
current_dir = Path(__file__).parent

data_path = current_dir / '../../data/color/num_groups_5_seed_0_fill_val_3.5.pickle'
with open(data_path, "rb") as f:
    data = pkl.load(f)
sim_mat_list = data["group_ave_mat"]
C1 = sim_mat_list[1]
C2 = sim_mat_list[2]

# emb1 = np.load('../data/AllenBrain/pseudo_mouse_A_emb.npy')
# C1 = ot.dist(emb1, metric="cosine")
# emb2 = np.load('../data/AllenBrain/pseudo_mouse_B_emb.npy')
# C2 = ot.dist(emb2, metric="cosine")

# show dissimilarity matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im1 = axes[0].imshow(C1, cmap='viridis')
cbar1 = fig.colorbar(im1, ax=axes[0])
im2 = axes[1].imshow(C2, cmap='viridis')
cbar2 = fig.colorbar(im2, ax=axes[1])

axes[0].set_title('Dissimilarity matrix #1')
axes[1].set_title('Dissmimilarity matrix #2')
plt.show()


#%%
# set the device ('cuda' or 'cpu') and variable type ('torch' or 'numpy')
device = 'cpu'
to_types = 'numpy'

# Set the range of epsilon
eps_list = [1e-5, 1e-1]
# eps_list = [1e-2, 1e-1, 1e-3]

eps_log = True # use log scale if True

# Set the params for the trial of optimize and max iteration for gw alignment computation
# set the number of trials, i.e., the number of epsilon values tested in optimization: default : 20
num_trial = 100

# the maximum number of iteration for GW optimization: default: 1000
max_iter = 500

# choose sampler
sampler_name = 'tpe'

# choose pruner
pruner_name = 'hyperband'
pruner_params = {'n_startup_trials': 1, 'n_warmup_steps': 2, 'min_resource': 2, 'reduction_factor' : 3}

# initialization of transportation plan
# 'uniform': uniform matrix, 'diag': diagonal matrix', random': random matrix, 'permutation': permutation matrix
init_mat_plan = 'uniform'

# the number of random initial matrices for 'random' or 'permutation' options：default: 100
n_iter = 1

## Set the parameters for GW alignment computation
# please choose the method of sinkhorn implemented by POT (URL : https://pythonot.github.io/gen_modules/ot.bregman.html#id87). For using GPU, "sinkhorn_log" is recommended.
sinkhorn_method='sinkhorn'

# user can define the dtypes both for numpy and torch, "float(=float32)" or "double(=float64)". For using GPU with "sinkhorn", double is storongly recommended.
data_type = "double"


for filename, problem_type in zip(["entropic_gw", "partial_gw"], ["entropic_gromov_wasserstein", "entropic_partial_gromov_wasserstein"]):
    if filename == "entropic_gw":
        continue

    save_path = current_dir / f'../../results/eval_of_partial_gw/{filename}'

    # Specify the RDB to use for distributed calculations
    db_params={"drivername": "sqlite"} # SQLite
    storage = "sqlite:///" + str(save_path) +  '/' + filename + '.db'

    # generate instance solves gw_alignment　
    test_gw = GW_Alignment(
        C1,
        C2,
        str(save_path),
        problem_type=problem_type,
        max_iter = max_iter,
        n_iter = n_iter,
        to_types = to_types,
        data_type = data_type,
        sinkhorn_method = sinkhorn_method,
        m=0.8
    )

        # generate instance optimize gw_alignment　
    opt = load_optimizer(
        save_path=str(save_path),
        filename=filename,
        storage=storage,
        init_mat_plan=init_mat_plan,
        n_iter = n_iter,
        num_trial = num_trial,
        n_jobs = 1,
        method = 'optuna',
        sampler_name = sampler_name,
        pruner_name = pruner_name,
        pruner_params = pruner_params,
    )

    ### Running the Optimization using `opt.run_study`
    # 2. run optimzation
    study = opt.run_study(
        test_gw,
        device,
        seed=42,
        init_mat_plan=init_mat_plan,
        eps_list=eps_list,
        eps_log=eps_log,
        search_space=None,
    )

    ### Extracting the Best Trial from the Study
    df_trial = study.trials_dataframe()
    best_trial = study.best_trial
    print(best_trial)

    # extracting optimized epsilon, GWD from best_trial
    eps_opt = best_trial.params['eps']
    GWD_opt = best_trial.values[0]

    # load the opitimized transportation plan from the saved file
    if to_types == 'numpy':
        OT = np.load(str(save_path)+f'/gw_{best_trial.number}.npy')
    elif to_types == 'torch':
        OT = torch.load(str(save_path)+f'/gw_{best_trial.number}.pt')
        OT = OT.to('cpu').numpy()

    # plot the optimal transportation plan
    plt.figure()
    plt.imshow(OT, aspect="equal")
    plt.title(f'OT eps:{eps_opt:.3f} GWD:{GWD_opt:.3f}')
    plt.savefig(str(save_path) + '/best_gw.png')
    plt.close()

    # evaluate accuracy of unsupervised alignment
    max_indices = np.argmax(OT, axis=1)
    accuracy = np.mean(max_indices == np.arange(OT.shape[0])) * 100
    print(f'accuracy={accuracy}%')

    #　figure plotting GWD as x-axis and accuracy as y-axis
    plt.figure()
    plt.scatter(df_trial['user_attrs_best_acc'] * 100, df_trial['value'], s = 50, c= df_trial['params_eps'])
    plt.xlabel('accuracy (%)')
    plt.ylabel('GWD')
    plt.colorbar(label='epsilon')
    plt.savefig(str(save_path) + '/acc_GW.png')
    plt.close()
