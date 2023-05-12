#%%
# Standard Library
import functools
import gc
import os
import pickle as pkl
import sys
import time
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

# Third Party Library
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import ot
import pandas as pd
import pymysql
import scipy.io
from scipy.spatial import distance
from sklearn.manifold import MDS
import seaborn as sns
import torch

# First Party Library
from src.gw_alignment import GW_Alignment
from src.utils.gw_optimizer import load_optimizer
from src.utils.init_matrix import InitMatrix
from src.utils.evaluation import calc_correct_rate_ot_plan, pairwise_k_nearest_matching_rate
from src.utils.utils_functions import procrustes, get_category_idx
from src.utils.visualize_functions import Visualize_Embedding

os.chdir(os.path.dirname(__file__))

# %load_ext autoreload
# warnings.simplefilter("ignore")

#%%
### load data
# you can choose the following data
# 'DNN': representations of 2000 imagenet images in AlexNet and VGG
# 'color': human similarity judgements of 93 colors for 5 paricipants groups
# 'face': human similarity judgements of 16 faces, attended vs unattended condition in the same participant
# 'THINGS' : human similarity judgements of 1854 objects for 4 paricipants groups
data_select = "color"

if data_select == "DNN":
    path1 = "../data/model1.pt"
    path2 = "../data/model2.pt"
    C1 = torch.load(path1)
    C2 = torch.load(path2)
elif data_select == "color":
    data_path = "../data/num_groups_5_seed_0_fill_val_3.5.pickle"
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    sim_mat_list = data["group_ave_mat"]
    C1 = sim_mat_list[0]
    C2 = sim_mat_list[1]
    # color label
    file_path = "../data/color_dict.csv"
    data_color = pd.read_csv(file_path)
    color_labels = data_color.columns.values
elif data_select == "face":
    data_path = "../data/faces_GROUP_interp.mat"
    mat_dic = scipy.io.loadmat(data_path)
    C1 = mat_dic["group_mean_ATTENDED"]
    C2 = mat_dic["group_mean_UNATTENDED"]
elif data_select == "THINGS":
    path1 = "../data/THINGS_embedding_Group1.npy"
    path2 = "../data/THINGS_embedding_Group2.npy"
    embedding1 = np.load(path1)[0]
    embedding2 = np.load(path2)[0]
    C1 = distance.cdist(embedding1, embedding1, metric="euclidean")
    C2 = distance.cdist(embedding2, embedding2, metric="euclidean")
    # Category data
    category_name_list = ["bird", "insect", "plant", "clothing", "furniture", "fruit", "drink", "vehicle"]
    category_mat = pd.read_csv("../data/category_mat_manual_preprocessed.csv", sep=",", index_col=0)
    category_idx_list, category_num_list = get_category_idx(category_mat, category_name_list, show_numbers=True)

### Get the embeddings
# Set the embedding dimension
dim = 3

# Get embeddings with MDS
if data_select != "THINGS":  # THINGS-data already has embeddings
    MDS_embedding = MDS(n_components=dim, dissimilarity="precomputed", random_state=0)
    embedding1 = MDS_embedding.fit_transform(C1)
    embedding2 = MDS_embedding.fit_transform(C2)

# Show dissimilarity matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im1 = axes[0].imshow(C1, cmap="viridis")
cbar1 = fig.colorbar(im1, ax=axes[0])
im2 = axes[1].imshow(C2, cmap="viridis")
cbar2 = fig.colorbar(im2, ax=axes[1])

axes[0].set_title("Dissimilarity matrix #1")
axes[1].set_title("Dissmimilarity matrix #2")
plt.show()
#%%
# set the filename and foldername for saving optuna results
# filename is also treated as optuna study_name
filename = "test"
save_path = "../results/gw_alignment/" + filename

# Delete previous optimization results or not
# If the same filename has different search space, optuna may not work well.
delete_study = False

# set the device ('cuda' or 'cpu') and variable type ('torch' or 'numpy')
device = "cuda:3"
to_types = "torch"

# the number of jobs
n_jobs = 4

# Specify the RDB to use for distributed calculations
sql_name = "sqlite"
storage = "sqlite:///" + save_path + "/" + filename + ".db"
# sql_name = 'mysql'
# storage = 'mysql+pymysql://root:olabGPU61@localhost/GridTest'

#%%
### Set the parameters for optimization
# initialization of transportation plan
# 'uniform': uniform matrix, 'diag': diagonal matrix
# 'random': random matrix, 'permutation': permutation matrix
init_plans_list = ["random"]

# you can select multiple options
# init_plans_list = ['uniform', 'random']

# set the number of trials, i.e., the number of epsilon values tested in optimization: default : 20
num_trial = 10

# the number of random initial matrices for 'random' or 'permutation' options：default: 100
n_iter = 10

# the maximum number of iteration for GW optimization: default: 1000
max_iter = 500

# the maximum number of iteration for sinkhorn: default: 1000
numItermax = 1000

# choose sampler
# 'random': randomly select epsilon between the range of epsilon
# 'grid': grid search between the range of epsilon
# 'tpe': Bayesian sampling
sampler_name = "tpe"

# set the range of epsilon
# set only the minimum value and maximum value for 'tpe' sampler
# for 'grid' or 'random' sampler, you can also set the step size
# eps_list = [1, 10] # for THINGS
eps_list = [0.02, 0.2]  # for colors

# eps_list = [1e-2, 1e-1, 1e-2]

eps_log = True  # use log scale if True

# choose pruner
# 'median': Pruning if the score is below the past median at a certain point in time
#   n_startup_trials: Do not activate the pruner until this number of trials has finished
#   n_warmup_steps: Do not activate the pruner for each trial below this step
# 'hyperband': Use multiple SuccessiveHalvingPrunerd that gradually longer pruning decision periods and that gradually stricter criteria
#   min_resource: Do not activate the pruner for each trial below this step
#   reduction_factor: How often to check for pruning. Smaller values result in more frequent pruning checks. Between 2 to 6.
# 'nop': no pruning
pruner_name = "hyperband"
pruner_params = {"n_startup_trials": 1, "n_warmup_steps": 2, "min_resource": 2, "reduction_factor": 3}

#%%
# distribution in the source space, and target space
p = ot.unif(len(C1))
q = ot.unif(len(C2))

# generate instance solves gw_alignment
test_gw = GW_Alignment(C1, C2, p, q, save_path, max_iter=max_iter, n_iter=n_iter, to_types=to_types)

# generate instance optimize gw_alignment
opt = load_optimizer(
    save_path,
    n_jobs=n_jobs,
    num_trial=num_trial,
    to_types=to_types,
    method="optuna",
    sampler_name=sampler_name,
    pruner_name=pruner_name,
    pruner_params=pruner_params,
    n_iter=n_iter,
    filename=filename,
    sql_name=sql_name,
    storage=storage,
    delete_study=delete_study,
)

### optimization
# 1. 初期値の選択。実装済みの初期値条件の抽出をgw_optimizer.pyからinit_matrix.pyに移動しました。
init_plans = InitMatrix().implemented_init_plans(init_plans_list)

# used only in grid search sampler below the two lines
eps_space = opt.define_eps_space(eps_list, eps_log, num_trial)
search_space = {"eps": eps_space, "initialize": init_plans}

# 2. run optimzation
# parallel = 'thread' or 'multiprocessing', default is 'multiprocessing'
study = opt.run_study(
    test_gw,
    device,
    # parallel="multiprocessing",
    parallel="thread",
    init_plans_list=init_plans,
    eps_list=eps_list,
    eps_log=eps_log,
    search_space=search_space,
)

# parallelは無意味だということがわかった, default is None
study = opt.run_study(
    test_gw,
    device,
    parallel=None,
    init_plans_list=init_plans,
    eps_list=eps_list,
    eps_log=eps_log,
    search_space=search_space,
)

### View Results
print(
    study.trials_dataframe().sort_values("params_eps")
)  # jupyterだと、displayでもいいが、vscodeでは警告がでるので、printに変えます。(2023.4.10 佐々木)

#%% checkresults
df_trial = study.trials_dataframe()
best_trial = study.best_trial
print(best_trial)

# optimized epsilon, GWD, and transportation plan
eps_opt = best_trial.params["eps"]
GWD_opt = best_trial.values[0]

if to_types == "numpy":
    OT = np.load(save_path + f"/{init_plans_list[0]}/gw_{best_trial.number}.npy")
elif to_types == "torch":
    OT = torch.load(save_path + f"/{init_plans_list[0]}/gw_{best_trial.number}.pt")
    OT = OT.to("cpu").detach().numpy().copy()

plt.imshow(OT)
plt.title(f"OT eps:{eps_opt:.3f} GWD:{GWD_opt:.3f}")
plt.show()

df_trial = study.trials_dataframe()

## Evaluate the accuracy of OT plan
# Calculate the top k accuracy (Count when the diagonal component of each row is equal to or greater than the kth largest value in that row)
top_k_list = [1, 5, 10]
accuracy_list = []
for k in top_k_list:
    acc = calc_correct_rate_ot_plan(OT, top_n=k)
    accuracy_list.append(acc)
# Plot
plt.scatter(x=top_k_list, y=accuracy_list)
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.ylim(0, 100)
plt.grid()
plt.show()

# figure plotting epsilon as x-axis and GWD as y-axis
sns.scatterplot(data=df_trial, x="params_eps", y="value", s=50)
plt.xlabel("$\epsilon$")
plt.ylabel("GWD")
plt.show()

# 　figure plotting GWD as x-axis and accuracy as y-axis
sns.scatterplot(data=df_trial, x="value", y="user_attrs_best_acc", s=50)
plt.xlabel("GWD")
plt.ylabel("accuracy")
plt.show()
# %%
## Procrustes alignment
# Move embedding2 closer to embedding1 using OT plan
Q, new_embedding2 = procrustes(embedding1, embedding2, OT)

# Evaluate the accuracy in terms of the proximity of embeddings
top_k_list = [1, 5, 10]
matching_rate_list = []
for k in top_k_list:
    acc = pairwise_k_nearest_matching_rate(embedding1, new_embedding2, top_n=k, metric="euclidean")
    matching_rate_list.append(acc)
# Plot
plt.scatter(x=top_k_list, y=matching_rate_list)
plt.xlabel("k")
plt.ylabel("Matching rate")
plt.ylim(0, 100)
plt.grid()
plt.show()

## Visualize the aligned embeddings
embedding_list = [embedding1, new_embedding2]
name_list = ["Group1", "Group2"]

# Color data
if data_select != "color":
    color_labels = None
# Category data
if data_select != "THINGS":
    category_name_list, category_num_list, category_idx_list = None, None, None

# Visualize
visualize = Visualize_Embedding(
    embedding_list, name_list, color_labels, category_name_list, category_num_list, category_idx_list
)
visualize.plot_embedding(dim=3)

# %%
