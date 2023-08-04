# %% [markdown]
#  # Tutorial for Gromov-Wassserstein unsupervised alignment 

# %%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../'))

import numpy as np
import pandas as pd
import pickle as pkl

from src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig

# %%
# list of representations where the instances of "Representation" class are included
representations = list()

# select data
data_select = "THINGS"

# %%
# define the coarce category labels
category_mat = pd.read_csv("../data/category_mat_manual_preprocessed.csv", sep = ",", index_col = 0)

# calculate the parameters for the coarce category labels
# Please prepare equivalent parameters when using other datasets.
from src.utils.utils_functions import get_category_data, sort_matrix_with_categories # get_category_data and sort_matrix_with_categories are functions specialied for this dataset
object_labels, category_idx_list, num_category_list, category_name_list = get_category_data(category_mat)

n_representations = 3 # Set the number of the instanses of "Representation". This number must be equal to or less than the number of the groups. 4 is the maximum for this data.
metric = "euclidean" # Please set the metric that can be used in "scipy.spatical.distance.cdist()".

for i in range(n_representations):
    name = f"Group{i+1}"
    embedding = np.load(f"../data/THINGS_embedding_Group{i+1}.npy")[0]
    
    representation = Representation(
        name=name, 
        embedding=embedding, # the dissimilarity matrix will be computed with this embedding.
        metric=metric,
        get_embedding=False, # If there is the embeddings, plese set this variable "False".
        object_labels=object_labels,
        category_name_list=category_name_list,
        category_idx_list=category_idx_list,
        num_category_list=num_category_list,
        func_for_sort_sim_mat=sort_matrix_with_categories
    )
    
    representations.append(representation)

# %%
# whether epsilon is sampled at log scale or not
eps_list_tutorial = [1, 10]
eps_log = True
num_trial = 4
init_mat_plan = "random"

# %%
config = OptimizationConfig(
    eps_list = eps_list_tutorial,
    eps_log = eps_log, 
    num_trial = num_trial, 
    sinkhorn_method='sinkhorn',
    to_types = 'torch', 
    device = 'cuda', 
    data_type = "double",
    n_jobs = 1, 
    multi_gpu = True,
    db_params={"drivername": "sqlite"},
    # db_params={"drivername": "mysql+pymysql", "username": "root", "password": "", "host": "localhost"},
    
    init_mat_plan = init_mat_plan,
    n_iter = 1,
    max_iter = 200,
)


# %%
sim_mat_format = "sorted"
visualize_config = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 15, 
    cbar_ticks_size=20,
    
    cmap = 'cividis',
    draw_category_line=True,
    category_line_color='C2',
    category_line_alpha=0.2,
    category_line_style='dashed',
    show_figure = True,
)



# %%
align_representation = AlignRepresentations(
    representations_list=representations,
    histogram_matching=False,
    config=config,
    metric="cosine",
    main_results_dir="../results",
    data_name = data_select,
)

# %%
compute_OT = False
delete_results = False

# %%
# ot_list = align_representation.gw_alignment(
#     compute_OT = compute_OT,
#     delete_results = delete_results,
    
#     return_data = False,
#     return_figure = True,
    
#     OT_format = sim_mat_format,
#     visualization_config = visualize_config,
#     show_log=False, 
#     fig_dir=None,
#     ticks='category', 
#     save_dataframe=False,
#     change_sampler_seed=True, 
#     fix_sampler_seed = 42, 
#     parallel_method="multithread",
# )


# %%
# GWOT without entropy
# By using optimal transportation plan obtained with entropic GW as an initial transportation matrix, we run the optimization of GWOT without entropy.  
# This procedure further minimizes GWD and enables us to fairly compare GWD values obtained with different entropy regularization values.  

align_representation.gwot_after_entropic(
    top_k=None,
    parallel_method = "multithread",
    category_mat=category_mat, 
    visualization_config = visualize_config,
)

# %%

# import glob
# import torch
# import matplotlib.pyplot as plt
# import ot
# import copy

# def get_top_k_trials(study, k):
#     trials = study.trials_dataframe()
#     sorted_trials = trials.sort_values(by="value", ascending=True)
#     top_k_trials = sorted_trials.head(k)
#     top_k_trials = top_k_trials[['number', 'value', 'params_eps']]
#     return top_k_trials

# def calculate_GWD(pairwise, top_k_trials):
#     GWD0_list = list()
#     OT0_list = list()
#     for i in top_k_trials['number']:
#         ot_path = glob.glob(pairwise.data_path + f"/gw_{i}.*")[0]
#         if '.npy' in ot_path:
#             OT = np.load(ot_path)
#         elif '.pt' in ot_path:
#             OT = torch.load(ot_path).to("cpu").numpy()

#         C1 = pairwise.source.sim_mat
#         C2 = pairwise.target.sim_mat
#         p = OT.sum(axis=1)
#         q = OT.sum(axis=0)
#         OT0, log0 = ot.gromov.gromov_wasserstein(C1, C2, p, q, log=True, verbose=False, G0=OT)
#         GWD0 = log0['gw_dist']
#         GWD0_list.append(GWD0)
#         OT0_list.append(OT0)
    
#     return GWD0_list, OT0_list

# def plot_OT(pairwise, pairwise_after):
#     pairwise.show_OT(title=f"before {pairwise.pair_name}")    
#     pairwise_after.show_OT(title=f"after {pairwise.pair_name}")

# def evaluate_accuracy_and_plot(pairwise, pairwise_after, eval_type):
#     df_before = pairwise.eval_accuracy(top_k_list = [1, 5, 10], eval_type=eval_type)
#     df_after = pairwise_after.eval_accuracy(top_k_list = [1, 5, 10], eval_type=eval_type)
#     width = 0.35  # Width of the bars
#     x = np.arange(len(df_before.index))
#     plt.bar(x - width/2, df_before[pairwise.pair_name], width, label='before')
#     plt.bar(x + width/2, df_after[pairwise.pair_name], width, label='after')
#     plt.title(f'{eval_type} accuracy')
#     plt.show()

# def plot_GWD_optimization(top_k_trials, GWD0_list, pair_name):
#     marker_size = 10
#     plt.figure(figsize=(8,6))
#     plt.scatter(top_k_trials["params_eps"], top_k_trials["value"], c = 'red', s=marker_size) # before
#     plt.scatter(top_k_trials["params_eps"], GWD0_list, c = 'blue', s=marker_size) # after
#     plt.xlabel("$\epsilon$")
#     plt.ylabel("GWD")
#     plt.xticks(rotation=30)
#     plt.title(f"$\epsilon$ - GWD ({pair_name})")
#     plt.grid(True)
#     plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
#     plt.tight_layout()
#     plt.show()
#     plt.clf()
#     plt.close()

# def run_optimization(pairwise_list, number_of_besttrials):
#     pairwise_after_list = list()
#     for pairwise in pairwise_list:
#         study = pairwise._run_optimization(compute_OT)
#         top_k_trials = get_top_k_trials(study, number_of_besttrials)
#         GWD0_list, OT0_list = calculate_GWD(pairwise, top_k_trials)
#         # create new instance for after optimization
#         pairwise_after = copy.deepcopy(pairwise)
#         pairwise_after.OT = OT0_list[np.argmin(GWD0_list)]
#         pairwise_after_list.append(pairwise_after)

#     return pairwise_after_list, top_k_trials, OT0_list, GWD0_list

# def plot_results(pairwise_list, pairwise_after_list, top_k_trials, OT0_list, GWD0_list):
#     # plot results
#     for pairwise, pairwise_after in zip(pairwise_list, pairwise_after_list):
#         plot_OT(pairwise, pairwise_after, OT0_list, GWD0_list)
#         evaluate_accuracy_and_plot(pairwise, pairwise_after, 'ot_plan')
#         evaluate_accuracy_and_plot(pairwise, pairwise_after, 'k_nearest')
#         plot_GWD_optimization(top_k_trials, GWD0_list, pairwise.pair_name)        

# #%%
# pairwise_list = align_representation.pairwise_list
# number_of_best_trials = 10 # This variable determines the number of best trials to select for further GWOT optimization without entropy.
# pairwise_after_list, top_k_trials, OT0_list, GWD0_list = run_optimization(pairwise_list, number_of_best_trials)
# plot_results(pairwise_list, pairwise_after_list, top_k_trials, OT0_list, GWD0_list)
# # %%



