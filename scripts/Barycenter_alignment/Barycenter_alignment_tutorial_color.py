# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import colorsys
import random
from scipy.spatial import distance
import pickle as pkl

import ot
from sklearn.manifold import MDS
import sys

sys.path.append("../")
from src.Barycenter_alignment import Barycenter_alignment
from src.GW_alignment import GW_alignment
#%%
# load data
# similarity matrix
n_colors = 93
n_sub = 5
data_path = f"../data/num_groups_5_seed_0_fill_val_3.5.pickle"
with open(data_path, "rb") as f:
    # Write the dictionary to the file
    data = pkl.load(f)
sim_mat_list = data["group_ave_mat"]

# color label
file_path = "../data/color_dict.csv"
data_color = pd.read_csv(file_path)
color_label = data_color.columns.values

#%%
# Show simmilarity matrices
fig = plt.figure(figsize=(25, 5))
for i in range(n_sub):
    ax = fig.add_subplot(1, 5, i + 1)
    sns.heatmap(sim_mat_list[i], square=True, ax = ax)
    plt.title("Subject {}".format(i + 1))
plt.show()
#%%
# MDS
seed = 0 # random seed
embedding_list = []
MDS_embedding = MDS(n_components=3, dissimilarity='precomputed', random_state=seed)
for sim_mat in sim_mat_list:
    X = MDS_embedding.fit_transform(sim_mat) # X: N x 3
    embedding_list.append(X)
#%%
# visualization of embeddigs for each subject
fig = plt.figure(figsize=(25, 5))
for i in range(n_sub):
    ax = fig.add_subplot(1, 5, i + 1, projection='3d')
    coords_i = embedding_list[i]
    title = "Subject {}".format(i+1)

    ax.scatter(xs=coords_i[:, 0], ys=coords_i[:, 1], zs=coords_i[:, 2],
           marker="o", color=color_label, s=10)
    plt.title(title, fontsize=14)
plt.show()
#%%
# Set parameters
n_init_plan = 5 #  the number of initial plans
n_epsilons = 10 # number of epsilons
epsilons = np.linspace(0.02, 0.14, n_epsilons)
epsilon_range = [0.02, 0.14] # search range of epsilon
n_jobs = 10 # the number of cores
init_diag = True 

DATABASE_URL = 'mysql+pymysql://root@localhost/oizumi'
name_list = [f"Group {i + 1}" for i in range(n_sub)]

#%%
# Set the instance
barycenter = Barycenter_alignment(n_sub, embedding_list, pivot = 0, DATABASE_URL = DATABASE_URL, name_list = name_list)

# GW alignment to the pivot
# With optuna
Pi_list_before = barycenter.gw_alignment_to_pivot(optuna = True, 
                                                  n_init_plan = n_init_plan, 
                                                  epsilon_range = epsilon_range, 
                                                  n_trials = n_epsilons, 
                                                  n_jobs = n_jobs, 
                                                  init_diag = init_diag)

# Without using optuna
# Pi_list_before = barycenter.gw_alignment_to_pivot(optuna = False, 
#                                                   n_init_plan = n_init_plan, 
#                                                   epsilons = epsilons, 
#                                                   n_trials = n_epsilons, 
#                                                   n_jobs = n_jobs, 
#                                                   init_diag = init_diag)

#%%
# Barycenter alignment
Pi_list_after = barycenter.main_compute(Pi_list_before, max_iter = 30)
barycenter.calc_correct_rate_for_pairs(Pi_list_before, Pi_list_after)

#%%
# Plot
markers_list = ["o", "v", "1","2","3","4","<",">","^","8","s","p","P","h","H","+","x","X","D","d","|","_",0,1,2,3,4,5,6,7,8]
barycenter.plot_embeddings(markers_list, color_label)
# %%
