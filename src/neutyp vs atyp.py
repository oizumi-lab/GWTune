#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import itertools

#from src.preprocess_utils import *
# from src.utils.utils import get_reorder_idxs
#from src.plot_utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split

import ot
from sklearn.metrics import pairwise_distances

# from src.embedding_model import EmbeddingModel, ModelTraining
from GW_methods.src.align_representations import Representation, Pairwise_Analysis, Visualization_Config, Align_Representations, Optimization_Config

#%%

data_list = ["neutyp", "atyp"]
N_groups_list = [14, 2]

list_of_representation_list = []
for data, N_groups in zip(data_list, N_groups_list): 
    rearranged_color_embeddings_list = np.load(f"../results/rearranged_embeddings_list_{data}_Ngroup={N_groups}.npy")
    
    representation_list = []
    for i in range(N_groups):
        representation = Representation(name=f"Group{i+1}_{data}", embedding=rearranged_color_embeddings_list[i], metric="euclidean")
        representation_list.append(representation)
        
    list_of_representation_list.append(representation_list)


opt_config = Optimization_Config(data_name=f"color_N-A_Ngroup={N_groups_list[0]}&{N_groups_list[1]}", 
                                    init_plans_list=["random"],
                                    num_trial=5,
                                    n_iter=5, 
                                    max_iter=200,
                                    sampler_name="tpe", 
                                    eps_list=[0.02, 0.2],
                                    eps_log=True,
                                    )

alignment = Align_Representations(config=opt_config, 
                                    representations_list=list_of_representation_list[0]+list_of_representation_list[1],
                                    metric="euclidean",
                                    pair_number_list=range(N_groups_list[0]*N_groups_list[1])
                                    )

### Set pairs
# make all the pairs N-A
pairs = []
for representation_neutyp in list_of_representation_list[0]:
    for representation_atyp in list_of_representation_list[1]:
        pair = Pairwise_Analysis(target = representation_atyp, source = representation_neutyp, config = opt_config)
        pairs.append(pair)
alignment.pairwise_list = pairs

vis_config = Visualization_Config()
alignment.gw_alignment(results_dir="../results/gw alignment/",
                        load_OT=True,
                        returned="figure",
                        visualization_config=vis_config,
                        show_log=False,
                        fig_dir="../results/figs/"
                        )

## Calculate the accuracy of the optimized OT matrix
alignment.calc_accuracy(top_k_list = [1, 3, 5], eval_type = "ot_plan")
alignment.plot_accuracy(eval_type = "ot_plan", scatter = True, fig_dir="../results/figs", fig_name = f"top_k_accuracy_N-A_Ngroup={N_groups_list[0]}&{N_groups_list[1]}")

## Calculate the accuracy based on k-nearest neighbors
alignment.calc_accuracy(top_k_list = [1, 3, 5], eval_type = "k_nearest")
alignment.plot_accuracy(eval_type = "k_nearest", scatter = True, fig_dir="../results/figs", fig_name = f"k_nearest_rate_N-A_Ngroup={N_groups_list[0]}&{N_groups_list[1]}")

alignment.top_k_accuracy.to_csv(f"../results/top_k_accuracy_N-A_Ngroup={N_groups_list[0]}&{N_groups_list[1]}.csv")
alignment.k_nearest_matching_rate.to_csv(f"../results/k_nearest_rate_N-A_Ngroup={N_groups_list[0]}&{N_groups_list[1]}.csv")
# %%
