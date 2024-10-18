# %%
import os, sys
import numpy as np
import scipy as sp
import pickle as pkl
import pandas as pd
import torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.getcwd(), '../../'))
import pytest
from src.align_representations import Representation

representations = list()
data_select = "THINGS"

# %%
n_representations = 4 
metric = "euclidean"

# %%
category_mat = pd.read_csv("../../data/THINGS/category_mat_manual_preprocessed.csv", sep = ",", index_col = 0)

# %%
# sorted_category_mat = category_mat.sort_index(axis=1)

# %%
from src.utils.utils_functions import get_category_data, sort_matrix_with_categories
object_labels, category_idx_list, num_category_list, category_name_list = get_category_data(category_mat, sort_by="default")

i = 0 
metric = "euclidean" 
name = f"Group{i+1}" # the name of the representation
embedding = np.load(f"../../data/THINGS/THINGS_embedding_Group{i+1}.npy")[0]


#%%
rep = Representation(
    name=name,
    embedding=embedding,
    metric=metric,
    get_embedding=False, 
    object_labels=object_labels,
    category_name_list=category_name_list, 
    category_idx_list=category_idx_list, 
    num_category_list=num_category_list, 
    func_for_sort_sim_mat=sort_matrix_with_categories,
)

#%%
rep.plot_sim_mat(return_sorted=False)
# %%
rep.plot_sim_mat(return_sorted=True)

# %%
metric = "cosine"
rep = Representation(
    name=name,
    embedding=embedding,
    metric=metric,
    get_embedding=False, 
    object_labels=object_labels,
    category_name_list=category_name_list, 
    category_idx_list=category_idx_list, 
    num_category_list=num_category_list, 
    func_for_sort_sim_mat=sort_matrix_with_categories,
)

#%%
rep.plot_sim_mat(return_sorted=False)
# %%
rep.plot_sim_mat(return_sorted=True)

# %%
