# %%
import pytest
import os, sys
import numpy as np
import pickle as pkl
import pandas as pd

#%%
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.getcwd(), '../../'))

from src.align_representations import AlignRepresentations, OptimizationConfig, VisualizationConfig
from debug_test.load_sample_data import sample_data

data_select = "THINGS"
representations, eps_list = sample_data(data_select)
main_results_dir = f"../../results/{data_select}"


#%%
if data_select == "THINGS":
    device = 'cuda'
    to_types = 'torch'
    multi_gpu = True
    
    category_name_list = ["bird", "insect", "plant", "clothing",  "furniture", "fruit", "drink", "vehicle"] # please specify the categories that you would like to visualize.
    category_mat = pd.read_csv("../../data/THINGS/category_mat_manual_preprocessed.csv", sep = ",", index_col = 0)

    from src.utils.utils_functions import get_category_data
    object_labels, category_idx_list, num_category_list, category_name_list = get_category_data(category_mat, category_name_list, show_numbers = True)
    color_labels = None

if data_select == "color":
    object_labels = None
    category_idx_list = None
    num_category_list = None
    category_name_list = None
    
    device = 'cpu'
    to_types = 'numpy'
    multi_gpu = False
    
    file_path = "../../data/color/color_dict.csv"
    data_color = pd.read_csv(file_path)
    color_labels = data_color.columns.values # Set color labels if exist
    
# %%
config = OptimizationConfig(  
    gw_type = "entropic_gromov_wasserstein",
    eps_list = eps_list, 
    eps_log = True,
    num_trial = 4,
    sinkhorn_method='sinkhorn', 
    to_types = to_types,
    device = device, 
    data_type = 'double', 
    n_jobs = 3,
    multi_gpu = multi_gpu,
    storage = None,
    db_params = {"drivername": "sqlite"},
    init_mat_plan = "random",
    n_iter = 1,
    max_iter = 200,
)

# %%
ar = AlignRepresentations(
    config=config,
    pairwise_method="combination",
    representations_list=representations,
    main_results_dir=main_results_dir,
    data_name=data_select,
)

#%%
visualization_embedding = VisualizationConfig(
    fig_ext="svg",
    figsize=(8, 8), 
    xlabel="PC1",
    ylabel="PC2", 
    zlabel="PC3", 
    marker_size=6,
    legend_size=10,
)

#%%
# PCA
# The figures made from the following code will be saved in the directory "/main_results_dir/data_name/visualize_embedding".
ar.visualize_embedding(
    dim=3,
    method="PCA",
    pivot=0, 
    visualization_config=visualization_embedding,
    category_name_list=category_name_list, 
    category_idx_list=category_idx_list, 
    num_category_list=num_category_list,
)
    
#%%


