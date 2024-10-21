# %%
import pytest
import os, sys
import numpy as np
import pickle as pkl
import pandas as pd
import copy 
#%%
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.getcwd(), '../../'))

from src.align_representations import AlignRepresentations, OptimizationConfig, VisualizationConfig
from debug_test.load_sample_data import sample_data

data_select = "THINGS"
representations, eps_list = sample_data(data_select)
main_results_dir = f"../../results/{data_select}"

# %%
config = OptimizationConfig(  
    gw_type = "entropic_gromov_wasserstein",
    eps_list = eps_list, 
    eps_log = True,
    num_trial = 2,
    sinkhorn_method='sinkhorn', 
    to_types = 'torch',
    device = 'cuda', 
    data_type = 'double', 
    n_jobs = 4,
    multi_gpu = True,
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

# %%
ar.gw_alignment(
    compute_OT=True,
    delete_results=False,
    save_dataframe=True,
    change_sampler_seed=False,
    sampler_seed=42,
    fix_random_init_seed=True,
    first_random_init_seed=None,
    parallel_method="multiprocess",
)


# %%
ar.drop_gw_alignment_files(drop_all=True, delete_database=True, delete_directory=True)



# %%