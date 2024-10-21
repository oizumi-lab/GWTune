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
    num_trial = 4,
    sinkhorn_method='sinkhorn', 
    to_types = 'torch',
    device = 'cpu', 
    data_type = 'double', 
    n_jobs = 3,
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
rsa_dict = ar.RSA_get_corr()

# %%
ar.set_pairs_computed(["Group1"])
ar.set_pairs_computed(["Group2"])
ar.set_pairs_computed(["Group3"])
ar.set_pairs_computed(["Group4"])
# %%
ar.set_pairs_computed(["Group1_vs_Group3"])

# %%
ar.set_specific_eps_list({"Group1": [1, 2]}, specific_only=True)
ar.set_specific_eps_list({"Group2": [1, 2]}, specific_only=False)
ar.set_specific_eps_list({"Group3": [1, 2]}, specific_only=True)
ar.set_specific_eps_list({"Group4": [1, 2]}, specific_only=False)

#%%
vis = VisualizationConfig(
    figsize = (8, 6),
    title_size = 30,
    xlabel = None,
    ylabel = None,
    xlabel_size = 20,
    ylabel_size = 20,
    xticks_rotation = 90,
    yticks_rotation = 0,
    cbar_ticks_size = 20,
    xticks_size = 5,
    yticks_size = 5,
    cbar_format = None,
    cbar_label = None,
    cbar_label_size = 20,
    cmap = "cividis",
    draw_category_line = True,
    category_line_alpha = 0.2,
    category_line_style = "dashed",
    category_line_color = "C2",
    dpi = 100,
    font = "Arial",
    fig_ext = "png",
    show_figure = True,
)

vis_hist = copy.deepcopy(vis)
vis_hist.set_params(title_size = 15, bins=10)

#%%
ar.show_sim_mat(
    # sim_mat_format="default",
    sim_mat_format="sorted",
    visualization_config=vis,
    visualization_config_hist=vis_hist,
    fig_dir=None,
    show_distribution=True,
)

# %%
ar2 = AlignRepresentations(
    config=config,
    representations_list=None,
    source_list=[representations[0], representations[2]],
    target_list=[representations[1], representations[3]],
    pairwise_method="permutation",
    metric="cosine", 
    main_results_dir = main_results_dir,
    data_name = data_select,
    pairs_computed=None,
    specific_eps_list=None,
)

# %%
ar2.set_pairs_computed(["Group1"])
ar2.set_pairs_computed(["Group2"])
ar2.set_pairs_computed(["Group3"])
ar2.set_pairs_computed(["Group4"])
# %%
# ar2.set_pairs_computed(["Group1_vs_Group3"])
ar2.set_pairs_computed(["Group1_vs_Group4"])

# %%
ar2.set_specific_eps_list({"Group1": [1, 2]}, specific_only=True)
ar2.set_specific_eps_list({"Group2": [1, 2]}, specific_only=False)
ar2.set_specific_eps_list({"Group3": [1, 2]}, specific_only=True)
ar2.set_specific_eps_list({"Group4": [1, 2]}, specific_only=False)

# %%
ar2.show_sim_mat(
    # sim_mat_format="default",
    sim_mat_format="sorted",
    visualization_config=vis,
    visualization_config_hist=vis_hist,
    fig_dir=None,
    show_distribution=True,
)

# %%
