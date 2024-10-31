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
    n_jobs = 1,
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

#%%
vis = VisualizationConfig(
    figsize = (10, 10),
    title_size = 20,
    xlabel = None,
    ylabel = None,
    xlabel_size = 20,
    ylabel_size = 20,
    xticks_rotation = 90,
    yticks_rotation = 0,
    cbar_ticks_size = 20,
    xticks_size = 10,
    yticks_size = 10,
    cbar_format = None,
    cbar_label = None,
    cbar_label_size = 20,
    cmap="rocket_r",
    ticks="category",
    ot_category_tick=True,
    draw_category_line = True,
    category_line_alpha = 0.2,
    category_line_style = "dashed",
    category_line_color = "C2",
    cbar_range=[0, 2e-5],
    dpi = 100,
    font = "Arial",
    fig_ext = "png",
    show_figure = True,
)

# %%
ar.gw_alignment(
    compute_OT=True,
    delete_results=False,
    return_data=False,
    return_figure=False,
    # OT_format="default",
    OT_format="sorted",
    visualization_config=vis,
    # show_log=True,
    fig_dir=None,
    save_dataframe=True,
    change_sampler_seed=False,
    sampler_seed=42,
    fix_random_init_seed=True,
    first_random_init_seed=None,
    parallel_method="multiprocess",
)


# %%