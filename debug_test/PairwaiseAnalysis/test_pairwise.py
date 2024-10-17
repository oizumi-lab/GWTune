# %%
import os, sys
import numpy as np
import pickle as pkl
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.getcwd(), '../../'))
from src.align_representations import Representation, PairwiseAnalysis, OptimizationConfig, VisualizationConfig

representations = list()
data_select = "THINGS"

# %%
category_mat = pd.read_csv("../../data/THINGS/category_mat_manual_preprocessed.csv", sep = ",", index_col = 0)

from src.utils.utils_functions import get_category_data, sort_matrix_with_categories
object_labels, category_idx_list, num_category_list, category_name_list = get_category_data(category_mat)

n_representations = 2
metric = "euclidean" 

for i in range(n_representations):
    name = f"Group{i+1}" # the name of the representation
    embedding = np.load(f"../../data/THINGS/THINGS_embedding_Group{i+1}.npy")[0]
    
    representation = Representation(
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
    
    representations.append(representation)


# %%
config = OptimizationConfig(  
    gw_type = "entropic_gromov_wasserstein",
    eps_list = [1, 10], 
    eps_log = True,
    num_trial = 4,
    sinkhorn_method='sinkhorn', 
    to_types = 'numpy',
    device = 'cpu', 
    data_type = 'double', 
    n_jobs = 1,
    multi_gpu = False,
    storage = None,
    db_params = {"drivername": "sqlite"},
    init_mat_plan = "random",
    n_iter = 1,
    max_iter = 200,
)

# %%
source, target = representations[0], representations[1]

pair_name = f"{source.name}_vs_{target.name}"
pairwise_name = data_select + "_" + pair_name
pair_results_dir = f"../../results/{data_select}/{pairwise_name}"

pair = PairwiseAnalysis(
    data_name=data_select,
    results_dir=pair_results_dir,
    config=config,
    source=source,
    target=target,
    pair_name=pair_name,
    instance_name=pairwise_name,
)

# %%
pair.pair_name

# %%
# RSA
pair.rsa("pearson")

#%%
pair.rsa("spearman")

#%%
pair.rsa_for_each_category("pearson")

# %%
# histogram matching test
pair.match_sim_mat_distribution(return_data=False, method="target", plot=True)


# %%
ot = pair.run_entropic_gwot(
    compute_OT=False, 
    save_dataframe=True, 
    fix_random_init_seed=False,
)

#%%
# pair.delete_previous_results(delete_database=True, delete_directory=True)

#%%
# df = pair.run_gwot_no_entropy(num_seed=10)

# %%
pair.OT

# %%
pair.sorted_OT

# %%
pair.calc_matching_rate(top_k_list=[1,5])
# %%
pair.plot_OT(return_sorted=False)
# %%
vis_ot = VisualizationConfig(
    show_figure = True,
    fig_ext="svg", # you can also use "png" or "pdf", and so on. Default is "png".
    figsize=(8, 6), 
    title_size = 15, 
    cmap = "rocket_r",
    cbar_ticks_size=20,
    
    ot_object_tick=False,
    ot_category_tick=True,
    
    xticks_size=10,
    yticks_size=10,
    
    # Note that please set ot_category_tick = True when drawing the category line.
    draw_category_line=True,
    category_line_color="black",
    category_line_alpha=0.5,
    category_line_style="dashed",
)

pair.plot_OT(return_sorted=True, visualization_config=vis_ot)

#%%
vis_log = VisualizationConfig(
    show_figure = True,
    fig_ext="svg", # you can also use "png" or "pdf", and so on. Default is "png".
    figsize=(8, 6), 
    title_size = 15, 
    cmap = "rocket_r",
    cbar_ticks_size=20,
    plot_eps_log=True,
)

pair.plot_optimization_log(visualization_config=vis_log)
# %%
