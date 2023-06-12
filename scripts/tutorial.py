# %% [markdown]
# # Tutorial for using `align_represenatations.py` 

# %%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../'))

import numpy as np
import pandas as pd
import pickle as pkl

from src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig

# %% [markdown]
# ## Load data
# you can choose the following data
# 1. 'DNN': representations of 2000 imagenet images in AlexNet and VGG19
# 1. 'color': human similarity judgements of 93 colors for 5 paricipants groups
# 1. 'face': human similarity judgements of 16 faces, attended vs unattended condition in the same participant
# 1. 'THINGS' : human similarity judgements of 1854 objects for 4 paricipants groups
# 
# 
# "data_select" in next code block can define which dataset are going to be used.

# %%
data_select = "THINGS"

# %%
# Set Representations
#   - A Representation needs a name and either an embedding or a similarity matrix.

# representations list that will be used in Align_Representations
representations = list()

# %% [markdown]
# ### Dataset No1. `color`

# %%
# Load data and create representations instance
if data_select == 'color':
    n_representations = 5 # Set the number of representations. This number must be equal to or less than the number of groups.
    metric = "euclidean" # Please set metric that can be used in "scipy.distance.cdist()"
    
    category_mat = None
    data_path = '../data/num_groups_5_seed_0_fill_val_3.5.pickle'
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    sim_mat_list = data["group_ave_mat"]
    for i in range(n_representations):
        name = f"Group{i+1}"
        sim_mat = sim_mat_list[i]
        representation = Representation(name = name, sim_mat = sim_mat)
        representations.append(representation)


# %% [markdown]
# ### Dataset No.2 `THINGS`

# %%
if data_select == "THINGS":
    # define the label information of the dataset
    category_mat = pd.read_csv("../data/category_mat_manual_preprocessed.csv", sep = ",", index_col = 0)
    
    # define the parameters for label info. 
    # Users can define these by themselves if they use a different dataset and the format of parameters are the same.
    from src.utils.utils_functions import get_category_data, sort_matrix_with_categories
    object_labels, category_idx_list, num_category_list, category_name_list = get_category_data(category_mat)
    
    n_representations = 4 # Set the number of representations. This number must be equal to or less than the number of groups.
    metric = "euclidean" # Please set metric that can be used in "scipy.distance.cdist()"
    
    for i in range(n_representations):
        name = f"Group{i+1}"
        embedding = np.load(f"../data/THINGS_embedding_Group{i+1}.npy")[0]
        
        representation = Representation(
            name = name, 
            embedding = embedding, 
            metric = metric, 
            object_labels = object_labels,
            category_name_list = category_name_list,
            category_idx_list = category_idx_list,
            num_category_list = num_category_list,
            func_for_sort_sim_mat = sort_matrix_with_categories
        )
        
        representations.append(representation)

# %% [markdown]
# ## Set the parameters for the optimazation of GWOT, and the parameters for visualizing matrices

# %%
config = OptimizationConfig(
    eps_list = [1, 10], # [1, 10] for THINGS data, [0.02, 0.2] for colors data
    eps_log = True,
    num_trial = 4,
 
    device = 'cuda',
    to_types = 'torch',
    
    n_jobs = 1,
    parallel_method="multithread", # "multiprocess" or "multithread". Default is "multithread".
    multi_gpu=[1,2], # "True" : all the GPU installed in your environment, "list (e.g.[0,2,3])"" : cuda:0,2,3, and "False" : single gpu (or cpu for numpy) will use for parallel computation.
    
    init_plans_list = ['random'],
    n_iter = 1,

    max_iter = 200,
    data_name = data_select, 
    delete_study = False, 
    sampler_name = 'tpe',
    pruner_name = 'hyperband',
    pruner_params = {'n_startup_trials': 1, 
                     'n_warmup_steps': 2, 
                     'min_resource': 2, 
                     'reduction_factor' : 3
                    },
)


# %% [markdown]
# ## Unsupervised alignment between Representations
#     - The object has methods for RSA, GW-alignment, evaluation of the alignment and visalization of aligned embeddings.
#     - The parameter "shuffle" means a method is applied for a shuffled similarity matrix.
#     - If you want to limit the pairs that are applied GW alignment, please set “AlignRepresentations.pair_number_list”. (e.g. pair_number_list = [[0, 1], [0, 2]])

# %%
# Set the instance
align_representation = AlignRepresentations(
    representations_list=representations,
    pair_number_list='all',
    histogram_matching=False,
    config=config,
    metric="cosine",
)

# RSA
if data_select == "THINGS":
    sim_mat_format = "sorted"
    visualize_matrix = VisualizationConfig(
        figsize=(8, 6), title_size = 15, cmap = 'Blues',
        category_line_alpha = 0.5, draw_category_line=True
        )
    sim_mat = align_representation.show_sim_mat(
        sim_mat_format=sim_mat_format, 
        visualization_config=visualize_matrix,
        fig_dir=None,
        show_distribution=False,
        ticks='category'
    )

if data_select == "color":
    sim_mat_format = "default"
    visualize_matrix = VisualizationConfig(figsize=(8, 6), title_size = 15)
    sim_mat = align_representation.show_sim_mat(
        sim_mat_format = sim_mat_format, 
        visualization_config = visualize_matrix,
    )
    

# %%
align_representation.RSA_get_corr(method = 'all')

# %% [markdown]
# ## Computing GW Alignment.
# Optimization results are saved in the folder by "config.data_name" + "representations.name" vs "representation.name".  
# If you want to change the name of the saved folder, please change "config.data_name" and "representations.name".

# %%
if data_select == "THINGS":
    sim_mat_format = "sorted"
    visualize_matrix = VisualizationConfig(figsize=(8, 6), title_size = 15, category_line_color = 'C1')

    ot_list = align_representation.gw_alignment(
        results_dir = "../results",
        compute_OT = False,  # If the computation was done and no need for, turn "compute_OT" False, then OT plans calculated before is loaded.
        return_data = False,
        return_figure = True,
        OT_format = sim_mat_format, 
        visualization_config = visualize_matrix,
        show_log=True,
        fig_dir=None,
         ticks = 'category', # you can use "objects" or "category" or "None"
    )

if data_select == "color":
    visualize_matrix = VisualizationConfig(figsize=(10, 10), title_size = 15)

    align_representation.gw_alignment(
        results_dir = "../results",
        compute_OT = False,
        return_data = False,
        return_figure = True,
        OT_format = sim_mat_format, # "default"
        visualization_config = visualize_matrix,
    )

# %%
## Show how the GWD was optimized
# align_representation.get_optimization_log()

# %% [markdown]
# ## Align embeddings with OT plans and Visualize the aligned embeddings

# %%
## Calculate the accuracy of the optimized OT matrix
align_representation.calc_accuracy(top_k_list = [1, 5, 10], eval_type = "ot_plan")
align_representation.plot_accuracy(eval_type = "ot_plan", scatter = True)

# %%
## Calculate the matching rate of k-nearest neighbors of embeddings
## Matching rate of k-nearest neighbors 
align_representation.calc_accuracy(top_k_list = [1, 5, 10], eval_type = "k_nearest")
align_representation.plot_accuracy(eval_type = "k_nearest", scatter = True)

# %%


# %% [markdown]
# ## category level analysis 
# User can use this analysis if the dataset has category info.

# %%
## Calclate the category level accuracy
if data_select == "THINGS":
    align_representation.calc_category_level_accuracy(category_mat=category_mat)

# %%
# Set color labels and category data if exist.
if data_select == "THINGS":
    category_name_list = ["bird", "insect", "plant", "clothing",  "furniture", "fruit", "drink", "vehicle"]
    category_mat = pd.read_csv("../data/category_mat_manual_preprocessed.csv", sep = ",", index_col = 0)   
    object_labels, category_idx_list, num_category_list, category_name_list = get_category_data(category_mat, category_name_list, show_numbers = True)  
    
    align_representation.visualize_embedding(
        dim = 3,  
        category_name_list = category_name_list, 
        category_idx_list = category_idx_list, 
        num_category_list = num_category_list,
    )

# %%
if data_select == 'color':
    file_path = "../data/color_dict.csv"
    data_color = pd.read_csv(file_path)
    color_labels = data_color.columns.values
    visualization_config = VisualizationConfig(color_labels = color_labels, figsize = (15, 15), xlabel = "PC1", ylabel = "PC2", zlabel = "PC3", legend_size = 10)
    align_representation.visualize_embedding(dim = 3, visualization_config = visualization_config)


