#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import pandas as pd
import pickle as pkl
from src.align_representations import Representation, Pairwise_Analysis, Align_Representations, Optimization_Config, Visualize_Matrix
from src.utils.utils_functions import get_category_idx

#%%
### load data
# you can choose the following data
# 'DNN': representations of 2000 imagenet images in AlexNet and VGG
# 'color': human similarity judgements of 93 colors for 5 paricipants groups
# 'face': human similarity judgements of 16 faces, attended vs unattended condition in the same participant
# 'THINGS' : human similarity judgements of 1854 objects for 4 paricipants groups
data_select = "THINGS"

'''
Set Representations
    - A Representation needs a name and either an embedding or a similarity matrix.
'''
# Parameters
n_representations = 4 # Set the number of representations. This number must be equal to or less than the number of groups.
metric = "euclidean"

# representations list that will be used in Align_Representations
representations = list()

# Load data and create representations instance
if data_select == 'color':
    data_path = '../data/num_groups_5_seed_0_fill_val_3.5.pickle'
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    sim_mat_list = data["group_ave_mat"]
    for i in range(n_representations):
        name = f"Group{i+1}"
        sim_mat = sim_mat_list[i]
        representation = Representation(name = name, sim_mat = sim_mat)
        representations.append(representation)
elif data_select == "THINGS":
    for i in range(n_representations):
        name = f"Group{i+1}"
        embedding = np.load(f"../data/THINGS_embedding_Group{i+1}.npy")[0]
        category_mat = pd.read_csv("../data/category_mat_manual_preprocessed.csv", sep = ",", index_col = 0)  
        representation = Representation(name = name, embedding = embedding, metric = metric, category_mat = category_mat)
        representations.append(representation)
    
#%%
'''
Set the parameters for the optimazation of GWOT
'''
config = Optimization_Config(data_name = data_select, 
                             delete_study = False, 
                             device = 'cpu',
                             to_types = 'numpy',
                             n_jobs = 1,
                             init_plans_list = ['random'],
                             num_trial = 4,
                             n_iter = 1,
                             max_iter = 200,
                             sampler_name = 'tpe',
                             eps_list = [1, 10], # [1, 10] for THINGS data, [0.02, 0.2] for colors data
                             eps_log = True,
                             pruner_name = 'hyperband',
                             pruner_params = {'n_startup_trials': 1, 'n_warmup_steps': 2, 'min_resource': 2, 'reduction_factor' : 3}
                             )

'''
Set the parameters for visualizing matrices
'''
visualize_matrix = Visualize_Matrix()
#%%
'''
Unsupervised alignment between Representations
    - The object has methods for RSA, GW-alignment, evaluation of the alignment and visalization of aligned embeddings.
    - The parameter "shuffle" means a method is applied for a shuffled similarity matrix.
'''
# Set the instance
align_representation = Align_Representations(representations_list = representations, config = config)

# RSA
sim_mat = align_representation.show_sim_mat(returned = "figure", sim_mat_format = "sorted", visualize_matrix = visualize_matrix)#fig_dir = "../figures")
align_representation.RSA_get_corr()

#%%
'''
GW alignment
'''
## If no need for computation, turn load_OT True, then OT plans calculated before is loaded.
align_representation.gw_alignment(load_OT = True, returned = "figure", OT_format = "sorted", visualize_matrix = visualize_matrix, show_log = True)

## Calculate the accuracy of the optimized OT matrix
align_representation.calc_accuracy(top_k_list = [1, 5, 10], eval_type = "ot_plan")
align_representation.plot_accuracy(eval_type = "ot_plan", scatter = True)

## Calclate the category level accuracy
align_representation.calc_category_level_accuracy()
#%%
'''
Align embeddings with OT plans
'''
## Calculate the matching rate of k-nearest neighbors of embeddings
## Matching rate of k-nearest neighbors 
align_representation.calc_accuracy(top_k_list = [1, 5, 10], eval_type = "k_nearest")
align_representation.plot_accuracy(eval_type = "k_nearest", scatter = True)

'''
Visualize the aligned embeddings
'''
# Set color labels and category data if exist.
if data_select == "THINGS":
    color_labels = None
    category_name_list = ["bird", "insect", "plant", "clothing",  "furniture", "fruit", "drink", "vehicle"]
    category_mat = pd.read_csv("../data/category_mat_manual_preprocessed.csv", sep = ",", index_col = 0)   
    category_idx_list, category_num_list = get_category_idx(category_mat, category_name_list, show_numbers = True)  
    align_representation.visualize_embedding(dim = 3, color_labels = color_labels, category_name_list = category_name_list, category_idx_list = category_idx_list, category_num_list = category_num_list)#, fig_dir = "../figures")
elif data_select == "color":
    file_path = "../data/color_dict.csv"
    data_color = pd.read_csv(file_path)
    color_labels = data_color.columns.values
    align_representation.visualize_embedding(dim = 3, color_labels = color_labels)#, fig_dir = "../figures")
# %%
