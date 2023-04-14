#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import pandas as pd
from n_group_analysis import Optimization_Config, Subject_Group, Pairwise_Analysis, N_Group_Analysis
from src.utils.utils_functions import get_category_idx
#%%
'''
Set the parameters for the optimazation of GWOT
'''
config = Optimization_Config(delete_study = True, 
                             device = 'cpu',
                             to_types = 'numpy',
                             n_jobs = 4,
                             init_plans_list = ['random'],
                             num_trial = 4,
                             n_iter = 1,
                             max_iter = 200,
                             sampler_name = 'tpe',
                             eps_list = [1, 10],
                             eps_log = True,
                             pruner_name = 'hyperband',
                             pruner_params = {'n_startup_trials': 1, 'n_warmup_steps': 2, 'min_resource': 2, 'reduction_factor' : 3}
                             )

#%%
'''
Create subject groups
    - A Subject_Group needs a name and either an embedding or a similarity matrix.
'''
# Parameters
n_group = 4
metric = "euclidean"

# Subject groups list that will be used in N_Group_Analysis
subject_groups = list()

# Load data and create subject groups instance
for i in range(n_group):
    name = f"Group{i+1}"
    embedding = np.load(f"../data/THINGS_embedding_Group{i+1}.npy")[0]
    
    group = Subject_Group(name = name, embedding = embedding, metric = metric)
    subject_groups.append(group)

#%%
'''
N groups analysis
    - The object has methods for RSA, GW-alignment, evaluation of the alignment and visalization of aligned embeddings.
    - The parameter "shuffle" means a method is applied for a shuffled similarity matrix.
'''
# Set the instance
n_group_analysis = N_Group_Analysis(subject_groups_list = subject_groups, config = config)

# RSA
n_group_analysis.show_sim_mat()
n_group_analysis.RSA_get_corr(shuffle = False)

# GW alignment
## If no need for computation, turn load_OT True, then OT plans calculated before is loaded.
n_group_analysis.gw_alignment(shuffle = False, load_OT = False)

#%%
'''
Evaluate the accuracy
'''
## Calculate the accuracy of the optimized OT matrix
n_group_analysis.calc_top_k_accuracy(k_list = [1, 5, 10], shuffle = False)
n_group_analysis.plot_accuracy(eval_type = "ot_plan", shuffle = False, scatter = True) # If scatter is True, the scatter plot is employed.

## Calculate the matching rate of k-nearest neighbors of embeddings
n_group_analysis.calc_k_nearest_matching_rate(k_list = [1, 5, 10], metric = metric)
n_group_analysis.plot_accuracy(eval_type = "k_nearest", shuffle = False, scatter = True)

#%%
'''
Visualize the aligned embeddings
'''
# Load the coarse categories data
## No need for this step if there is no coarse categories data
category_data = True
if category_data:
    color_labels = None
    category_name_list = ["bird", "insect", "plant", "clothing",  "furniture", "fruit", "drink", "vehicle"]
    category_mat = pd.read_csv("../data/category_mat_manual_preprocessed.csv", sep = ",", index_col = 0)   
    category_idx_list, category_num_list = get_category_idx(category_mat, category_name_list, show_numbers = True)  
else:
    color_labels = [] # Set color labels

n_group_analysis.visualize_embedding(dim = 3, color_labels = color_labels, category_name_list = category_name_list, category_idx_list = category_idx_list, category_num_list = category_num_list)