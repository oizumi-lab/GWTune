#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import pandas as pd
from scripts.align_representations import Optimization_Config, Representation, Pairwise_Analysis, Align_Representations
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
n_representations = 4
metric = "euclidean"

# representations list that will be used in Align_Representations
representations = list()

# Load data and create representations instance
for i in range(n_representations):
    name = f"Group{i+1}"
    embedding = np.load(f"../data/THINGS_embedding_Group{i+1}.npy")[0]
    
    representation = Representation(name = name, embedding = embedding, metric = metric)
    representations.append(representation)
    
#%%
'''
Set the parameters for the optimazation of GWOT
'''
config = Optimization_Config(data_name = data_select, 
                             delete_study = True, 
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
Unsupervised alignment between Representations
    - The object has methods for RSA, GW-alignment, evaluation of the alignment and visalization of aligned embeddings.
    - The parameter "shuffle" means a method is applied for a shuffled similarity matrix.
'''
# Set the instance
align_representation = Align_Representations(representations_list = representations, config = config)

# RSA
align_representation.show_sim_mat()
align_representation.RSA_get_corr(shuffle = False)

'''
GW alignment
'''
## If no need for computation, turn load_OT True, then OT plans calculated before is loaded.
align_representation.gw_alignment(pairnumber_list = [1, 2], shuffle = False, load_OT = False)

## Calculate the accuracy of the optimized OT matrix
align_representation.calc_top_k_accuracy(k_list = [1, 5, 10], shuffle = False)
align_representation.plot_accuracy(eval_type = "ot_plan", shuffle = False, scatter = True) # If scatter is True, the scatter plot is employed.

#%%
'''
Align embeddings with OT plans
'''
## Calculate the matching rate of k-nearest neighbors of embeddings
align_representation.calc_k_nearest_matching_rate(k_list = [1, 5, 10], metric = metric)
align_representation.plot_accuracy(eval_type = "k_nearest", shuffle = False, scatter = True)

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

align_representation.visualize_embedding(dim = 3, color_labels = color_labels, category_name_list = category_name_list, category_idx_list = category_idx_list, category_num_list = category_num_list)