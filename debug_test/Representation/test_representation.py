# %%
import os, sys
import numpy as np
import pickle as pkl
import pandas as pd
import torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.getcwd(), '../../'))
from src.align_representations import Representation, VisualizationConfig

representations = list()
data_select = "THINGS"

# %%
n_representations = 4 

# %%
category_mat = pd.read_csv("../../data/THINGS/category_mat_manual_preprocessed.csv", sep = ",", index_col = 0)

# %%
# sorted_category_mat = category_mat.sort_index(axis=1)

# %%
from src.utils.utils_functions import get_category_data, sort_matrix_with_categories
object_labels, category_idx_list, num_category_list, category_name_list = get_category_data(category_mat)

i = 0 
# metric = "euclidean"
# metric = "cosine" 
metric = "dot"
name = f"Group{i+1}" # the name of the representation
embedding = np.load(f"../../data/THINGS/THINGS_embedding_Group{i+1}.npy")[0]

embedding = torch.from_numpy(embedding)
main_results_dir = f"../../results/{data_select}"

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
vis = VisualizationConfig(
    title_size=10,
    xlabel="x",
    ylabel="y",
)

# %%
rep.plot_sim_mat(return_sorted = False, visualization_config=vis)

# %%
rep.plot_sim_mat(return_sorted = True, visualization_config=vis)

# %%
rep.show_sim_mat_distribution(bins=10, visualization_config=vis)


# %%
### define the category info from label data in the validation dataset.
lab_path = '../../data/DNN/label.pt'
lab = torch.load(lab_path, weights_only=False).to('cpu').numpy()

### category_mat needs to be an one-hot encoding. 
category_mat = pd.get_dummies(lab)

category_mat.columns = np.load('../../data/DNN/label_name.npy')

from src.utils.utils_functions import get_category_data, sort_matrix_with_categories 
object_labels, category_idx_list, num_category_list, category_name_list = get_category_data(category_mat = category_mat)

model_name = 'ResNet50'

emb_path = f'../../data/DNN/{model_name}_emb.pt'
cos_path = f'../../data/DNN/{model_name}_cosine.pt'

emb = torch.load(emb_path, weights_only=False).to('cpu')


#%%
model_rep = Representation(
    name=model_name, 
    embedding=emb, 
    metric='cosine',
    get_embedding=False,
    object_labels=object_labels,
    category_name_list=category_name_list,
    category_idx_list=category_idx_list,
    num_category_list=num_category_list, 
    func_for_sort_sim_mat=sort_matrix_with_categories,
    save_conditional_rdm_path=None,
)
    
model_rep.plot_sim_mat(return_sorted = False)
model_rep.show_sim_mat_distribution(bins=10, visualization_config=vis)

#%%
model_rep = Representation(
    name=model_name, 
    embedding=emb, 
    metric='euclidean',
    get_embedding=False,
    object_labels=object_labels,
    category_name_list=category_name_list,
    category_idx_list=category_idx_list,
    num_category_list=num_category_list, 
    func_for_sort_sim_mat=sort_matrix_with_categories,
    save_conditional_rdm_path=None,
)
    
model_rep.plot_sim_mat(return_sorted = False)
model_rep.show_sim_mat_distribution(bins=10, visualization_config=vis)

#%%
model_rep = Representation(
    name=model_name, 
    embedding=emb, 
    metric='dot',
    get_embedding=False,
    object_labels=object_labels,
    category_name_list=category_name_list,
    category_idx_list=category_idx_list,
    num_category_list=num_category_list, 
    func_for_sort_sim_mat=sort_matrix_with_categories,
    save_conditional_rdm_path=None,
)
    
model_rep.plot_sim_mat(return_sorted = False)
model_rep.show_sim_mat_distribution(bins=10, visualization_config=vis)

#%%
model_rep = Representation(
    name=model_name, 
    embedding=emb, 
    metric='dot',
    get_embedding=False,
    object_labels=object_labels,
    category_name_list=category_name_list,
    category_idx_list=category_idx_list,
    num_category_list=num_category_list, 
    func_for_sort_sim_mat=sort_matrix_with_categories,
    save_conditional_rdm_path="../../data/DNN/",
)
    
model_rep.plot_sim_mat(return_sorted = False)
model_rep.show_sim_mat_distribution(bins=10, visualization_config=vis)

#%%