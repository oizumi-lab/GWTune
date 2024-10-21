# %%
import os, sys
import numpy as np
import torch
import pickle as pkl
import pandas as pd
from src.align_representations import Representation

# %%
def sample_data(data_select):
    representations = list()

    if data_select == "THINGS":
        eps_list = [1,10]
        category_mat = pd.read_csv("../../data/THINGS/category_mat_manual_preprocessed.csv", sep = ",", index_col = 0)

        from src.utils.utils_functions import get_category_data, sort_matrix_with_categories
        object_labels, category_idx_list, num_category_list, category_name_list = get_category_data(category_mat)

        n_representations = 4
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
        
        return representations, eps_list
    
    elif data_select == "color":
        n_representations = 4 # Set the number of the instanses of "Representation". This number must be equal to or less than the number of the groups. 5 is the maximum for this data.
        metric = "euclidean" # Please set the metric that can be used in "scipy.spatical.distance.cdist()".
        
        eps_list = [0.02, 0.2]
        data_path = '../../data/color/num_groups_5_seed_0_fill_val_3.5.pickle'
        with open(data_path, "rb") as f:
            data = pkl.load(f)
        sim_mat_list = data["group_ave_mat"]
        for i in range(n_representations):
            name = f"Group{i+1}" # "name" will be used as a filename for saving the results
            sim_mat = sim_mat_list[i] # the dissimilarity matrix of the i-th group
            # make an instance "Representation" with settings 
            representation = Representation(
                name=name, 
                metric=metric,
                sim_mat=sim_mat,  #: np.ndarray
                embedding=None,   #: np.ndarray 
                get_embedding=True, # If true, the embeddings are computed from the dissimilarity matrix automatically using the MDS function. Default is False. 
                MDS_dim=3, # If "get embedding" is True, please set the dimensions of the embeddings.
                object_labels=None,
                category_name_list=None,
                num_category_list=None,
                category_idx_list=None,
                func_for_sort_sim_mat=None,
        ) 
            representations.append(representation)
        
        return representations, eps_list

    elif data_select == "DNN":
        ### define the category info from label data in the validation dataset.
        lab_path = '../../data/DNN/label.pt'
        lab = torch.load(lab_path).to('cpu').numpy()
        eps_list = [1e-4, 1e-2]
        ### category_mat needs to be an one-hot encoding. 
        category_mat = pd.get_dummies(lab)

        category_mat.columns = np.load('../../data/DNN/label_name.npy')

        from src.utils.utils_functions import get_category_data, sort_matrix_with_categories 
        object_labels, category_idx_list, num_category_list, category_name_list = get_category_data(category_mat = category_mat)

        model_name_list = ['ResNet50', 'VGG19']

        for model_name in model_name_list:
            
            emb_path = f'../../data/DNN/{model_name}_emb.pt'
            cos_path = f'../../data/DNN/{model_name}_cosine.pt'
            
            emb = torch.load(emb_path).to('cpu').numpy()
            sim_mat = torch.load(cos_path).to('cpu').numpy()

            model_rep = Representation(
                name=model_name, 
                sim_mat=sim_mat, 
                embedding=emb, 
                get_embedding=False,
                object_labels=object_labels,
                category_name_list=category_name_list,
                category_idx_list=category_idx_list,
                num_category_list=num_category_list, 
                func_for_sort_sim_mat=sort_matrix_with_categories,
            )

            representations.append(model_rep)

        return representations, eps_list

    elif data_select == "AllenBrain":
        eps_list = [1e-05, 1e-01]
        
        for name in ["pseudo_mouse_A", "pseudo_mouse_B"]:
            emb = np.load(f"../../data/AllenBrain/{name}_emb.npy")
            representation = Representation(
                name=name,
                embedding=emb,  # the dissimilarity matrix will be computed with this embedding.
                metric="cosine",
                get_embedding=False, # If there is the embeddings, plese set this variable "False".
                object_labels=np.arange(emb.shape[0]) 
            )
            representations.append(representation)
        
        return representations, eps_list