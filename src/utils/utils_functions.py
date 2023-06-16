#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random

def fix_random_seed(seed = 42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def get_category_data(category_mat: pd.DataFrame, category_name_list=None, show_numbers=False):
    if category_name_list is None:
        new_category_name_list = category_mat.columns.tolist()
    else:
        new_category_name_list = category_name_list
    
    category_idx_list = []
    category_num_list = []
    object_labels = []
    
    for category in new_category_name_list:
        category_idx = category_mat[category].values == 1
        
        category_idx_list.append(np.where(category_idx)[0])
        category_num_list.append(category_idx.sum())
        object_labels.extend(category_mat.index[category_idx].tolist())
    
    if show_numbers:
        num_each_category = pd.DataFrame(category_num_list, index = new_category_name_list, columns = ["Number"])
        print(num_each_category)

    return object_labels, category_idx_list, category_num_list, new_category_name_list

def sort_matrix_with_categories(matrix, category_idx_list):
    new_mat_blocks = []
    for i in category_idx_list:
        row_blocks = []
        for j in category_idx_list:
            block = matrix[i][:, j]
            row_blocks.append(block)
        new_mat_blocks.append(np.concatenate(row_blocks, axis=1))
    
    new_mat = np.concatenate(new_mat_blocks, axis=0)
    return new_mat
 
# %%
if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    category_mat = pd.read_csv("../../data/category_mat_manual_preprocessed.csv", sep = ",", index_col = 0)  
    object_labels, category_idx_list, category_num_list, new_category_name_list = get_category_data(category_mat = category_mat)

    a = torch.load('../../data/mean_all_images_vgg19.pt').to('cpu').numpy()
    b = sort_matrix_with_categories(a, category_idx_list)
    
    plt.figure()
    plt.subplot(121)

    plt.title("source")
    plt.imshow(a)

    plt.subplot(122)
    plt.title("target")
    plt.imshow(b)
    plt.tight_layout()
    plt.show()
    
# %%
