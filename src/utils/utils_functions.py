#%%
import numpy as np
from scipy.spatial import distance
import ot
import pandas as pd
from scipy.stats import spearmanr, pearsonr
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

def procrustes(embedding_1, embedding_2, Pi):
    """
    embedding_2をembedding_1に最も近づける回転行列Qを求める

    Args:
        embedding_1 : shape (n_1, m)
        embedding_2 : shape (n_2, m)
        Pi : shape (n_2, n_1) 
            Transportation matrix of 2→1
        
    Returns:
        Q : shape (m, m) 
            Orthogonal matrix 
        new_embedding_2 : shape (n_2, m)
    """
    U, S, Vt = np.linalg.svd(np.matmul(embedding_2.T, np.matmul(Pi, embedding_1)))
    Q = np.matmul(U, Vt)
    new_embedding_2 = np.matmul(embedding_2, Q)
    
    return Q, new_embedding_2

def get_category_idx(category_mat, category_name_list, show_numbers = False):
    if show_numbers:
        object_numbers = list()
        for column in category_mat.columns:
            num = (category_mat[column].values == 1).sum()
            object_numbers.append(num)
        num_each_category = pd.DataFrame(object_numbers, index = category_mat.columns, columns = ["Number"])
        print(num_each_category)
        
    category_idx_list = []
    n_category_list = []
    for category in category_name_list:
        category_idx = category_mat[category].values == 1
        category_idx_list.append(category_idx)
        n_category_list.append(category_idx.sum())
    
    return category_idx_list, n_category_list 

def sort_matrix_with_categories(matrix, category_idx_list):
    new_mat_blocks = []
    for i in range(len(category_idx_list)):
        row_blocks = []
        for j in range(len(category_idx_list)):
            block = matrix[category_idx_list[i]][:, category_idx_list[j]]
            row_blocks.append(block)
        new_mat_blocks.append(np.concatenate(row_blocks, axis=1))
    
    new_mat = np.concatenate(new_mat_blocks, axis=0)
    return new_mat


# %%
