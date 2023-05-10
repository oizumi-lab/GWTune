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

#def RSA_get_corr(matrix1, matrix2, metric = "spearman"):
#    upper_tri1 = matrix1[np.triu_indices(matrix1.shape[0], k=1)]
#    upper_tri2 = matrix2[np.triu_indices(matrix2.shape[0], k=1)]
#    if metric == "spearman":
#        corr, _ = spearmanr(upper_tri1, upper_tri2)
#    elif metric == "pearson":
#        corr, _ = pearsonr(upper_tri1, upper_tri2)
#    return corr

#def shuffle_RDM(matrix):
#    """ 
#    The function for shuffling the lower trianglar matrix.
#    """
#    # Get the lower triangular elements of the matrix
#    lower_tri = matrix[np.tril_indices(matrix.shape[0], k=-1)]
#
#    # Shuffle the lower triangular elements
#    np.random.shuffle(lower_tri)
#
#    # Create a new matrix with the shuffled lower triangular elements
#    new_matrix = np.zeros_like(matrix)
#    new_matrix[np.tril_indices(new_matrix.shape[0], k=-1)] = lower_tri
#    new_matrix = new_matrix + new_matrix.T
#
#    return new_matrix

#def shuffle_matrix(matrix):
#    # Get a random permutation of the indices
#    indices = np.random.permutation(matrix.size)
#    
#    # Reshape the indices to match the matrix shape
#    indices = np.unravel_index(indices, matrix.shape)
#    
#    # Shuffle the matrix elements using the permutation
#    shuffled_matrix = matrix[indices]
#    shuffled_matrix = shuffled_matrix.reshape(matrix.shape)
#    # Return the shuffled matrix
#    return shuffled_matrix


#def shuffle_symmetric_block_mat(matrix, block_sizes):
#    new_mat = np.tril(matrix, k = -1)
#    cum_block_sizes = np.cumsum(block_sizes)
#    
#    for i, size_1 in enumerate(cum_block_sizes):
#        for j, size_2 in enumerate(cum_block_sizes[:i + 1]):
#            start1 = cum_block_sizes[i-1] if i !=0 else 0
#            start2 = cum_block_sizes[j-1] if j !=0 else 0
#            end1 = cum_block_sizes[i]
#            end2 = cum_block_sizes[j]
#            # Get the indices of the current block
#            block_indices = np.ix_(range(start1, end1), range(start2, end2))
#            block = new_mat[block_indices]
#            if i == j:
#                shuffled_mat = np.tril(shuffle_RDM(block), k = -1)
#            else:
#                shuffled_mat = shuffle_matrix(block)
#            new_mat[block_indices] = shuffled_mat
#    new_mat = new_mat + new_mat.T
#    return new_mat
    

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

# %%
