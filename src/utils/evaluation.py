import numpy as np
from scipy.spatial import distance

def calc_correct_rate_ot_plan(Pi, top_n):
    count = 0
    for i in range(Pi.shape[0]):
        idx = np.argsort(-Pi[i, :])
        count += (idx[:top_n] == i).sum()    
    count /= Pi.shape[0]
    count *= 100
    
    return count

def calc_correct_rate_top_n(Pi, top_n, embedding = None, RDM = None, components = False, metric = "cosine"):
    """対応する点が正解の近傍top_nに入っていれば正解

    Args:
        Pi (array): shape (m, n) 
            subject → pivotの対応関係
        top_n (int): top n neighbors
        embedding (array): shape (n, n_dim)
            pivotのembedding
        RDM (array): shape(n, n)
            pivotのRDM
        components : Trueのとき, Piの成分をそのまま足しあわせる

    Returns:
        correct_rate: correct rate
    """
    if RDM is None:
        RDM = distance.cdist(embedding, embedding, metric = metric)
    if components:
        count = 0
        for i in range(Pi.shape[0]):
            idx_pivot = np.argsort(RDM[i, :])
            count += (Pi[i,:][idx_pivot][:top_n]).sum()
        correct_rate = count * 100
    else:
        count = 0
        for i in range(Pi.shape[0]):
            idx = np.argmax(Pi[i, :])
            idx_pivot = np.argsort(RDM[i, :])
            count += (idx == idx_pivot[:top_n]).sum()
        correct_rate = count / Pi.shape[0] * 100
    return correct_rate

def pairwise_k_nearest_matching_rate(embedding_1, embedding_2, top_n, metric = "cosine"):
    """Count it if a point of embedding_1 is in k-nearest neighbors of a corresponding point of embedding_2

    Args:
        embedding1 (_type_): _description_
        embedding2 (_type_): _description_
        top_n (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Compute distances between each points
    dist_mat = distance.cdist(embedding_1, embedding_2, metric)

    # Get sorted indices 
    sorted_idx = np.argsort(dist_mat, axis = 1)

    # Get the same colors and count k-nearest
    count = 0
    for i in range(embedding_1.shape[0]):
        count += (sorted_idx[i, :top_n]  == i).sum() 
    count /= embedding_1.shape[0]
    count *= 100 # return percentage

    return count

def pairwise_k_nearest_mappings(embedding_1, embedding_2, top_n):
    # Compute distances between each points
    dist_mat = distance.cdist(embedding_1, embedding_2, "euclidean")
    
    # Get sorted indices 
    sorted_idx = np.argsort(dist_mat, axis = 1)
    
    mapping = np.zeros_like(dist_mat)
    for i in range(embedding_1.shape[0]):
        for k in range(top_n):
            mapping[i, sorted_idx[i, k]] = 1/(dist_mat[i, sorted_idx[i, k]] + 1e-10)
        mapping[i] /= mapping[i].sum()
        
    return mapping