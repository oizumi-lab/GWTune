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
