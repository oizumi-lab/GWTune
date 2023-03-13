import numpy as np
from scipy.spatial import distance
import ot


def procrustes(embedding_1, embedding_2, Pi):
    """embedding_2をembedding_1に最も近づける回転行列Qを求める

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
    U, S, Vt = np.linalg.svd(
        np.matmul(embedding_2.T, np.matmul(Pi, embedding_1)))
    Q = np.matmul(U, Vt)
    new_embedding_2 = np.matmul(embedding_2, Q)

    return Q, new_embedding_2


def aligned_wasserstein(embedding_1, embedding_2, Pi):
    """After transportation matrix was obtained, now get a new assignment using wasserstein

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
    # Procrustes
    Q, new_embedding_2 = procrustes(embedding_1, embedding_2, Pi)

    # Wassserstein
    M = distance.cdist(new_embedding_2, embedding_1, "euclidean")
    new_Pi = ot.emd(np.sum(Pi, axis=0), np.sum(Pi, axis=1), M)

    return new_Pi


def calc_correct_rate_same_dim(coupling):
    count = 0
    for i in range(coupling.shape[0]):
        idx = np.argmax(coupling[i, :])
        if i == idx:
            count += 1
    correct_rate = count / coupling.shape[0] * 100
    return correct_rate


def calc_correct_rate_top_n(Pi, top_n, embedding=None, RDM=None, components=False):
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
        RDM = distance.cdist(embedding, embedding, metric="cosine")
    if components:
        count = 0
        for i in range(Pi.shape[0]):
            idx_pivot = np.argsort(RDM[i, :])
            count += (Pi[i, :][idx_pivot][:top_n]).sum()
        correct_rate = count * 100
    else:
        count = 0
        for i in range(Pi.shape[0]):
            idx = np.argmax(Pi[i, :])
            idx_pivot = np.argsort(RDM[i, :])
            count += (idx == idx_pivot[:top_n]).sum()
        correct_rate = count / Pi.shape[0] * 100
    return correct_rate