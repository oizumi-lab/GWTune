#%%
import sys
sys.path.append("../../")

import numpy as np
import ot


def randOrderedMatrix(n):
    """各行・各列に重複なしに[0,n]のindexを持つmatrixを作成

    Parameters
    ----------
    n : int
        行列のサイズ

    Returns
    -------
    np.ndarray
        重複なしのindexを要素に持つmatrix
    """
    matrix = np.zeros((n, n))
    rows = np.tile(np.arange(0,n),2)
    for i in range(n):
        matrix[i,:] = rows[i:i+n]

    r = np.random.choice(n,n,replace=False)
    c = np.random.choice(n,n,replace=False)
    matrix = matrix[r,:]
    matrix = matrix[:,c]
    return matrix.astype(int)


def initialize_matrix(n,ts=None):
    """gw alignmentのための行列初期化

    Parameters
    ----------
    n : int
        行列のサイズ

    Returns
    -------
    np.ndarray
        初期値
    """
    matrix = randOrderedMatrix(n)
    if ts is None:
        ts = np.random.uniform(0,1,n)
        ts = ts/(n*np.sum(ts))

    T = np.array([ts[idx] for idx in matrix])
    return T
