# %%
import sys
sys.path.append("../../")

from src.GW_alignment import my_entropic_gromov_wasserstein, my_entropic_gromov_wasserstein2
import numpy as np
import ot
from scipy.spatial.distance import squareform
from scipy.special import comb
from scipy.stats import pearsonr
import optuna

import matplotlib.pyplot as plt
import seaborn as sns

# %config InlineBackend.figure_formats = {'png', 'retina'} # for notebook?
plt.rcParams["font.size"] = 14

# %%
# functions


def sort_for_scaling(X):
    x = squareform(X)
    x_sorted = np.sort(x)
    x_inverse_idx = np.argsort(x).argsort()
    return x, x_sorted, x_inverse_idx


def histogram_matching(X, Y):
    # X, Y: dissimilarity matrices
    x, x_sorted, x_inverse_idx = sort_for_scaling(X)
    y, y_sorted, y_inverse_idx = sort_for_scaling(Y)

    y_t = x_sorted[y_inverse_idx]
    Y_t = squareform(y_t)  # transformed matrix
    return Y_t


def sc_plot(x, y, labels):
    plt.figure()
    plt.plot(x, y, '.')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()


def im_plot(X, Y, title_list):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, dm, t in zip(axes.reshape(-1), [X, Y], title_list):
        a = ax.imshow(dm)
        ax.set_title(t)
        cbar = fig.colorbar(a, ax=ax, shrink=0.7)
    plt.show()


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


def initialize_matrix(n):
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
    ts = np.random.uniform(0,1,n)
    ts = ts/(n*np.sum(ts))

    T = np.array([ts[idx] for idx in matrix])
    return T



def gw_alignment(X, Y, epsilon, random_init=False):
    n = X.shape[0]
    if random_init:
        T = initialize_matrix(n)
        p,q = ot.unif(n), ot.unif(n)
        gw, log = my_entropic_gromov_wasserstein2(
            C1=X, C2=Y, p=p, q=q, T=T, epsilon=epsilon, loss_fun="square_loss", verbose=True, log=True)
    else:
        p = ot.unif(n)
        q = ot.unif(n)
        gw, log = ot.gromov.entropic_gromov_wasserstein(
            X, Y, p, q, 'square_loss', epsilon=epsilon, log=True, verbose=True)

    plt.figure(figsize=(5, 5))
    sns.heatmap(gw, square=True)
    plt.show()
    return gw, log


# %%
# generate disimilarity matrix data
n = 1000  # number of elements
sigma = 1

np.random.seed(seed=42)
x_sim = np.random.uniform(0, 1, size=comb(n, 2, exact=True))
np.random.seed(seed=0)
y_sim = 2 * x_sim + np.random.uniform(0, sigma, size=comb(n, 2, exact=True))

X = squareform(x_sim)  # disimilarity matrix 1
Y = squareform(y_sim)  # disimilarity matrix 2

# %%
# RSA correlation
x_flat = squareform(X)
y_flat = squareform(Y)

corr, _ = pearsonr(x_flat, y_flat)
print(f'pearson r = {corr}')

sc_plot(x_flat, y_flat, ['X', 'Y'])
im_plot(X, Y, ['X', 'Y'])

# histogram alignment
Y_t = histogram_matching(X, Y)
im_plot(X, Y_t, ['X', 'transformed Y'])

# %%
# Normal GW alignment (without histogram matching)
epsilon = 0.0045
gw, log = gw_alignment(X, Y, epsilon=epsilon, random_init=False)
gwd = log['gw_dist']
print(f'With histogram matching: GWD = {gwd}')

#%%
# GW alignment after histogram matching
epsilon = 0.0005
gw_t, log_t = gw_alignment(X, Y_t, epsilon=epsilon, random_init=False)
gwd_t = log_t['gw_dist']
print(f'With histogram matching: GWD = {gwd_t}')  # %%

# %%
