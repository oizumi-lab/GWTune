# Standard Library
import itertools
import os

# Third Party Library
import numpy as np
import optuna
import ot
from joblib import parallel_backend
from ot.backend import get_backend
from ot.bregman import sinkhorn
from ot.gromov import gwggrad, gwloss, init_matrix
from ot.lp import emd, emd_1d
from ot.optim import cg
from ot.utils import UndefinedParameter, check_random_state, dist, list_to_array, unif
from scipy.spatial import distance
from scipy.spatial.distance import squareform
from sklearn.metrics import confusion_matrix, accuracy_score

def sort_for_scaling(X):
    x = squareform(X, checks=False)
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
    rows = np.tile(np.arange(0, n), 2)
    for i in range(n):
        matrix[i, :] = rows[i : i + n]

    r = np.random.choice(n, n, replace=False)
    c = np.random.choice(n, n, replace=False)
    matrix = matrix[r, :]
    matrix = matrix[:, c]
    return matrix.astype(int)


def initialize_matrix(n, ts=None):
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
        ts = np.random.uniform(0, 1, n)
        ts = ts / (n * np.sum(ts))

    T = np.array([ts[idx] for idx in matrix])
    return T


def calcurate_gwd(C1, C2, p, q, T, loss_fun):
    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)
    gwd = gwloss(constC, hC1, hC2, T)
    return gwd


def correct_rate_mv(gw):
    y_pred = np.argmax(gw, axis=1)
    y_true = np.arange(0, len(gw))
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return acc, cm


class HistogramMatching:
    def __init__(self, X, Y, database_uri, study_name):
        self.X, self.Y = X, Y
        self.database_uri = database_uri
        self.study_name = study_name

        self.x, self.x_sorted, self.x_inverse_idx = sort_for_scaling(X)
        self.y, self.y_sorted, self.y_inverse_idx = sort_for_scaling(Y)

        # hyperparameter
        self.alpha1, self.alpha2 = 1, 1
        self.lam1, self.lam2 = 1, 1
        self.fixed_array = "X"

    def sort_for_scaling(self, data):
        x = squareform(data)
        x_sorted = np.sort(x)
        x_inverse_idx = np.argsort(x).argsort()
        return x, x_sorted, x_inverse_idx

    def pointwise_matching(self):
        y_t = self.x_sorted[self.y_inverse_idx]
        Y_t = squareform(y_t)
        return Y_t

    def model_normalize(self, v, alpha, lam):
        data = alpha * ((np.power(1 + v, lam) - 1) / lam)
        return data

    def __call__(self, trial):
        fixed_aray = trial.suggest_categorical("fixed_array", ["X", "Y"])
        lam1 = trial.suggest_float("lam1", 1e-6, 1e2, log=True)
        lam2 = trial.suggest_float("lam2", 1e-6, 1e2, log=True)

        if fixed_aray == "X":
            alpha1 = trial.suggest_float("alpha1", 1, 1, log=True)
            alpha2 = trial.suggest_float("alpha2", 1e-2, 1e1, log=True)
        else:
            alpha1 = trial.suggest_float("alpha1", 1e-2, 1e1, log=True)
            alpha2 = trial.suggest_float("alpha2", 1, 1, log=True)

        x_data, y_data = self.model_normalize(
            self.x_sorted, alpha1, lam1
        ), self.model_normalize(self.y_sorted, alpha2, lam2)

        l = ot.emd2_1d(x_data, y_data)
        return l

    def run_yeojohnson_study(
        self, database_uri, study_name, concurrency=20, num_trial=5000
    ):
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            sampler=optuna.samplers.RandomSampler(seed=42),
            storage=database_uri,
            load_if_exists=False,
        )
        # max_cpu count
        max_cpu = os.cpu_count()
        if not concurrency <= max_cpu:
            raise ValueError("concurrency > max_cpu")

        # with parallel_backend("multiprocessing", n_jobs = n_gpu):
        study.optimize(self, n_trials=num_trial, n_jobs=concurrency)

        # set parameters
        for key, value in study.best_params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        return study

    def yeojohnson_matching(self, params=None):
        if params is None:
            study = optuna.load_study(
                storage=self.database_uri, study_name=self.study_name
            )
            params = study.best_params
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        x_t = self.model_normalize(self.x, self.alpha1, self.lam1)
        y_t = self.model_normalize(self.y, self.alpha2, self.lam2)

        X_t = squareform(x_t)
        Y_t = squareform(y_t)
        return X_t, Y_t
