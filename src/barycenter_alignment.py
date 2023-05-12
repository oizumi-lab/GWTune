#%%
# Standard Library
import gc
import math
import os
import sys
import time
import warnings
from typing import Any, Union, Optional

# Third Party Library
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import optuna
import ot
import pandas as pd

# warnings.simplefilter("ignore")
import seaborn as sns
import torch
from joblib import parallel_backend
from scipy.spatial import distance
from tqdm.auto import tqdm

# First Party Library
from utils.backend import Backend
from utils.gw_optimizer import load_optimizer
from utils.init_matrix import InitMatrix


def fixed_weight_barycenter(
    embedding_list, Y_init, bi_list, a, lambdas, tol=1e-8, metric="euclidean", reg=1e-2, maxiter=20, bregmanmaxiter=30
):
    """重みを固定したwasserstein barycenter

    Args:
        embedding_list : shape (n_group, n, m)
        Y_init : shape (n, m)
        bi_list : shape (n_group, n)
            uniform distribution
        a : shape (n, )
            uniform distribution
        lambdas : shape (n_group, )
            uniform distribution
        tol (_type_, optional):  Defaults to 1e-8.
        metric (str, optional):  Defaults to 'euclidean'.
        reg (_type_, optional):  Defaults to 1e-2.
        maxiter (int, optional):  Defaults to 20.
        bregmanmaxiter (int, optional):  Defaults to 30.

    Returns:
        Y : shape (n, m)
            the location of the barycenter
    """
    displacement = 1
    niter = 0
    Y = Y_init

    while displacement > tol and niter < maxiter:
        Y_prev = Y
        Tsum = np.zeros(Y.shape)

        for i in range(0, len(bi_list)):
            M = distance.cdist(Y, embedding_list[i], metric=metric)
            # T = ot.sinkhorn(a, bi[i], M, reg)
            T = ot.emd(a, bi_list[i], M)
            Tsum = Tsum + lambdas[i] * np.reshape(1.0 / a, (-1, 1)) * np.matmul(T, embedding_list[i])

        displacement = np.sum(np.square(Tsum - Y))

        Y = Tsum
        niter += 1

    return Y


class Barycenter_alignment:
    def __init__(
        self,
        n_group: int,
        embeddings: Union[dict, np.ndarray],
        pivot: int,
        name_list: Optional[Union[list, np.ndarray]] = None,
    ) -> None:
        self.n_group = n_group
        if isinstance(embeddings, dict):
            self.name_list = list(embeddings.keys())
            self.embedding_list = self._dict_to_numpy(embeddings)
        else:
            self.name_list = name_list
            self.embedding_list = embeddings

        self.pivot = pivot
        self.name_list = name_list
        self.embedding_barycenter = 0

        # parameters for barycenter alignment
        self.bi_list = [ot.unif(embedding_list[0].shape[0]) for _ in range(n_group)]
        self.a = ot.unif(embedding_list[0].shape[0])
        self.lambdas = ot.unif(n_group)

        # parameters for using optuna
        self.DATABASE_URL = DATABASE_URL  #'mysql+pymysql://root@localhost/takeda_GWOT'

    def _dict_to_numpy(self, embeddings: dict):
        """
        Convert dictionary embeddings to numpy
        """
        dist_l = []
        for dist in embeddings.values():
            dist_l.append(dist)

        return np.r_[dist_l]

    def gw_alignment_to_pivot(
        self,
        optuna=True,
        n_init_plan=None,
        epsilons=None,
        epsilon_range=None,
        n_trials=None,
        n_jobs=None,
        init_diag=True,
    ):
        """GW alignment to the pivot

        Args:
            optuna : If True, use optuna
            n_init_plan : the number of initial plans
            epsilons : list of epsilons. It is needed when you don't use optuna
            epsilon_range (list): search range of epsilons
            n_trials (int): the number of seaching for epsilons
            n_jobs (int): the number of cores of cpu used
            init_diag (bool, optional): If True, initial plans contain the diagonal plan for each epsilons. Defaults to True.

        Returns:
            Pi_list (array-like): List of transportation matrices
        """
        Pi_list = list()
        for i in range(self.n_group):
            if i == self.pivot:
                Pi = np.diag([1 / self.embedding_list[0].shape[0] for _ in range(self.embedding_list[0].shape[0])])
                min_epsilon = np.nan
                correct_rate = 100
                Pi_list.append(Pi)

            else:
                RDM_i = distance.cdist(self.embedding_list[i], self.embedding_list[i], metric="cosine")
                RDM_pivot = distance.cdist(
                    self.embedding_list[self.pivot], self.embedding_list[self.pivot], metric="cosine"
                )
                if optuna:
                    # Set the instance
                    if self.name_list is not None:
                        study_name = "{} vs {}".format(self.name_list[i], self.name_list[self.pivot])
                    else:
                        study_name = "Group{} vs Group{}".format(i + 1, self.pivot + 1)
                    gw = GW_alignment(
                        RDM_i,
                        RDM_pivot,
                        n_init_plan=n_init_plan,
                        epsilon_range=epsilon_range,
                        DATABASE_URL=self.DATABASE_URL,
                        study_name=study_name,
                        init_diag=init_diag,
                    )

                    Pi, min_gwd, min_epsilon, correct_rate = gw.optimize(
                        n_trials=n_trials, n_jobs=n_jobs, parallel=True
                    )
                    Pi_list.append(Pi)

                else:
                    gw = GW_alignment(RDM_i, RDM_pivot, n_init_plan=n_init_plan, epsilons=epsilons, init_diag=init_diag)

                    Pi, min_gwd, min_epsilon, correct_rate = gw.GW_for_epsilons_and_init_plans()
                    Pi_list.append(Pi)

            # visualize
            if self.name_list is not None:
                title = "Pi (before) {} vs {} ($\epsilon = {:.5f}$, Correct rate = {:.2f})".format(
                    self.name_list[i], self.name_list[self.pivot], min_epsilon, correct_rate
                )
            else:
                title = "Pi (before) Group{} vs Group{} ($\epsilon = {:.5f}$, Correct rate = {:.2f})".format(
                    i + 1, self.pivot + 1, min_epsilon, correct_rate
                )
            plt.figure()
            plt.imshow(Pi)
            plt.title(title)
            plt.show()

            # Show Top n accuracy
            for i in [10, 20, 30, 40, 50]:
                correct_rate = calc_correct_rate_top_n(Pi, top_n=i, embedding=self.embedding_list[self.pivot])
                print(f"Top {i} : ", correct_rate)

        return Pi_list

    def main_compute(self, Pi_list_before, max_iter=30, show_result=True):
        """Alignment to the barycenter

        Args:
            Pi_list_before (_type_): Transportation matrices to the pivot
            max_iter (_type_):

        Returns:
            _type_: _description_
        """
        # Procrustes alignment to the pivot
        for i in range(self.n_group):
            Q, self.embedding_list[i] = procrustes(
                self.embedding_list[self.pivot], self.embedding_list[i], Pi_list_before[i]
            )

        # Barycenter alignment
        # Y_init = np.mean(self.embedding_list, axis = 0) # Initial position of the barycenter
        # Y_init = self.embedding_list[self.pivot] # Initial position of the barycenter
        # Initial position of the barycenter
        Y_init = np.zeros_like(self.embedding_list[0])
        for i in range(self.n_group):
            Y_init += np.matmul(Pi_list_before[i].T * Pi_list_before[i].shape[0], self.embedding_list[i])
        Y_init /= self.n_group

        Pi_list_after = [0 for _ in range(self.n_group)]
        loss_list = list()
        iter = 0
        while iter < max_iter:
            # Calculate Barycenter
            Y = fixed_weight_barycenter(self.embedding_list, Y_init, self.bi_list, self.a, self.lambdas)

            loss = 0
            for i in range(self.n_group):
                # Wasserstein alignment to the barycenter
                M = distance.cdist(self.embedding_list[i], Y, metric="euclidean")
                Pi = ot.emd(self.bi_list[i], self.a, M)
                Pi_list_after[i] = Pi

                # Procrustes alignment to the barycenter
                Q, self.embedding_list[i] = procrustes(Y, self.embedding_list[i], Pi)

                # Calculate loss
                loss += np.linalg.norm(Y - np.matmul(Pi.T * Pi.shape[0], self.embedding_list[i]))
            loss /= self.n_group
            loss_list.append(loss)
            Y_init = Y
            iter += 1

            plt.figure(1)
            plt.clf()
            plt.title("Alignment loss")
            plt.plot(loss_list)

        self.embedding_barycenter = Y

        if show_result:
            # Show transportation matrices
            for i in range(self.n_group):
                Pi = np.matmul(Pi_list_before[i], Pi_list_before[self.pivot])  # Assignment of i → barycenter → pivot
                correct_rate = calc_correct_rate_same_dim(Pi)
                if self.name_list is not None:
                    title = "Pi (after) {} vs {} (Correct rate = {:.2f})".format(
                        self.name_list[i], self.name_list[self.pivot], correct_rate
                    )
                else:
                    title = "Pi (after) Group{} vs Group{} (Correct rate = {:.2f})".format(
                        i + 1, self.pivot + 1, correct_rate
                    )
                plt.figure()
                plt.imshow(Pi)
                plt.title(title)
                plt.show()

                # Show Top n accuracy
                for i in [10, 20, 30, 40, 50]:
                    correct_rate = calc_correct_rate_top_n(Pi, top_n=i, embedding=self.embedding_list[self.pivot])
                    print(f"Top {i} : ", correct_rate)

        return Pi_list_after

    def calc_correct_rate_for_pairs(self, Pi_list_before, Pi_list_after):
        pairs = list(itertools.combinations([i for i in range(self.n_group)], 2))
        correct_rate_before = list()
        correct_rate_after = list()
        for pair in pairs:
            Pi_before = np.matmul(
                Pi_list_before[pair[0]], Pi_list_before[pair[1]].T
            )  # Assignment of pair[0] → pivot → pair[1]
            Pi_after = np.matmul(
                Pi_list_after[pair[0]], Pi_list_after[pair[1]].T
            )  # Assignment of pair[0] → barycenter → pair[1]

            correct_rate = calc_correct_rate_same_dim(Pi_before)
            correct_rate_before.append(correct_rate)
            correct_rate = calc_correct_rate_same_dim(Pi_after)
            correct_rate_after.append(correct_rate)

        # Show correct rate
        fig = plt.figure(figsize=(10, 4))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.hist(correct_rate_before, bins=10)
        ax1.set_title("Correct rate (Before)")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.hist(correct_rate_after, bins=10)
        ax2.set_title("Correct rate (After)")

    def plot_embeddings(self, markers_list, color_label):
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 3, 3, projection="3d")
        title = "Aligned embeddings"

        for i in range(self.n_group):
            coords_i = self.embedding_list[i]
            ax.scatter(
                xs=coords_i[:, 0],
                ys=coords_i[:, 1],
                zs=coords_i[:, 2],
                marker=markers_list[i],
                color=color_label,
                s=10,
                alpha=1,
                label="Group{}".format(i + 1),
            )

            ax.set_xlabel("X", fontsize=14)
            ax.set_ylabel("Y", fontsize=14)
            ax.set_zlabel("Z", fontsize=14)
            ax.legend(fontsize=12, loc="best")
            plt.title(title, fontsize=14)
        plt.show()
