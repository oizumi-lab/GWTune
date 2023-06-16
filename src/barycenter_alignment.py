#%%
# Third Party Library
# import jax
# import jax.numpy as jnp
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

#%%
class Barycenter_Alignment:
    def __init__(self,
                 embedding_list,
                 pivot_number,
                 OT_to_pivot_list
                 ) -> None:
        """_summary_

        Args:
            embedding_list (List): All embeddings including the one of the pivot
            pivot_number (_type_): The index of the pivot.
            OT_to_pivot_list (_type_): The list of the OT to the pivot. The target of each OT must be the pivot.
        """
        self.pivot = pivot_number
        self.OT_to_pivot_list = OT_to_pivot_list
        self.embedding_pivot = embedding_list[pivot_number]
        self.embedding_others = embedding_list[:pivot_number] + embedding_list[pivot_number + 1:]

    def procurustes(self, embedding_list, embedding_target, OT_to_target_list):
        new_embedding_list = []
        for i, embedding in enumerate(embedding_list):
            OT = OT_to_target_list[i]
            U, S, Vt = np.linalg.svd(np.matmul(embedding.T, np.matmul(OT, embedding_target)))
            Q = np.matmul(U, Vt)
            new_embedding = np.matmul(embedding, Q)
            new_embedding_list.append(new_embedding)

        return new_embedding_list

    def main_compute(self, n_iter, metric, plot_log = False):
        ### procrustes to the pivot
        embedding_list = self.procurustes(self.embedding_others, self.embedding_pivot, self.OT_to_pivot_list) # measures locations
        embedding_list.insert(self.pivot, self.embedding_pivot)

        X_init = np.mean(embedding_list, axis=0) # initial Dirac locations
        weights_list = [ot.unif(len(embedding)) for embedding in embedding_list] # measures weights
        b = ot.unif(len(X_init)) # weights of the barycenter

        loss_list = []
        for i in range(n_iter):
            X = ot.lp.free_support_barycenter(embedding_list, weights_list, X_init, b) # the location of the barycenter

            OT_list = []
            loss = 0
            for j, embedding in enumerate(embedding_list):
                ### OT to the barycenter
                C = distance.cdist(embedding, X, metric=metric)
                OT, log = ot.emd(weights_list[j], b, C, log=True)
                OT_list.append(OT)
                loss += log['cost']

            loss /= len(embedding_list)
            loss_list.append(loss)

            ### procrustes to the barycenter
            # update the embedding list
            embedding_list = self.procurustes(embedding_list, embedding_target=X, OT_to_target_list=OT_list)

            # update the initial Dirac locations of the barycenter
            X_init = X

        if plot_log:
            plt.figure()
            plt.plot(loss_list)
            plt.xlabel("iteration")
            plt.ylabel("Mean Wasserstein distance")

        return X, OT_list, embedding_list

#%%
if __name__ == "__main__":
    embedding_list = [np.random.randn(3, 2),
                      np.random.randn(3, 2),
                      np.random.randn(3, 2)]
    pivot = 0

    OT_to_pivot_list = [np.random.randn(3, 3), np.random.randn(3, 3)]

    barycenter = Barycenter_Alignment(embedding_list=embedding_list,
                                      pivot_number=pivot,
                                      OT_to_pivot_list=OT_to_pivot_list)

    X, OT_list, embedding_list = barycenter.main_compute(n_iter=10, metric="euclidean", plot_log=True)

# %%
