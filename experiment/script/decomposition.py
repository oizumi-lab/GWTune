#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
#%%
# load data
category_mat = pd.read_csv("../../data/THINGS/category_mat_manual_preprocessed.csv", sep=",", index_col=0)
embedding = np.load('../../data/THINGS/THINGS_embedding_Group1.npy')[0]

from src.utils.utils_functions import get_category_data, sort_matrix_with_categories  # get_category_data and sort_matrix_with_categories are functions specialied for this dataset
object_labels, category_idx_list, num_category_list, category_name_list = get_category_data(category_mat)
#%%

RDM = cdist(embedding, embedding, 'euclidean')
RDM = sort_matrix_with_categories(RDM, category_idx_list)
# %%
# plot RDM
plt.imshow(RDM)
# %%
# eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(RDM)

# %%
#reconstructed_RDM = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
#%%
for i in range(5):
    RDM_i = eigenvectors[:, i].reshape(1, -1).T @ eigenvectors[:, i].reshape(1, -1)
    plt.figure()
    plt.imshow(RDM_i)
    plt.show()

# %%
# plot eigenvalues
plt.plot(eigenvalues[:10])
# %%
i = 5
reconstructed_RDM = eigenvectors[:, :i] @ np.diag(eigenvalues[:i]) @ eigenvectors[:, :i].T
plt.imshow(reconstructed_RDM)
# %%
