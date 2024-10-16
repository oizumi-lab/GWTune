#%%
import os
import random
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from typing import Any, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, MDS


def fix_random_seed(seed: int = 42) -> None:
    """Set the seed for various random number generators to ensure reproducibility.

    Args:
        seed (int, optional): The seed value to be set. Defaults to 42.
    """

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


def get_category_data(
    category_mat: pd.DataFrame,
    category_name_list: Optional[List[str]] = None,
    show_numbers: bool = False
) -> List[str]:
    """Extract specific category data from the provided DataFrame.

    Args:
        category_mat (pd.DataFrame): DataFrame containing category data. Columns represent categories, rows represent objects,
                                     and entries are binary (1 if the object belongs to that category, 0 otherwise).
        category_name_list (Optional[List[str]]): List of category names to extract data for. If None, all categories
                                                  in the DataFrame will be considered. Defaults to None.
        show_numbers (bool): If True, displays the number of occurrences of each category. Defaults to False.

    Returns:
        object_labels (List[str]): List of object labels for each category.
        category_idx_list (List[np.ndarray]): List of indices for each category.
        category_num_list (List[int]): List containing the number of occurrences of each category.
        new_category_name_list (List[str]): List of category names considered.
    """

    if category_name_list is None:
        new_category_name_list = category_mat.columns.tolist()
    else:
        new_category_name_list = category_name_list

    category_idx_list = []
    category_num_list = []
    object_labels = []

    for category in new_category_name_list:
        category_idx = category_mat[category].values == 1

        category_idx_list.append(np.where(category_idx)[0])
        category_num_list.append(category_idx.sum())
        object_labels.extend(category_mat.index[category_idx].tolist())

    if show_numbers:
        num_each_category = pd.DataFrame(category_num_list, index = new_category_name_list, columns = ["Number"])
        print(num_each_category)

    return object_labels, category_idx_list, category_num_list, new_category_name_list


def sort_matrix_with_categories(matrix: Any, category_idx_list: List[int]) -> np.ndarray:
    """Reorganize a matrix based on category indices, producing blocks corresponding to the categories.

    Args:
        matrix (Any): The input matrix to be reorganized.
        category_idx_list (List[int]): A list of index, each representing the index for a category.

    Returns:
        new_mat (np.ndarray): A new matrix where the rows and columns are sorted according to the specified categories.
    """

    new_mat_blocks = []
    for i in category_idx_list:
        row_blocks = []
        for j in category_idx_list:
            block = matrix[i][:, j]
            row_blocks.append(block)
        new_mat_blocks.append(np.concatenate(row_blocks, axis=1))

    new_mat = np.concatenate(new_mat_blocks, axis=0)
    return new_mat


def obtain_embedding(
    embedding_list: List[np.ndarray],
    dim: int,
    emb_name: Optional[str] = "PCA",
    emb_transformer: Optional[Any] = None,
    **kwargs
) -> Tuple[List[np.ndarray], Any]:
    
    # preprocessing
    X = np.vstack(embedding_list)

    # load transformer
    if (emb_transformer is None) and (emb_name is not None):
        emb_transformer = load_transformer(emb_name, dim, **kwargs)

    assert emb_transformer is not None, "You should provide both emb_name and emb_transformer"

    # fit_transform transformer
    new_X = emb_transformer.fit_transform(X)

    # use transformer
    new_embedding_list = [new_X[i * embedding_list[i].shape[0]: (i+1) * embedding_list[i].shape[0]] for i in range(len(embedding_list))]

    return new_embedding_list, emb_transformer


def load_transformer(
    emb_name: str,
    dim: int,
    **kwargs
) -> Any:

    if emb_name == "PCA":
        emb_transformer = PCA(n_components=dim, **kwargs)

    elif emb_name == "TSNE":
        emb_transformer = TSNE(n_components=dim, **kwargs)

    elif emb_name == "Isomap":
        emb_transformer = Isomap(n_components=dim, **kwargs)

    elif emb_name == "MDS":
        emb_transformer = MDS(n_components=dim, **kwargs)

    else:
        raise ValueError(f"Unknown embedding algorithm: {emb_name}")

    return emb_transformer

# %%
if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    category_mat = pd.read_csv("../../data/category_mat_manual_preprocessed.csv", sep = ",", index_col = 0)
    object_labels, category_idx_list, category_num_list, new_category_name_list = get_category_data(category_mat = category_mat)

    a = np.random.randn(1854,1854)
    b = sort_matrix_with_categories(a, category_idx_list)

    plt.figure()
    plt.subplot(121)

    plt.title("source")
    plt.imshow(a)

    plt.subplot(122)
    plt.title("target")
    plt.imshow(b)
    plt.tight_layout()
    plt.show()
