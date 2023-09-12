from typing import Any, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, MDS


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
    new_idx_list = np.cumsum([0] + [len(embedding) for embedding in embedding_list])

    # use transformer
    new_embedding_list = []
    for start_idx, end_idx in zip(new_idx_list[:-1], new_idx_list[1:]):
        new_embedding_list.append(new_X[start_idx:end_idx])

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
