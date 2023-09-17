# %%
from typing import Tuple

import numpy as np


#%%
class SimpleHistogramMatching():
    """A class to perform simple histogram matching between two dissimilarity matrices.

    The SimpleHistogramMatching class provides a mechanism to align the histograms of two given dissimilarity
    matrices (source and target). The histogram matching can be performed in two directions, either adjusting the
    source to match the target's histogram or vice versa. The class uses a sorting-based approach to achieve the
    histogram matching between the matrices.

    Args:
        source (np.ndarray): Source's dissimilarity matrix.
        target (np.ndarray): Target's dissimilarity matrix.
    """

    def __init__(self, source:np.ndarray, target:np.ndarray) -> None:
        """Simple Histogram Matching

        Args:
            source (np.ndarray): source's dissimilarity matrix
            target (np.ndarray): target's dissimilarity matrix
        """
        self.source = source
        self.target = target
        pass

    def _sort_for_scaling(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sorts the input matrix for scaling.

        Args:
            X (np.ndarray): Input dissimilarity matrix.

        Returns:
            x (np.ndarray): Flattened input matrix.
            x_sorted (np.ndarray): Sorted input matrix.
            x_inverse_idx (np.ndarray): Inverse index of the sorted input matrix.
        """
        x = X.flatten()
        x_sorted = np.sort(x)
        x_inverse_idx = np.argsort(x).argsort()
        return x, x_sorted, x_inverse_idx

    def _simple_histogram_matching(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Performs simple histogram matching between two matrices.

        Args:
            X (np.ndarray): Source's dissimilarity matrix.
            Y (np.ndarray): Target's dissimilarity matrix.

        Returns:
            Y_t (np.ndarray): Transformed target's dissimilarity matrix.
        """
        # X, Y: dissimilarity matrices
        _, x_sorted, _ = self._sort_for_scaling(X)
        _, _, y_inverse_idx = self._sort_for_scaling(Y)

        y_t = x_sorted[y_inverse_idx]
        Y_t = y_t.reshape(Y.shape)  # transformed matrix
        return Y_t

    def simple_histogram_matching(self, method: str = 'target') -> np.ndarray:
        """Public method to perform simple histogram matching.

        Args:
            method (str, optional): Direction of the histogram matching. Either "source" or "target". Defaults to "target".

        Returns:
            np.ndarray: New embedding after histogram matching.

        Raises:
            ValueError: If method is not "source" or "target".
        """
        if method == 'target':
            new_emb = self._simple_histogram_matching(self.source, self.target)

        elif method == 'source':
            new_emb = self._simple_histogram_matching(self.target, self.source)

        else:
            raise ValueError('method should be source or target only.')

        return new_emb
