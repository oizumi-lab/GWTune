# %%
# Standard Library
import os
import numpy as np
import scipy as sp

#%%
class SimpleHistogramMatching():
    def __init__(self, source:np.ndarray, target:np.ndarray) -> None:
        """
        Simple Histogram Matching

        Args:
            source (np.ndarray): source's dis-similarity matrix
            target (np.ndarray): target's dis-similarity matrix
        """
        self.source = source
        self.target = target
        pass

    def _sort_for_scaling(self, X):
        x = X.flatten()
        x_sorted = np.sort(x)
        x_inverse_idx = np.argsort(x).argsort()
        return x, x_sorted, x_inverse_idx

    def _simple_histogram_matching(self, X, Y):
        # X, Y: dissimilarity matrices
        x, x_sorted, x_inverse_idx = self._sort_for_scaling(X)
        y, y_sorted, y_inverse_idx = self._sort_for_scaling(Y)

        y_t = x_sorted[y_inverse_idx]
        Y_t = y_t.reshape(Y.shape)  # transformed matrix
        return Y_t

    def simple_histogram_matching(self, method = 'target'):
        if method == 'target':
            new_emb = self._simple_histogram_matching(self.source, self.target)

        elif method == 'source':
            new_emb = self._simple_histogram_matching(self.target, self.source)

        else:
            raise ValueError('method should be source or target only.')

        return new_emb