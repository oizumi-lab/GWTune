# %%
from typing import List, Union, Optional

import numpy as np
import ot


# %%
class InitMatrix():
    """A class for creating initial matrices for Gromov-Wasserstein alignment.

    This class provides methods to generate initial matrices based on various conditions
    such as "random", "uniform", "diag", or user-defined matrices.

    Attributes:
        source_size (int): The size of the source distribution.
        target_size (int): The size of the target distribution.
        user_define_init_mat_list (List[np.ndarray], optional): User-defined initial matrix or matrices.
    """

    def __init__(self, source_size: int, target_size: int) -> None:
        """Initializes the InitMatrix class with given source and target sizes.

        Args:
            source_size (int): The size of the source distribution.
            target_size (int): The size of the target distribution.
        """

        self.source_size = source_size
        self.target_size = target_size

    def set_user_define_init_mat_list(self, mat: Union[np.ndarray, List[np.ndarray]]) -> None:
        """Sets the user-defined initial matrix or matrices.

        This method allows users to specify their own initial matrices for the Gromov-Wasserstein alignment.
        Users can either provide a single matrix or a list of matrices.

        Args:
            mat (Union[np.ndarray, List[np.ndarray]]): A single matrix or a list of matrices.

        Raises:
            ValueError: If the provided input is neither a numpy array nor a list of numpy arrays.
        """

        if isinstance(mat, list):
            self.user_define_init_mat_list = mat

        elif isinstance(mat, np.ndarray):
            self.user_define_init_mat_list = [mat]

        else:
            raise ValueError('The provided input should either be a numpy array or a list of numpy arrays.')

    def make_initial_T(self, init_mat_plan: str, seed: int = 42, **kwargs) -> np.ndarray:
        """Generates the initial matrix for Gromov-Wasserstein alignment.

        Users can specify the initialization method for the matrix, choosing from options such as
        "random", "uniform" or "diag".

        Args:
            init_mat_plan (str): Method to initialize the matrix. Options are "random", "uniform" or "diag".
            seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 42.

        Returns:
            np.ndarray: The initialized transportation plan matrix.

        Raises:
            ValueError: If the provided initialization method is not implemented.
        """

        np.random.seed(seed) # fix the seed of numpy.random, seed can be changed by user.

        if init_mat_plan == 'random':
            T = self.make_random_init_plan(**kwargs)

        elif init_mat_plan == 'uniform':
            T = np.outer(ot.unif(self.source_size), ot.unif(self.target_size))

        elif init_mat_plan == 'diag':
            T = np.diag(ot.unif(self.source_size))

        else:
            raise ValueError('Not defined init_mat_plan.')

        return T

    def make_random_init_plan(
        self,
        p: Optional[np.ndarray] = None,
        q: Optional[np.ndarray] = None,
        tol: float = 1e-3,
        max_iter: int = 1000,
    )-> np.ndarray:
        """Generates a random initial matrix for Gromov-Wasserstein alignment.

        The method creates a random matrix and normalizes it to ensure that the sum of each row and column forms an input distribution.
        If no input distribution is provided, uniform distribution is used.
        This normalization process is repeated multiple times to obtain a doubly stochastic matrix.

        Args:
            p (np.ndarray): source distribution. Defaults to None. If None, uniform distribution is used.
            q (np.ndarray): target distribution. Defaults to None. If None, uniform distribution is used.
            tol (float, optional): stop condition. Defaults to 1e-3.
            max_iter (int, optional): maximum number of iterations. Defaults to 1000.

        Returns:
            T (np.ndarray): initial matrix
        """

        # initialize p and q
        if p is None:
            p = ot.unif(self.source_size).reshape(-1, 1)
        if q is None:
            q = ot.unif(self.target_size).reshape(1, -1)

        # check the dimension of p and q
        if p.ndim < 2:
            p = p.reshape(-1, 1)
        if q.ndim < 2:
            q = q.reshape(1, -1)

        # create a random matrix
        T = np.random.rand(self.source_size, self.target_size)

        for _ in range(max_iter):
            # normalize each row so that the sum corresponds to the source distribution
            T = (T * p) / T.sum(axis=1, keepdims=True)

            # normalize each column so that the sum corresponds to the target distribution
            T = (T * q) / T.sum(axis=0, keepdims=True)

            # check the stop condition
            if np.linalg.norm(T.sum(axis=1, keepdims=True) - p) < tol:
                if np.linalg.norm(T.sum(axis=0, keepdims=True) - q) < tol:
                    break

        return T


# %%
if __name__ == '__main__':
    test_builder = InitMatrix(2000)
    t = test_builder.make_initial_T('diag')

# %%
