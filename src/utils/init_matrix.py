# %%
from typing import List, Union

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


    def make_initial_T(self, init_mat_plan: str, seed: int = 42) -> np.ndarray:
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
            T = self.make_random_init_plan()

        elif init_mat_plan == 'uniform':
            T = np.outer(ot.unif(self.source_size), ot.unif(self.target_size))

        elif init_mat_plan == 'diag':
            T = np.diag(ot.unif(self.source_size))

        else:
            raise ValueError('Not defined init_mat_plan.')

        return T

    def make_random_init_plan(self) -> np.ndarray:
        """Generates a random initial matrix for Gromov-Wasserstein alignment.

        The method creates a random matrix and normalizes it to ensure that the sum of each row and column forms a uniform distribution.
        This normalization process is repeated multiple times to obtain a doubly stochastic matrix.

        Returns:
            np.ndarray: The initialized transportation plan matrix.
        """
        T = np.random.rand(self.source_size, self.target_size) # create a random matrix
        rep = 100 # number of repetitions

        for _ in range(rep):
            # normalize each row so that the sum is 1
            p = T.sum(axis=1, keepdims=True)
            T = T / p

            # normalize each column so that the sum is 1
            q = T.sum(axis=0, keepdims=True)
            T = T / q

        T = T / T.sum()

        return T

# %%
if __name__ == '__main__':
    test_builder = InitMatrix(2000)
    t = test_builder.make_initial_T('diag')

# %%
