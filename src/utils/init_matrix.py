# %%
import numpy as np
import ot

# %%
class InitMatrix():
    def __init__(self, source_size:int, target_size:int):
        self.source_size = source_size
        self.target_size = target_size

    def set_user_define_init_mat_list(self, mat):
        if isinstance(mat, list):
            self.user_define_init_mat_list = mat
        
        elif isinstance(mat, np.ndarray):
            self.user_define_init_mat_list = [mat]
           

    def make_initial_T(self, initialize:str, seed:int = 42):
        """
        define the initial matrix for GW alignment.
        Users can choose the condition of it from "random", "uniform", "diag" or "user_define".

        Args:
            initialize (str): _description_
            seed (int, optional): _description_. Defaults to 42.

        Returns:
            initial OT plans (np.ndarray)
        """

        np.random.seed(seed) # fix the seed of numpy.random, seed can be changed by user.
        
        if initialize == 'random':
            T = self.make_random_init_plan()

        elif initialize == 'uniform':
            T = np.outer(ot.unif(self.source_size), ot.unif(self.target_size))

        elif initialize == 'diag':
            T = np.diag(ot.unif(self.source_size))

        else:
            raise ValueError('Not defined initialize matrix.')

        return T

    def make_random_init_plan(self):
        """
        make random initial matrix.
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
        
        T = T / T.sum() #2023.7.14 fixed by Ohizumi
        return T

# %%
if __name__ == '__main__':
    test_builder = InitMatrix(2000)
    t = test_builder.make_initial_T('diag')

# %%
