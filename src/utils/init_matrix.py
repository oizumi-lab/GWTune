# %%
import numpy as np
import ot

# %%
class InitMatrix():
    def __init__(self, matrix_size):
        self.matrix_size = matrix_size
        pass

    def make_initial_T(self, initialize, seed = 42):
        """
        To do : 説明を書く
        """
        np.random.seed(seed) # numpyの乱数を固定する。seed値の変更も可能。
        if initialize == 'random':
            T = self.make_random_initplan()
            return T

        elif initialize == 'permutation':
            ts = np.zeros(self.matrix_size)
            ts[0] = 1 / self.matrix_size
            T = self.initialize_matrix(self.matrix_size, ts=ts)
            return T

        elif initialize == 'any_permutation':
            T = self.initialize_matrix(self.matrix_size)
            return T

        elif initialize == 'beta':
            ts = np.random.beta(2, 5, self.matrix_size)
            ts = ts / (self.matrix_size * np.sum(ts))
            T = self.initialize_matrix(self.matrix_size, ts=ts)
            return T

        elif initialize == 'uniform':
            T = np.outer(ot.unif(self.matrix_size), ot.unif(self.matrix_size))
            return T

        elif initialize == 'diag':
            T = np.diag(ot.unif(self.matrix_size))
            return T

        else:
            raise ValueError('Not defined initialize matrix.')

    def randOrderedMatrix(self):
        """
        各行・各列に重複なしに[0,n]のindexを持つmatrixを作成
        Parameters
            n : int 行列のサイズ
        Returns
            np.ndarray 重複なしのindexを要素に持つmatrix
        """
        matrix = np.zeros((self.matrix_size, self.matrix_size))
        rows = np.tile(np.arange(0, self.matrix_size), 2)

        for i in range(self.matrix_size):
            matrix[i, :] = rows[i : i + self.matrix_size]

        r = np.random.choice(self.matrix_size, self.matrix_size, replace=False)
        c = np.random.choice(self.matrix_size, self.matrix_size, replace=False)
        matrix = matrix[r, :]
        matrix = matrix[:, c]
        return matrix.astype(int)

    def initialize_matrix(self, ts=None):
        """
        gw alignmentのための行列初期化
        Parameters
            n : int 行列のサイズ
        Returns
            np.ndarray 初期値
        """
        matrix = self.randOrderedMatrix(self.matrix_size)
        if ts is None:
            ts = np.random.uniform(0, 1, self.matrix_size)
            ts = ts / (self.matrix_size * np.sum(ts))

        T = np.array([ts[idx] for idx in matrix])
        return T

    def make_random_initplan(self):
        """
        大泉先生が作ったもの。
        """
        # make random initial transportation plan (N x N matrix)
        T = np.random.rand(self.matrix_size, self.matrix_size) # create a random matrix of size n x n
        rep = 100 # number of repetitions
        for _ in range(rep):
            # normalize each row so that the sum is 1
            p = T.sum(axis=1, keepdims=True)
            T = T / p
            # normalize each column so that the sum is 1
            q = T.sum(axis=0, keepdims=True)
            T = T / q
        T = T / self.matrix_size
        return T

# %% 
if __name__ == '__main__':
    test_builder = InitMatrix(2000)
    t = test_builder.make_initial_T('diag')
    
# %% 