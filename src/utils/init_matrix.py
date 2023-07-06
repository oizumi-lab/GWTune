# %%
import numpy as np
import ot

# %%
class InitMatrix():
    def __init__(self, source_size, target_size):
        self.source_size = source_size
        self.target_size = target_size
        self.initialize = ['uniform', 'random', 'permutation', 'diag'] # already implemented.

    def set_user_define_init_mat_list(self, mat):
        if isinstance(mat, list):
            self.user_define_init_mat_list = mat
        
        elif isinstance(mat, np.ndarray):
            self.user_define_init_mat_list = [mat]
            

    def make_initial_T(self, initialize, seed = 42):
        """
        To do : 説明を書く
        """
        np.random.seed(seed) # numpyの乱数を固定する。seed値の変更も可能。
        if initialize == 'random':
            T = self.make_random_init_plan()

        elif initialize == 'permutation':
            ts = np.zeros(self.matrix_size)
            ts[0] = 1 / self.matrix_size
            T = self.initialize_matrix(ts=ts)

        elif initialize == 'any_permutation':
            T = self.initialize_matrix()

        elif initialize == 'beta':
            ts = np.random.beta(2, 5, self.matrix_size)
            ts = ts / (self.matrix_size * np.sum(ts))
            T = self.initialize_matrix(ts=ts)

        elif initialize == 'uniform':
            T = np.outer(ot.unif(self.source_size), ot.unif(self.target_size))

        elif initialize == 'diag':
            T = np.diag(ot.unif(self.source_size))

        else:
            raise ValueError('Not defined initialize matrix.')

        return T

    def randOrderedMatrix(self): #ここも間違っているかもしれないです(2023/7/6)
        """
        各行・各列に重複なしに[0,n]のindexを持つmatrixを作成
        Parameters
            n : int 行列のサイズ
        Returns
            np.ndarray 重複なしのindexを要素に持つmatrix
        """
        matrix = np.zeros((self.source_size, self.target_size))
        rows = np.tile(np.arange(0, self.matrix_size), 2)

        for i in range(self.matrix_size):
            matrix[i, :] = rows[i : i + self.matrix_size]

        r = np.random.choice(self.source_size, self.target_size, replace=False)
        c = np.random.choice(self.source_size, self.target_size, replace=False)
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
        matrix = self.randOrderedMatrix()
        if ts is None:
            ts = np.random.uniform(0, 1, self.matrix_size)
            ts = ts / (self.matrix_size * np.sum(ts))

        T = np.array([ts[idx] for idx in matrix])
        return T

    def make_random_init_plan(self):
        """
        大泉先生が作ったもの。
        """
        # make random initial transportation plan (N x N matrix)
        T = np.random.rand(self.source_size, self.target_size) # create a random matrix of size n x n
        rep = 100 # number of repetitions
        for _ in range(rep):
            # normalize each row so that the sum is 1
            p = T.sum(axis=1, keepdims=True)
            T = T / p
            # normalize each column so that the sum is 1
            q = T.sum(axis=0, keepdims=True)
            T = T / q
        T = T / self.source_size #ここ間違っているかもしれないです・・・ 
        return T

# %%
if __name__ == '__main__':
    test_builder = InitMatrix(2000)
    t = test_builder.make_initial_T('diag')

# %%
