# %%
import numpy as np
import ot

# %%
class InitMatrix():
    def __init__(self, matrix_size = None):
        self.matrix_size = matrix_size
        self.initialize = ['uniform', 'random', 'permutation', 'diag'] # 実装済みの方法の名前を入れる。

    def implemented_init_plans(self, init_plans_list): # リストを入力して、実行可能な方法のみをリストにして返す。
        """
        ここから、初期値の条件を1個または複数個選択することができる。
        選択はself.initializeの中にあるものの中から。
        選択したい条件が1つであっても、リストで入力をすること。

        Args:
            init_plans_list (list) : 初期値の条件を1個または複数個入れたリスト。

        Raises:
            ValueError: 選択したい条件が1つであっても、リストで入力をすること。

        Returns:
            list : 選択希望の条件のリスト。
        """

        if type(init_plans_list) != list:
            raise ValueError('variable named "init_plans_list" is not list!')

        else:
            return [v for v in self.initialize if v in init_plans_list]

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
            T = np.outer(ot.unif(self.matrix_size), ot.unif(self.matrix_size))

        elif initialize == 'diag':
            T = np.diag(ot.unif(self.matrix_size))

        else:
            raise ValueError('Not defined initialize matrix.')

        return T

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
