import numpy as np

from src.utils.init_matrix import InitMatrix


def test_make_random_init_T():
    np.random.seed(0)
    p = np.random.rand(2000)
    p /= p.sum()
    q = np.random.rand(1000)
    q /= q.sum()

    init_matrix = InitMatrix(2000, 1000, p, q)

    T = init_matrix.make_initial_T(
        init_mat_plan="random",
        seed = 0,
        tol=1e-3,
        max_iter=10000,
    )

    assert T.shape == (2000, 1000)
    assert np.allclose(T.sum(axis=1), p, atol=1e-3)
    assert np.allclose(T.sum(axis=0), q, atol=1e-3)
