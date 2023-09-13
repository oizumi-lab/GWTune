#%%
import numpy as np
import ot
import pytest

from src.utils.init_matrix import InitMatrix


def test_make_random_init_T():
    p = np.random.rand(2000)
    p /= p.sum()
    q = np.random.rand(1000)
    q /= q.sum()

    init_matrix = InitMatrix(2000, 1000, p, q)

    T = init_matrix.make_initial_T(
        init_mat_plan="random",
        seed=0,
        tol=1e-3,
        max_iter=10000,
    )

    assert T.shape == (2000, 1000)
    assert np.allclose(T.sum(axis=1), p, atol=1e-3)
    assert np.allclose(T.sum(axis=0), q, atol=1e-3)

    # check randomization is done
    T2 = init_matrix.make_initial_T(
        init_mat_plan="random",
        seed=1,
        tol=1e-3,
        max_iter=10000,
    )
    not np.testing.assert_allclose(T, T2, atol=1e-3)


def test_make_uniform_init_T():
    init_matrix = InitMatrix(2000, 1000)

    T = init_matrix.make_initial_T(init_mat_plan="uniform", seed=0)

    assert T.shape == (2000, 1000)
    assert np.allclose(T.sum(axis=1), ot.unif(2000), atol=1e-3)
    assert np.allclose(T.sum(axis=0), ot.unif(1000), atol=1e-3)


def test_make_permutation_init_T():
    init_matrix = InitMatrix(2000, 1000)

    with pytest.raises(AssertionError):
        init_matrix.make_initial_T(init_mat_plan="permutation", seed=0)

    init_matrix = InitMatrix(1000, 1000)
    T = init_matrix.make_initial_T(init_mat_plan="permutation", seed=0)
    assert T.shape == (1000, 1000)
    assert np.allclose(T.sum(axis=1), ot.unif(1000), atol=1e-3)
    assert np.allclose(T.sum(axis=0), ot.unif(1000), atol=1e-3)
    assert (T != 0).sum() == 1000

    T2 = init_matrix.make_initial_T(init_mat_plan="permutation", seed=1)
    not np.testing.assert_allclose(T, T2, atol=1e-3)


#%%
