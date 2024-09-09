#%%
import ot
import numpy as np


from ot.utils import dist, UndefinedParameter, list_to_array
from ot.optim import cg, line_search_armijo, solve_1d_linesearch_quad, generic_conditional_gradient
from ot.utils import check_random_state, unif
from ot.backend import get_backend, NumpyBackend
from ot.lp import emd
from ot.bregman import sinkhorn

from ot.gromov._utils import init_matrix, gwloss, gwggrad
#from ot.gromov._utils import update_square_loss, update_kl_loss, update_feature_matrix

import warnings

#%%

def faster_gromov_wasserstein(C1, C2, p=None, q=None, eps=None, loss_fun='square_loss', symmetric=None, log=False, armijo=False, G0=None,
                       max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, **kwargs):
    r"""
    Returns the Gromov-Wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`.

    The function solves the following optimization problem using Conditional Gradient:

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{\gamma} \mathbf{1} &= \mathbf{p}

             \mathbf{\gamma}^T \mathbf{1} &= \mathbf{q}

             \mathbf{\gamma} &\geq 0

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.
    .. note:: All computations in the conjugate gradient solver are done with
        numpy to limit memory overhead.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : str, optional
        loss function used for the solver either 'square_loss' or 'kl_loss'
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research. Else closed form is used.
        If there are convergence issues use False.
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 must satisfy marginal constraints and will be used as initial transport of the solver.
    max_iter : int, optional
        Max number of iterations
    tol_rel : float, optional
        Stop threshold on relative error (>0)
    tol_abs : float, optional
        Stop threshold on absolute error (>0)
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver

    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Coupling between the two spaces that minimizes:

            :math:`\sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}`
    log : dict
        Convergence information and loss.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [13] Mémoli, Facundo. Gromov–Wasserstein distances and the
        metric approach to object matching. Foundations of computational
        mathematics 11.4 (2011): 417-487.

    .. [47] Chowdhury, S., & Mémoli, F. (2019). The gromov–wasserstein
        distance between networks and stable network invariants.
        Information and Inference: A Journal of the IMA, 8(4), 757-787.
    """
    arr = [C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(C2.shape[0], type_as=C2)
    if G0 is not None:
        G0_ = G0
        arr.append(G0)

    nx = get_backend(*arr)
    p0, q0, C10, C20 = p, q, C1, C2

    p = nx.to_numpy(p0)
    q = nx.to_numpy(q0)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)
    if symmetric is None:
        symmetric = np.allclose(C1, C1.T, atol=1e-10) and np.allclose(C2, C2.T, atol=1e-10)

    if G0 is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = nx.to_numpy(G0_)
        # Check marginals of G0
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-08)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-08)
    # cg for GW is implemented using numpy on CPU
    np_ = NumpyBackend()

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun, np_)

    def f(G):
        return gwloss(constC, hC1, hC2, G, np_)
    
    #def f(G):
    #    return gwloss(constC, hC1, hC2, G, np_) + eps * np_.sum(G * np_.log(G + 1e-15))

    if symmetric:
        def df(G):
            return gwggrad(constC, hC1, hC2, G, np_)
    else:
        constCt, hC1t, hC2t = init_matrix(C1.T, C2.T, p, q, loss_fun, np_)

        def df(G):
            return 0.5 * (gwggrad(constC, hC1, hC2, G, np_) + gwggrad(constCt, hC1t, hC2t, G, np_))
    
    #if symmetric:
    #    def df(G):
    #        return gwggrad(constC, hC1, hC2, G, np_) + eps * (np_.log(G + 1e-15) + 1)
    #else:
    #    constCt, hC1t, hC2t = init_matrix(C1.T, C2.T, p, q, loss_fun, np_)
#
    #    def df(G):
    #        return 0.5 * (gwggrad(constC, hC1, hC2, G, np_) + gwggrad(constCt, hC1t, hC2t, G, np_)) + eps * (np_.log(G + 1e-15) + 1)
    
    if loss_fun == 'kl_loss':
        armijo = True  # there is no closed form line-search with KL

    if armijo:
        def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
            return line_search_armijo(cost, G, deltaG, Mi, cost_G, nx=np_, **kwargs)
    else:
        def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
            return solve_gromov_linesearch(G, deltaG, cost_G, C1, C2, M=0., reg=1., nx=np_, **kwargs)
    if log:
        res, log = entropic_cg(p, q, 0., 1., eps, f, df, G0, line_search, log=True, numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs)
        log['gw_dist'] = nx.from_numpy(log['loss'][-1], type_as=C10)
        log['u'] = nx.from_numpy(log['u'], type_as=C10)
        log['v'] = nx.from_numpy(log['v'], type_as=C10)
        return nx.from_numpy(res, type_as=C10), log
    else:
        return nx.from_numpy(entropic_cg(p, q, 0., 1., eps, f, df, G0, line_search, log=False, numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs), type_as=C10)


def solve_gromov_linesearch(G, deltaG, cost_G, C1, C2, M, reg,
                            alpha_min=None, alpha_max=None, nx=None, **kwargs):
    """
    Solve the linesearch in the FW iterations

    Parameters
    ----------

    G : array-like, shape(ns,nt)
        The transport map at a given iteration of the FW
    deltaG : array-like (ns,nt)
        Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration
    cost_G : float
        Value of the cost at `G`
    C1 : array-like (ns,ns), optional
        Structure matrix in the source domain.
    C2 : array-like (nt,nt), optional
        Structure matrix in the target domain.
    M : array-like (ns,nt)
        Cost matrix between the features.
    reg : float
        Regularization parameter.
    alpha_min : float, optional
        Minimum value for alpha
    alpha_max : float, optional
        Maximum value for alpha
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    alpha : float
        The optimal step size of the FW
    fc : int
        nb of function call. Useless here
    cost_G : float
        The value of the cost for the next iteration


    .. _references-solve-linesearch:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    if nx is None:
        G, deltaG, C1, C2, M = list_to_array(G, deltaG, C1, C2, M)

        if isinstance(M, int) or isinstance(M, float):
            nx = get_backend(G, deltaG, C1, C2)
        else:
            nx = get_backend(G, deltaG, C1, C2, M)

    dot = nx.dot(nx.dot(C1, deltaG), C2.T)
    a = -2 * reg * nx.sum(dot * deltaG)
    b = nx.sum(M * deltaG) - 2 * reg * (nx.sum(dot * G) + nx.sum(nx.dot(nx.dot(C1, G), C2.T) * deltaG))

    alpha = solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    # the new cost is deduced from the line search quadratic function
    cost_G = cost_G + a * (alpha ** 2) + b * alpha

    return alpha, 1, cost_G

def entropic_cg(a, b, M, reg, eps, f, df, G0=None, line_search=line_search_armijo,
       numItermax=200, numItermaxEmd=100000, stopThr=1e-9, stopThr2=1e-9,
       verbose=False, log=False, **kwargs): ## add entropy parameter
    r"""
    Solve the general regularized OT problem with conditional gradient

        The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot f(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`f` is the regularization term (and `df` is its gradient)
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is conditional gradient as discussed in :ref:`[1] <references-cg>`


    Parameters
    ----------
    a : array-like, shape (ns,)
        samples weights in the source domain
    b : array-like, shape (nt,)
        samples in the target domain
    M : array-like, shape (ns, nt)
        loss matrix
    reg : float
        Regularization term >0
    G0 :  array-like, shape (ns,nt), optional
        initial guess (default is indep joint density)
    line_search: function,
        Function to find the optimal step.
        Default is line_search_armijo.
    numItermax : int, optional
        Max number of iterations
    numItermaxEmd : int, optional
        Max number of iterations for emd
    stopThr : float, optional
        Stop threshold on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshold on the absolute variation (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    **kwargs : dict
             Parameters for linesearch

    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-cg:
    References
    ----------

    .. [1] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.

    See Also
    --------
    ot.lp.emd : Unregularized optimal transport
    ot.bregman.sinkhorn : Entropic regularized optimal transport

    """

    def lp_solver(a, b, M, **kwargs):
        return emd(a, b, M, numItermaxEmd, log=True)
    
    def sinkhorn(a, b, M, **kwargs):
        return ot.bregman.sinkhorn(a, b, M, eps, numItermax=numItermaxEmd, log=True)

    return my_generic_conditional_gradient(a, b, M, f, df, reg, None, sinkhorn, line_search, G0=G0,
                                        numItermax=numItermax, stopThr=stopThr,
                                        stopThr2=stopThr2, verbose=verbose, log=log, **kwargs)


def my_generic_conditional_gradient(a, b, M, f, df, reg1, reg2, lp_solver, line_search, G0=None,
                                 numItermax=200, stopThr=1e-9,
                                 stopThr2=1e-9, verbose=False, log=False, **kwargs):
    r"""
    Solve the general regularized OT problem or its semi-relaxed version with
    conditional gradient or generalized conditional gradient depending on the
    provided linear program solver.

        The function solves the following optimization problem if set as a conditional gradient:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_1} \cdot f(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b} (optional constraint)

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`f` is the regularization term (and `df` is its gradient)
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is conditional gradient as discussed in :ref:`[1] <references-cg>`

        The function solves the following optimization problem if set a generalized conditional gradient:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_1}\cdot f(\gamma) + \mathrm{reg_2}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0

    where :

    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`

    The algorithm used for solving the problem is the generalized conditional gradient as discussed in :ref:`[5, 7] <references-gcg>`

    Parameters
    ----------
    a : array-like, shape (ns,)
        samples weights in the source domain
    b : array-like, shape (nt,)
        samples weights in the target domain
    M : array-like, shape (ns, nt)
        loss matrix
    f : function
        Regularization function taking a transportation matrix as argument
    df: function
        Gradient of the regularization function taking a transportation matrix as argument
    reg1 : float
        Regularization term >0
    reg2 : float,
        Entropic Regularization term >0. Ignored if set to None.
    lp_solver: function,
        linear program solver for direction finding of the (generalized) conditional gradient.
        If set to emd will solve the general regularized OT problem using cg.
        If set to lp_semi_relaxed_OT will solve the general regularized semi-relaxed OT problem using cg.
        If set to sinkhorn will solve the general regularized OT problem using generalized cg.
    line_search: function,
        Function to find the optimal step. Currently used instances are:
        line_search_armijo (generic solver). solve_gromov_linesearch for (F)GW problem.
        solve_semirelaxed_gromov_linesearch for sr(F)GW problem. gcg_linesearch for the Generalized cg.
    G0 :  array-like, shape (ns,nt), optional
        initial guess (default is indep joint density)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshold on the absolute variation (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    **kwargs : dict
             Parameters for linesearch

    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-cg:
    .. _references_gcg:
    References
    ----------

    .. [1] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.

    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, "Optimal Transport for Domain Adaptation," in IEEE Transactions on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1

    .. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015). Generalized conditional gradient: analysis of convergence and applications. arXiv preprint arXiv:1510.06567.

    See Also
    --------
    ot.lp.emd : Unregularized optimal transport
    ot.bregman.sinkhorn : Entropic regularized optimal transport
    """
    a, b, M, G0 = list_to_array(a, b, M, G0)
    if isinstance(M, int) or isinstance(M, float):
        nx = get_backend(a, b)
    else:
        nx = get_backend(a, b, M)

    loop = 1

    if log:
        log = {'loss': []}

    if G0 is None:
        G = nx.outer(a, b)
    else:
        # to not change G0 in place.
        G = nx.copy(G0)

    if reg2 is None:
        def cost(G):
            return nx.sum(M * G) + reg1 * f(G)
    else:
        def cost(G):
            return nx.sum(M * G) + reg1 * f(G) + reg2 * nx.sum(G * nx.log(G))
    cost_G = cost(G)
    if log:
        log['loss'].append(cost_G)

    it = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
        print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, cost_G, 0, 0))

    num_loop = 0
    while loop:

        it += 1
        old_cost_G = cost_G
        # problem linearization
        Mi = M + reg1 * df(G)

        if not (reg2 is None):
            Mi = Mi + reg2 * (1 + nx.log(G))
        # set M positive
        Mi = Mi + nx.min(Mi)

        # solve linear program
        Gc, innerlog_ = lp_solver(a, b, Mi, **kwargs)

        # line search
        deltaG = Gc - G

        alpha, fc, cost_G = line_search(cost, G, deltaG, Mi, cost_G, **kwargs)

        G = G + alpha * deltaG

        # test convergence
        if it >= numItermax:
            loop = 0

        abs_delta_cost_G = abs(cost_G - old_cost_G)
        relative_delta_cost_G = abs_delta_cost_G / abs(cost_G)
        if relative_delta_cost_G < stopThr or abs_delta_cost_G < stopThr2:
            loop = 0

        if log:
            log['loss'].append(cost_G)

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
                    'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
            print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, cost_G, relative_delta_cost_G, abs_delta_cost_G))

        num_loop += 1

    print("Number of loops: ", num_loop)
    
    if log:
        log.update(innerlog_)
        return G, log
    else:
        return G

def entropic_gromov_wasserstein(
        C1, C2, p=None, q=None, loss_fun='square_loss', epsilon=0.1, symmetric=None, G0=None, max_iter=1000,
        tol=1e-9, solver='PGD', warmstart=False, verbose=False, log=False, **kwargs):
    r"""
    Returns the Gromov-Wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`
    estimated using Sinkhorn projections.

    If `solver="PGD"`, the function solves the following entropic-regularized
    Gromov-Wasserstein optimization problem using Projected Gradient Descent [12]:

    .. math::
        \mathbf{T}^* \in \mathop{\arg\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} - \epsilon H(\mathbf{T})

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

    Else if `solver="PPA"`, the function solves the following Gromov-Wasserstein
    optimization problem using Proximal Point Algorithm [51]:

    .. math::
        \mathbf{T}^* \in \mathop{\arg\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0
    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices
    - `H`: entropy

    .. note:: If the inner solver `ot.sinkhorn` did not convergence, the
        optimal coupling :math:`\mathbf{T}` returned by this function does not
        necessarily satisfy the marginal constraints
        :math:`\mathbf{T}\mathbf{1}=\mathbf{p}` and
        :math:`\mathbf{T}^T\mathbf{1}=\mathbf{q}`. So the returned
        Gromov-Wasserstein loss does not necessarily satisfy distance
        properties and may be negative.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    loss_fun :  string, optional
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float, optional
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 will be used as initial transport of the solver. G0 is not
        required to satisfy marginal constraints but we strongly recommand it
        to correcly estimate the GW distance.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    solver: string, optional
        Solver to use either 'PGD' for Projected Gradient Descent or 'PPA'
        for Proximal Point Algorithm.
        Default value is 'PGD'.
    warmstart: bool, optional
        Either to perform warmstart of dual potentials in the successive
        Sinkhorn projections.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.
    **kwargs: dict
        parameters can be directly passed to the ot.sinkhorn solver.
        Such as `numItermax` and `stopThr` to control its estimation precision,
        e.g [51] suggests to use `numItermax=1`.
    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two spaces

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [47] Chowdhury, S., & Mémoli, F. (2019). The gromov–wasserstein
        distance between networks and stable network invariants.
        Information and Inference: A Journal of the IMA, 8(4), 757-787.

    .. [51] Xu, H., Luo, D., Zha, H., & Duke, L. C. (2019). Gromov-wasserstein
        learning for graph matching and node embedding. In International
        Conference on Machine Learning (ICML), 2019.
    """
    if solver not in ['PGD', 'PPA']:
        raise ValueError("Unknown solver '%s'. Pick one in ['PGD', 'PPA']." % solver)

    C1, C2 = list_to_array(C1, C2)
    arr = [C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(C2.shape[0], type_as=C2)

    if G0 is not None:
        arr.append(G0)

    nx = get_backend(*arr)

    if G0 is None:
        G0 = nx.outer(p, q)

    T = G0
    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun, nx)

    if symmetric is None:
        symmetric = nx.allclose(C1, C1.T, atol=1e-10) and nx.allclose(C2, C2.T, atol=1e-10)
    if not symmetric:
        constCt, hC1t, hC2t = init_matrix(C1.T, C2.T, p, q, loss_fun, nx)

    cpt = 0
    err = 1

    if warmstart:
        # initialize potentials to cope with ot.sinkhorn initialization
        N1, N2 = C1.shape[0], C2.shape[0]
        mu = nx.zeros(N1, type_as=C1) - np.log(N1)
        nu = nx.zeros(N2, type_as=C2) - np.log(N2)

    if log:
        log = {'err': []}

    num_loops = 0
    while (err > tol and cpt < max_iter):

        Tprev = T

        # compute the gradient
        if symmetric:
            tens = gwggrad(constC, hC1, hC2, T, nx)
        else:
            tens = 0.5 * (gwggrad(constC, hC1, hC2, T, nx) + gwggrad(constCt, hC1t, hC2t, T, nx))

        if solver == 'PPA':
            tens = tens - epsilon * nx.log(T)

        if warmstart:
            T, loginn = sinkhorn(p, q, tens, epsilon, method='sinkhorn', log=True, warmstart=(mu, nu), **kwargs)
            mu = epsilon * nx.log(loginn['u'])
            nu = epsilon * nx.log(loginn['v'])

        else:
            T = sinkhorn(p, q, tens, epsilon, method='sinkhorn', **kwargs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1
        
        num_loops += 1
    print("Number of loops: ", num_loops)

    if abs(nx.sum(T) - 1) > 1e-5:
        warnings.warn("Solver failed to produce a transport plan. You might "
                      "want to increase the regularization parameter `epsilon`.")
    if log:
        log['gw_dist'] = gwloss(constC, hC1, hC2, T, nx)
        return T, log
    else:
        return T

#%%
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    import pickle as pkl
    from scipy.spatial.distance import cdist
    from src.utils.init_matrix import InitMatrix
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import time
    import torch
    
    
    #%%
    # load data
    data = "THINGS"

    if data == "color":
        data_path = "../../data/color/num_groups_5_seed_0_fill_val_3.5.pickle"
        with open(data_path, "rb") as f:
            data = pkl.load(f)
        sim_mat_list = data["group_ave_mat"]

        sim_mat1 = sim_mat_list[0]
        sim_mat2 = sim_mat_list[1]

    if data == "AllenBrain":
        emb1 = np.load("../../data/AllenBrain/pseudo_mouse_A_emb.npy")
        emb2 = np.load("../../data/AllenBrain/pseudo_mouse_B_emb.npy")
        sim_mat1 = cdist(emb1, emb1, metric="cosine")
        sim_mat2 = cdist(emb2, emb2, metric="cosine")

    if data == "THINGS":
        emb1 = np.load("../../data/THINGS/male_embeddings.npy")[0]
        emb2 = np.load("../../data/THINGS/female_embeddings.npy")[0]
        sim_mat1 = cdist(emb1, emb1, metric="euclidean")
        sim_mat2 = cdist(emb2, emb2, metric="euclidean")
        
        ### add noise
        noise = np.random.normal(0, 5, sim_mat1.shape)
        sim_mat1 += noise


    p = np.ones(sim_mat1.shape[0]) / sim_mat1.shape[0]
    q = np.ones(sim_mat2.shape[0]) / sim_mat2.shape[0]

    #T_init = np.outer(p, q)
    
    epsilon = 5
    n_init = 10
    
    # compare the time of calculation
    # count the time of calculation
    best_T = None
    best_gwd = np.inf
    t_start = time.time()
    for i in tqdm(range(n_init)):
        test_builder = InitMatrix(sim_mat1.shape[0], sim_mat1.shape[1])
        T_init = test_builder.make_initial_T('random', seed=i)
        T_faster_ent, log = faster_gromov_wasserstein(sim_mat1, sim_mat2, p, q, epsilon, log=True, G0=T_init)
        if log['gw_dist'] < best_gwd:
            best_gwd = log['gw_dist']
            best_T = T_faster_ent
    t_end = time.time()
    
    # show the best T
    plt.figure()
    plt.imshow(best_T)
    plt.colorbar()
    plt.title("Faster Entropic GWOT")
    plt.show()
    plt.gcf().clear()
    print("Faster Entropic GWOT time: ", t_end - t_start)
    print("Best GW distance: ", best_gwd)
    
    
    best_T = None
    best_gwd = np.inf
    t_start = time.time()
    for i in tqdm(range(n_init)):
        test_builder = InitMatrix(sim_mat1.shape[0], sim_mat1.shape[1])
        T_init = test_builder.make_initial_T('random', seed=i)
        T_ent, log = entropic_gromov_wasserstein(sim_mat1, sim_mat2, p, q, "square_loss", epsilon, G0=T_init, max_iter=1000, tol=1e-9, verbose=False, log=True)
        if log['gw_dist'] < best_gwd:
            best_gwd = log['gw_dist']
            best_T = T_ent
    t_end = time.time()
    # show the best T
    plt.figure()
    plt.imshow(best_T)
    plt.colorbar()
    plt.title("Entropic GWOT")
    plt.show()
    plt.gcf().clear()
    print("Entropic GWOT time: ", t_end - t_start)
    print("Best GW distance: ", best_gwd)
    
    
    best_T = None
    best_gwd = np.inf
    t_start = time.time()
    for i in tqdm(range(n_init)):
        test_builder = InitMatrix(sim_mat1.shape[0], sim_mat1.shape[1])
        T_init = test_builder.make_initial_T('random', seed=i)
        T_normal, log = ot.gromov.gromov_wasserstein(sim_mat1, sim_mat2, p, q, "square_loss", G0=T_init, max_iter=1000, tol=1e-9, verbose=False, log=True)
        if log['gw_dist'] < best_gwd:
            best_gwd = log['gw_dist']
            best_T = T_normal
    t_end = time.time()
    # show the best T
    plt.figure()
    plt.imshow(best_T)
    plt.colorbar()
    plt.title("Normal GWOT")
    plt.show()
    plt.gcf().clear()
    print("Normal GWOT time: ", t_end - t_start)
    print("Best GW distance: ", best_gwd)
    
    
    #%%
    # compare the time of calculation of emd and sinkhorn
    # set the random data
    
    # cpu vs gpu
    # float vs double
    n = 100
    np.random.seed(0)
    a = np.ones(n) / n
    b = np.ones(n) / n
    M = np.random.rand(n, n)
    
    epsilons = np.logspace(-3.5, 0, 100)
    #epsilons = np.logspace(-1, 0, 10)
    devices = ["cuda"]
    types = ["float32", "float64"]
    sinkhorn_log = True
    
    # count the time of calculation
    t_start = time.time()
    ot.emd(a, b, M, numItermax=100000, log=False)
    t_end = time.time()
    t_emd = t_end - t_start
    print("emd time: ", t_emd)
    
    t_sinkhorn_all = {}
    for device in devices:
        for dtype in types:
            t_sinkhorn = []
            dtype = torch.float32 if dtype == "float32" else torch.float64
            a, b, M = torch.tensor(a, device=device, dtype=dtype), torch.tensor(b, device=device, dtype=dtype), torch.tensor(M, device=device, dtype=dtype)
            
            for eps in epsilons:
                t_start = time.time()
                #ot.bregman.sinkhorn(a, b, M, eps, numItermax=100000, log=False)
                if sinkhorn_log:
                    ot.bregman.sinkhorn_log(a, b, M, eps, numItermax=100000, log=False, device=device, dtype=dtype, stopThr=1e-5) #stopThr=1e-9
                else:
                    ot.bregman.sinkhorn(a, b, M, eps, numItermax=100000, log=False, device=device, dtype=dtype)
                t_end = time.time()

                t = t_end - t_start
                print(f"epsilon: {eps} \n sinkhorn time: ", t)
                t_sinkhorn.append(t)
            t_sinkhorn_all[(device, dtype)] = t_sinkhorn
    
    #%%
    # plot the time of calculation
    
    # sinkhorn_log
    # float vs double
    
    import os, sys
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import time
    import torch
    import ot
    import numpy as np
    from tqdm.auto import tqdm
    
    #%%
    n = 2000
    np.random.seed(0)
    
    epsilons = np.logspace(-3.5, 0, 100)
    devices = ["cuda"]
    types = ["float", "double"]
    
    # count the time of calculation
    a = np.ones(n) / n
    b = np.ones(n) / n
    M = np.random.rand(n, n)
            
    t_start = time.time()
    ot.emd(a, b, M, numItermax=100000, log=False)
    t_end = time.time()
    t_emd = t_end - t_start
    print("emd time: ", t_emd)
    
    
    sinkhorn_log = True
    #%%
    t_sinkhorn_all = {}
    for device in devices:
        for dtype in types:
            dtype = torch.float32 if dtype == "float" else torch.double
            
            torch_a = torch.tensor(a, device=device, dtype=dtype).clone().detach()
            torch_b = torch.tensor(b, device=device, dtype=dtype).clone().detach()
            torch_M = torch.tensor(M, device=device, dtype=dtype).clone().detach()
            
            t_sinkhorn_log = []
            for eps in tqdm(epsilons):
                t_start = time.time()
                ot.bregman.sinkhorn_log(torch_a, torch_b, torch_M, eps, numItermax=100000, log=True)#, stopThr=1e-5) 
                t_end = time.time()
                
                t = t_end - t_start
                # print(f"epsilon: {eps} \n sinkhorn time: ", t)
                t_sinkhorn_log.append(t)
            
            t_sinkhorn_all[(device, dtype)] = t_sinkhorn_log
    
    # %%
    import pickle
    with open(f"../figures/sinkhorn_log, N={n}_default.pkl","wb") as f:
        pickle.dump(t_sinkhorn_all, f)
    
    # %%
    import pickle
    with open(f"../figures/sinkhorn_log, N={n}_default.pkl","rb") as f:
         t_sinkhorn_all = pickle.load(f)
    
    #%%
    plt.figure()
    for key, value in t_sinkhorn_all.items():
        plt.plot(epsilons, value, label=f"sinkhorn_log_{key[0]}_{key[1]}")
    plt.axhline(y=t_emd, color='r', linestyle='-', label="emd")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("epsilon")
    plt.ylabel("time")
    if sinkhorn_log:
        plt.title(f"Time of calculation of sinkhorn_log \n N={n}")
    else:
        plt.title(f"Time of calculation of emd and sinkhorn \n N={n}")
    plt.legend()
    plt.savefig(f"../figures/time_of_calculation_emd_sinkhorn_log_{n}.png")
    plt.show()
    plt.gcf().clear()
    
    
    #%%
    double_results = t_sinkhorn_all[("cuda", torch.double)]
    float_results = t_sinkhorn_all[("cuda", torch.float32)]
    
    values = np.array(double_results) / np.array(float_results)
    plt.figure()    
    plt.plot(epsilons, values, label="time ratio (double / float)")
    plt.xlabel("epsilon")
    plt.ylabel("double / float")
    plt.xscale("log")
    plt.title(f"time ratio (double / float) \n N={n}")
    plt.savefig(f"../figures/time_of_calculation_emd_sinkhorn_log_{n}.png")
    plt.show()
    plt.gcf().clear()
    
    
    
# %%
