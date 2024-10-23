# implement simulated annealing algorithm

#%%
import numpy as np
import matplotlib.pyplot as plt
import ot
import time
import pickle as pkl
import tqdm
from scipy.spatial.distance import cdist

ot.gromov.entropic_gromov_wasserstein

#%%
def entropic_gromov_wasserstein(C1, C2, p, q, loss_fun, epsilon,T=None,
                                max_iter=1000, tol=1e-9, verbose=False, log=False):
    C1, C2, p, q = ot.utils.list_to_array(C1, C2, p, q)
    nx = ot.backend.get_backend(C1, C2, p, q)
    # add T as an input
    if T is None:
      T = nx.outer(p, q)
    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)
    cpt = 0
    err = 1
    if log:
        log = {'err': []}
    while (err > tol and cpt < max_iter):
        Tprev = T
        # compute the gradient
        tens = ot.gromov.gwggrad(constC, hC1, hC2, T)
        T = ot.bregman.sinkhorn(p, q, tens, epsilon, method='sinkhorn')
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
    if log:
        log['gw_dist'] = ot.gromov.gwloss(constC, hC1, hC2, T)
        return T, log
    else:
        return T

def gromov_wasserstein(C1, C2, p=None, q=None, loss_fun='square_loss', symmetric=None, log=False, armijo=False, G0=None,
                       max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, **kwargs):
    
    arr = [C1, C2]
    if p is not None:
        arr.append(ot.utils.list_to_array(p))
    else:
        p = ot.unif(C1.shape[0], type_as=C1)
    if q is not None:
        arr.append(ot.utils.list_to_array(q))
    else:
        q = ot.unif(C2.shape[0], type_as=C2)
    if G0 is not None:
        G0_ = G0
        arr.append(G0)

    nx = ot.utils.get_backend(*arr)
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
    np_ = ot.backend.NumpyBackend()

    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun, np_)

    def f(G):
        return ot.gromov.gwloss(constC, hC1, hC2, G, np_)

    if symmetric:
        def df(G):
            return ot.gromov.gwggrad(constC, hC1, hC2, G, np_)
    else:
        constCt, hC1t, hC2t = ot.gromov.init_matrix(C1.T, C2.T, p, q, loss_fun, np_)

        def df(G):
            return 0.5 * (ot.gromov.gwggrad(constC, hC1, hC2, G, np_) + ot.gromov.gwggrad(constCt, hC1t, hC2t, G, np_))
    if loss_fun == 'kl_loss':
        armijo = True  # there is no closed form line-search with KL

    if armijo:
        def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
            return ot.optim.line_search_armijo(cost, G, deltaG, Mi, cost_G, nx=np_, **kwargs)
    else:
        def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
            return ot.gromov.solve_gromov_linesearch(G, deltaG, cost_G, C1, C2, M=0., reg=1., nx=np_, **kwargs)
    if log:
        res, log = ot.optim.cg(p, q, 0., 1., f, df, G0, line_search, log=True, numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs)
        log['gw_dist'] = nx.from_numpy(log['loss'][-1], type_as=C10)
        log['u'] = nx.from_numpy(log['u'], type_as=C10)
        log['v'] = nx.from_numpy(log['v'], type_as=C10)
        return nx.from_numpy(res, type_as=C10), log
    else:
        return nx.from_numpy(ot.optim.cg(p, q, 0., 1., f, df, G0, line_search, log=False, numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs), type_as=C10)
    
    
#%%
# set up the parameters
eps_steps = 100
eps_range = [1e-1, 100]

epsilons = np.logspace(np.log10(eps_range[1]), np.log10(eps_range[0]), eps_steps)

#%%
# load the data
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
    

p = np.ones(sim_mat1.shape[0]) / sim_mat1.shape[0]
q = np.ones(sim_mat2.shape[0]) / sim_mat2.shape[0]

T_init = np.outer(p, q)
#%%
gwds = []
entropic_Ts = {}
final_Ts = {}
for i, epsilon in tqdm.tqdm(enumerate(epsilons)):
    start = time.time()
    T, log = entropic_gromov_wasserstein(sim_mat1, sim_mat2, p, q, "square_loss", epsilon, T_init, max_iter=1000, tol=1e-9, verbose=False, log=True)
    end = time.time()
    #print("epsilon: ", epsilon)
    #print("time: ", end - start)
    #print("gw_dist: ", log["gw_dist"])
    #print("err: ", log["err"][-1])
    #print("\n")
    
    T_init = T
    gwds.append(log["gw_dist"])
    
    if i % 10 == 0:
        plt.figure()
        plt.imshow(T)
        plt.colorbar()
        plt.title(f"GWOT, epsilon: {epsilon}")
        plt.show()
        plt.gcf().clear()
        entropic_Ts[epsilon] = T
        
        try:
            T_fin, log = ot.gromov.gromov_wasserstein(sim_mat1, sim_mat2, p, q, loss_fun='square_loss', symmetric=None, log=True, armijo=False, G0=T_init,)
            plt.figure()
            plt.imshow(T_fin)
            plt.colorbar()
            plt.show()
            plt.gcf().clear()
            final_Ts[epsilon] = T_fin
            
        except AssertionError as e:
            print(e)
            break
        
#%%
# visualize Ts for different epsilons
plt.figure()
plt.subplots(2, len(final_Ts), figsize=(30, 10))

for i, epsilon in enumerate(final_Ts.keys()):
    plt.subplot(2, len(final_Ts), i + 1)
    plt.imshow(entropic_Ts[epsilon])
    #plt.colorbar()
    plt.title(f"GWOT, epsilon: {epsilon:.2f}")
    
    plt.subplot(2, len(final_Ts), i + 1 + len(final_Ts))
    plt.imshow(final_Ts[epsilon])
    #plt.colorbar()
    plt.title(f"GWOT")
        
    
# %%
print(gwds)
# %%
