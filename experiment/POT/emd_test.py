# %%
import numpy as np
import scipy as sp
import ot

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import ot

if __name__ == "__main__":
    n = 10
    np.random.seed(42)
    x1 = np.random.rand(10)
    
    np.random.seed(44)
    x2 = np.random.rand(10)
    
    x1_prob = ot.unif(len(x1))
    x2_prob = ot.unif(len(x2))

    # make distance matrix
    dist = cdist(x1[:,np.newaxis], x2[:,np.newaxis])
    
    # EMD Loss from pot libraray
    ot_emd = ot.emd2(x1_prob, x2_prob, dist)
    print(ot_emd)
    
    # EMD Loss from pot libraray
    ot_emd2_1d = ot.emd2_1d(x1, x2, metric='euclidean', p = 2)
    print(ot_emd2_1d)
    
    # linear sum assignment ref : https://blog.mktia.com/emd-with-python/
    assignment = linear_sum_assignment(dist)
    emd = dist[assignment].sum() / n
    print(emd) 
    
    # wasserstain distance by scipy
    wd = wasserstein_distance(x1, x2)
    print(wd)
    
# %%
