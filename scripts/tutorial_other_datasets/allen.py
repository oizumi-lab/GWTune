# %%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../../'))

import numpy as np
from src.align_representations import Representation, AlignRepresentations, OptimizationConfig

# %%
data_select = "AllenBrain"

# %%
representations = []
for name in ["pseudo_a_VISal", "pseudo_b_VISam"]:
    emb = np.load(f"../../data/AllenBrain/{name}.npy")
    representation = Representation(
        name=name,
        embedding=emb, 
        metric="cosine",
        get_embedding=False,
        object_labels=np.arange(emb.shape[0]) 
    )
    representations.append(representation)

# %%
eps_list_tutorial = [1e-4, 1e-1]
device = 'cpu'
to_types = 'numpy'
multi_gpu = False

eps_log = True
num_trial = 8910
# init_mat_plan = 'random'
# sampler_name = 'tpe'

init_mat_plan = 'uniform'
sampler_name = 'grid'

# init_mat_plan = 'random'
# sampler_name = 'grid'

# %%
config = OptimizationConfig(    
    eps_list = eps_list_tutorial,
    eps_log = eps_log,
    num_trial = num_trial,
    sinkhorn_method='sinkhorn_log',
    to_types = to_types,
    device = device,
    data_type = "double", 
    n_jobs = 1,
    multi_gpu = multi_gpu, 
    db_params={"drivername": "sqlite"},
    init_mat_plan = init_mat_plan,
    n_iter = 1,
    max_iter = 200,
    sampler_name = sampler_name,
)
# %%
# Create an "AlignRepresentations" instance
align_representation = AlignRepresentations(
    config=config,
    representations_list=representations,   
    histogram_matching=False,
    main_results_dir = f"../../results/{data_select}/{sampler_name}",
    data_name = data_select,
)

# %%
delete_results = False

# %%
align_representation.gw_alignment(
    compute_OT = True,
    delete_results = delete_results,
    return_data = False,
)

#%%