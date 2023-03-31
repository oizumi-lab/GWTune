# %%
import os, sys
import jax
import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from IPython import display
from matplotlib import animation, cm

from ott.geometry import pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein
import ott
import torch

# %%
path1 = '../data/model1_1000.pt'
path2 = '../data/model2_1000.pt'
unittest_save_path = '../results/experiment/jax_test'

model1 = torch.load(path1).to('cpu')
model2 = torch.load(path2).to('cpu')

# %%
model1_jax = jnp.array(model1)
model2_jax = jnp.array(model2)

# %%
# apply https://ott-jax.readthedocs.io/en/latest/geometry.html
geom1 = ott.geometry.geometry.Geometry(cost_matrix = model1_jax)
geom2 = ott.geometry.geometry.Geometry(cost_matrix = model2_jax)
# prob = quadratic_problem.QuadraticProblem(geom1, geom2, tolerances = 1e-9)
# solver = gromov_wasserstein.GromovWasserstein(epsilon=1e-5, max_iterations=100)
# out = solver(prob)

# %%
solver = ott.solvers.linear.sinkhorn.Sinkhorn(threshold = 1e-9)
out = ott.solvers.quadratic.gromov_wasserstein.solve(geom1, geom2, epsilon=1e-3, linear_ot_solver = solver, min_iterations=100, max_iterations=1000)

n_outer_iterations = jnp.sum(out.costs != -1)
has_converged = bool(out.linear_convergence[n_outer_iterations - 1])
print(f"{n_outer_iterations} outer iterations were needed.")
print(f"The last Sinkhorn iteration has converged: {has_converged}")
print(f"The outer loop of Gromov Wasserstein has converged: {out.converged}")
print(f"The final regularized GW cost is: {out.reg_gw_cost:.3f}")


transport = out.matrix
fig = plt.figure(figsize=(8, 6))
plt.imshow(transport, cmap="inferno")
plt.xlabel("model 1")
plt.ylabel("model 2")
plt.colorbar()
plt.show()


# %%
