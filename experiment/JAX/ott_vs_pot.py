# %% [markdown]
# ## OTT vs. POT   

# The Python Optimal Transport (POT) toolbox paved the way for much progress in OT.  
# <br>
# POT implements several OT solvers (LP and regularized), and is complemented with various tools (e.g., barycenters, domain adaptation, Gromov-Wasserstein distances, sliced W, etc.).
# <br>
# The goal of this notebook is to compare the performance of OTT's and POT's <ot.sinkhorn> Sinkhorn solvers. 
# <br>
# OTT benefits from just-in-time compilation, which should give it an edge.
# <br>
# The comparisons carried out below have limitations: minor modifications in the setup (e.g., data distributions, tolerance thresholds, accelerator...) could have an impact on these results.  
# <br>  
# Feel free to change these settings and experiment by yourself!


# %%
# https://github.com/ott-jax/ott/blob/main/docs/tutorials/notebooks/OTT_%26_POT.ipynb
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import ot

import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1

import ott
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

plt.rc("font", size=20)

def solve_ot(a, b, x, y, 𝜀, threshold):
    _, log = ot.sinkhorn(a, b, ot.dist(x, y), 𝜀, stopThr=threshold, method="sinkhorn_stabilized", log=True, numItermax=1000)
    
    f, g = 𝜀 * log["logu"], 𝜀 * log["logv"]
    f, g = f - np.mean(f), g + np.mean(f)  # center variables, useful if one wants to compare them
    reg_ot = (np.sum(f * a) + np.sum(g * b) if log["err"][-1] < threshold else np.nan)
    return f, g, reg_ot


@jax.jit
def solve_ott(a, b, x, y, 𝜀, threshold):
    geom = pointcloud.PointCloud(x, y, epsilon=𝜀)
    prob = linear_problem.LinearProblem(geom, a=a, b=b)

    solver = sinkhorn.Sinkhorn(threshold=threshold, lse_mode=True, max_iterations=1000)
    out = solver(prob)

    f, g = out.f, out.g
    f, g = f - np.mean(f), g + np.mean(f)  # center variables, useful if one wants to compare them
    reg_ot = jnp.where(out.converged, jnp.sum(f * a) + jnp.sum(g * b), jnp.nan)
    return f, g, reg_ot

def run_simulation(rng, n, 𝜀, threshold, solver_spec):
    #  setting global variables helps avoir a timeit bug.
    global solver_
    global a, b, x, y

    # extract specificities of solver.
    solver_, env, name = solver_spec

    # draw data at random using JAX
    rng, *rngs = jax.random.split(rng, 5)
    x = jax.random.uniform(rngs[0], (n, dim))
    y = jax.random.uniform(rngs[1], (n, dim)) + 0.1
    
    a = jax.random.uniform(rngs[2], (n,))
    b = jax.random.uniform(rngs[3], (n,))
    
    a = a / jnp.sum(a) # ここで、確率に変えている
    b = b / jnp.sum(b)

    # map to numpy if needed #ここの書き方は使えそう。
    if env == "np":
        a, b, x, y = map(np.array, (a, b, x, y))

    start = time.time()
    out = solver_(a, b, x, y, 𝜀, threshold) # 関数を変数のように使う書き方。使い方次第。
    end = time.time() - start
    exec_time = np.nan if np.isnan(out[-1]) else end
    return exec_time, out


dim = 3

POT = (solve_ot, "np", "POT")
OTT = (solve_ott, "jax", "OTT")

rng = jax.random.PRNGKey(0)
solvers = (POT, OTT)
n_range = 2 ** np.arange(8, 13)
𝜀_range = 10 ** np.arange(-2.0, 0.0)

threshold = 1e-2

exec_time = {}
reg_ot = {}
for solver_spec in solvers:
    solver, env, name = solver_spec
    
    print("----- ", name)
    exec_time[name] = np.ones((len(n_range), len(𝜀_range))) * np.nan
    reg_ot[name] = np.ones((len(n_range), len(𝜀_range))) * np.nan
    
    for i, n in enumerate(n_range):
        for j, 𝜀 in enumerate(𝜀_range):
            t, out = run_simulation(rng, n, 𝜀, threshold, solver_spec)
            exec_time[name][i, j] = t
            reg_ot[name][i, j] = out[-1]


# %%
# ここからはグラフ化をおこなうだけ
list_legend = []
fig = plt.figure(figsize=(14, 8))

for solver_spec, marker, col in zip(solvers, ("p", "o"), ("blue", "red")):
    solver, env, name = solver_spec
    p = plt.plot(
        exec_time[name],
        marker=marker,
        color=col,
        markersize=16,
        markeredgecolor="k",
        lw=3,
    )
    p[0].set_linestyle("dotted")
    p[1].set_linestyle("solid")
    list_legend += [name + r"  $\varepsilon $=" + f"{𝜀:.2g}" for 𝜀 in 𝜀_range]

plt.xticks(ticks=np.arange(len(n_range)), labels=n_range)
plt.legend(list_legend)
plt.yscale("log")
plt.xlabel("dimension $n$")
plt.ylabel("time (s)")
plt.title(r"Execution Time vs Dimension for OTT and POT for two $\varepsilon$ values")
plt.show()

fig = plt.figure(figsize=(12, 8))
ax = plt.gca()
im = ax.imshow(reg_ot["OTT"].T - reg_ot["POT"].T)
plt.xticks(ticks=np.arange(len(n_range)), labels=n_range)
plt.yticks(ticks=np.arange(len(𝜀_range)), labels=𝜀_range)
plt.xlabel("dimension $n$")
plt.ylabel(r"regularization $\varepsilon$")
plt.title("Gap in objective, >0 when OTT is better")
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax)
plt.show()

for name in ("POT", "OTT"):
    print("----", name)
    print("Objective")
    print(reg_ot[name])
    print("Execution Time")
    print(exec_time[name])
# %%
