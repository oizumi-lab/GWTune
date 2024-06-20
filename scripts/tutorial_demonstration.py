#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../'))

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.spatial import distance

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig

#%%
# 絡まった紐の座標を生成する関数
def tangled_rope(num_points=100, num_loops=3, radius=1, beta=0.3, pattern=1):
    t = np.linspace(0, 2 * np.pi * num_loops, num_points)
    
    # set radius as a function of t
    if pattern == 1:
        radius = radius + beta * t / (2 * np.pi)
    elif pattern == 2:
        radius = radius + beta * t / (2 * np.pi) + np.sin(t / (2 * np.pi))
    
    x = np.sin(t) * radius
    y = np.cos(t) * radius
    z = t / (2 * np.pi)
    
    embedding = np.column_stack((x, y, z))
    return embedding

# 絡まった紐の座標を生成
embedding_1 = tangled_rope(beta=0.3, pattern=1)
embedding_2 = tangled_rope(beta=0.3, pattern=2)

# %%
# 3次元プロットの設定
def plot(embedding):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 色を連続的に変化させる
    norm = Normalize(vmin=min(embedding[:, 2]), vmax=max(embedding[:, 2]))
    cmap = plt.get_cmap('cool')
    scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    colors = scalar_map.to_rgba(embedding[:, 2])

    # 絡まった紐をプロット
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=colors)

    # 軸ラベルの設定
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 表示
    plt.show()
    return colors
    

colors = plot(embedding_1)
colors = plot(embedding_2)

# %%
Group1 = Representation(name="1", metric="euclidean", embedding=embedding_1)
Group2 = Representation(name="2", metric="euclidean", embedding=embedding_2)

# %%
vis_emb = VisualizationConfig(
    figsize=(8, 8), 
    legend_size=20,
    cbar_ticks_size=20,
    color_labels=colors,
    fig_ext='svg',
)

vis_emb_2 = VisualizationConfig(
    figsize=(8, 8), 
    legend_size=12,
    marker_size=60,
    color_labels=colors,
    fig_ext='svg',
    markers_list=['X'],
)


#%%
# show embeddings
Group1.show_embedding(dim=3, visualization_config=vis_emb, fig_name="Group1", legend=False)
Group2.show_embedding(dim=3, visualization_config=vis_emb_2, fig_name="Group2", legend=False)



# %%
config = OptimizationConfig(
    eps_list=[0.1, 1],
    num_trial=100,
    db_params={"drivername": "sqlite"},
    n_iter=1,
)

# %%
vis_config = VisualizationConfig(
    figsize=(10, 10), 
    title_size = 0, 
    cmap = "rocket_r",
    cbar_ticks_size=20,
    color_labels=colors,
    fig_ext='svg',
)


#%%
alignment = AlignRepresentations(
    config=config,
    representations_list=[Group1, Group2],
    main_results_dir="../results/demonstration",
    data_name="demonstration"
)

#%%
# RSA
alignment.show_sim_mat(
    visualization_config=vis_config, 
    show_distribution=False,
)

#%%
alignment.RSA_get_corr()

# %%
# GW
compute_OT=False

# %%
alignment.gw_alignment(
    compute_OT=compute_OT,
    delete_results=False,
    visualization_config=vis_config,
    fix_sampler_seed=1,
)



# %%
vis_log = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 0, 
    cmap = "viridis",
    cbar_ticks_size=23,
    xlabel_size=20,
    xticks_size=20,
    ylabel_size=20,
    yticks_size=20,
    cbar_label_size=20,
    marker_size=150,
    plot_eps_log=True,
    fig_ext='svg',
)



# %%
alignment.show_optimization_log(
    fig_dir="../results/demonstration",
    visualization_config=vis_log,
)

# %%
alignment.calc_accuracy(top_k_list=[1, 3, 5], eval_type="ot_plan")

# %%
vis_emb3d = VisualizationConfig(
    figsize=(8, 8), 
    legend_size=20,
    marker_size=60,
    color_labels=colors,
    fig_ext='svg',
    cmap="cool",
    colorbar_shrink=0.8,
)

# %%
alignment.visualize_embedding(
    dim=3, 
    pivot=0, 
    visualization_config=vis_emb3d, 
    fig_dir="../results/demonstration"
)


# %%
# new_method
pair = alignment.pairwise_list[0]

#%%
ot = pair.OT

# %%
source = pair.source.embedding
target = pair.target.embedding

# %%
new_source = pair.OT.T @ target * len(target)
new_target = pair.OT @ source * len(source)

# %%
new_rep_list = [
    Representation(name="Data 1", embedding=source),
    Representation(name="Data 2", embedding=new_target),
]

# %%
ar = AlignRepresentations(
    config=config,
    representations_list=new_rep_list,
    main_results_dir="../results/demnstration",
    data_name="demonstration"
)

# %%
# embedding alignment
emb_name = "PCA" #"TSNE", "PCA", "MDS"
dim=3
ar.visualize_embedding(dim=3, method=emb_name, visualization_config=vis_emb3d, fig_dir="../results/demonstration")
#%%
