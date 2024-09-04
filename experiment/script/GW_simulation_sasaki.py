#%%
import os, sys
set_cpu = 4
os.environ["OPENBLAS_NUM_THREADS"] = str(set_cpu)
os.environ["MKL_NUM_THREADS"] = str(set_cpu)
os.environ["OMP_NUM_THREADS"] = str(set_cpu)

sys.path.append(os.path.join(os.getcwd(), '../../'))

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.spatial import distance

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


from src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig

#%%
### Pattern 1

def create_circle(N, center=(0,0), radius=10):
    center = np.array(center)

    angles = np.linspace(0, 2*np.pi, N, endpoint=False)

    x_coords = center[0] + radius * np.cos(angles)
    y_coords = center[1] + radius * np.sin(angles)
    points = np.column_stack((x_coords, y_coords))
    
    return points

def add_random_offset(points, max_offset, seed=0):
    angles = np.arctan2(points[:, 1], points[:, 0])
    
    np.random.seed(seed)
    random_offsets = np.random.uniform(-max_offset, max_offset, size=points.shape[0])
    
    points[:, 0] += random_offsets * np.cos(angles)
    points[:, 1] += random_offsets * np.sin(angles)
    
    return points

def add_independent_noise(points, max_noise, seed=0):
    np.random.seed(seed)
    noise_x = np.random.uniform(-max_noise, max_noise, size=points.shape[0])
    noise_y = np.random.uniform(-max_noise, max_noise, size=points.shape[0])
    
    points_with_noise = points + np.column_stack((noise_x, noise_y))
    
    return points_with_noise

def rotate_points_around_center(points, max_offset, start=0, seed=0):
    np.random.seed(seed)
    center = np.mean(points, axis=0)
    num_points = len(points)
    angles = np.random.uniform(-max_offset, max_offset, num_points)
    
    rotated_points = []
    for i in range(num_points):
        idx = (i + start)%num_points
        angle = angles[idx]
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        rotated_points.append(np.dot(points[i] - center, rotation_matrix.T) + center)
    
    return np.array(rotated_points)

#%%
### Pattern 4
# dataset = "Simulation"
dataset = "Simulation_noise"

#%%
# 絡まった紐の座標を生成する関数
def tangled_rope(num_points=100, num_loops=3, radius=1, beta=0.3, pattern=1):
    
    t = np.linspace(0, 2 * np.pi * num_loops, num_points)
    
    # set radius as a function of t
    if pattern == 1:
        radius = radius + beta * t / (2 * np.pi)
    elif pattern == 2:
        radius = radius + beta * t / (2 * np.pi) + np.sin(t / (2 * np.pi))
    
    np.random.seed(pattern)
    noize = np.random.normal(0, 0.07, num_points)
    
    if dataset == "Simulation_noise":   
        x = np.sin(t) * radius + noize
        y = np.cos(t) * radius + noize
        z = t / (2 * np.pi) + noize
    
    elif dataset == "Simulation":
        x = np.sin(t) * radius
        y = np.cos(t) * radius
        z = t / (2 * np.pi)
        
    embedding = np.column_stack((x, y, z))
    return embedding

# 絡まった紐の座標を生成
embedding_1 = tangled_rope(beta=0.3, pattern=1)
embedding_2 = tangled_rope(beta=0.3, pattern=2)

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
    
#%%
colors = plot(embedding_1)
colors = plot(embedding_2)
# %%
Group1 = Representation(
    name="Embedding X",
    metric="euclidean",
    embedding=embedding_1,
)

# %%
Group2 = Representation(
    name="Embedding Y",
    metric="euclidean",
    embedding=embedding_2,
)

# %%
config = OptimizationConfig(
    eps_list=[1e-3, 1],
    num_trial=100,
    sinkhorn_method = 'sinkhorn_log',
    db_params={"drivername": "sqlite"},
    n_iter=1,
)

# %%
vis_sim_mat= VisualizationConfig(
    figsize=(10, 10), 
    title_size = 0, 
    cmap = "rocket_r",
    cbar_ticks_size=30,
    font="Arial",
    cbar_label="Dissimilarity",
    cbar_label_size=40,
    color_labels=colors,
    color_label_width=3,
    fig_ext='svg',
)

#%%
vis_ot = VisualizationConfig(
    figsize=(10, 10), 
    title_size = 0, 
    cmap = "rocket_r",
    cbar_ticks_size=30,
    font="Arial",
    cbar_label="Probability",
    cbar_label_size=40,
    color_labels=colors,
    color_label_width=3,
    fig_ext='svg',
)

# %%
vis_emb = VisualizationConfig(
    figsize=(8, 8), 
    legend_size=20,
    cbar_ticks_size=20,
    color_labels=colors,
    fig_ext='svg',
)

# %%
vis_emb_2 = VisualizationConfig(
    figsize=(8, 8), 
    legend_size=12,
    marker_size=60,
    color_labels=colors,
    fig_ext='svg',
    markers_list=['X'],
)

vis_log = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 0, 
    cmap = "viridis",
    cbar_ticks_size=23,
    font="Arial",
    xlabel_size=20,
    xticks_size=20,
    ylabel_size=20,
    yticks_size=20,
    cbar_label_size=20,
    marker_size=60,
    plot_eps_log=True,
    fig_ext='svg',
)

# %%
vis_emb3d = VisualizationConfig(
    figsize=(8, 8), 
    legend_size=20,
    marker_size=60,
    color_labels=colors,
    fig_ext='svg',
    markers_list=['o', 'X'],
    # xlabel="PC1",
    # ylabel="PC2",
    # zlabel="PC3",
    font="Arial",
    cmap="cool",
    # colorbar_label="short movies",
    # colorbar_range=[0, len(embedding_1)],
    colorbar_shrink=0.8,
    # xlabel_size=20,
    # ylabel_size=20,
    # zlabel_size=20,
)


#%%
alignment = AlignRepresentations(
    config=config,
    representations_list=[Group1, Group2],
    main_results_dir=f"../results/{dataset}",
    data_name=dataset,
)

#%%
# RSA
alignment.show_sim_mat(
    visualization_config=vis_sim_mat, 
    show_distribution=False,
)

#%%
alignment.RSA_get_corr()

#%%
# show embeddings
Group1.show_embedding(dim=3, fig_dir=f"../results/{dataset}", visualization_config=vis_emb, fig_name="Embedding X", legend=False)
Group2.show_embedding(dim=3, fig_dir=f"../results/{dataset}", visualization_config=vis_emb_2, fig_name="Embedding Y", legend=False)

# %%
# GW
compute_OT=False

# %%
alignment.gw_alignment(
    compute_OT=compute_OT,
    delete_results=False,
    visualization_config=vis_ot,
    save_dataframe=True,
    fix_random_init_seed=True,
)

# %%
alignment.show_optimization_log(
    fig_dir=f"../results/{dataset}",
    visualization_config=vis_log,
)

#%%
study = alignment.pairwise_list[0]._run_optimization(compute_OT = False)
df_trial = study.trials_dataframe()

#%%
plt.figure(figsize=(8,6))
plt.scatter(df_trial["params_eps"], df_trial["value"], s = 60)
plt.xlabel("epsilon", fontsize=20)
plt.ylabel("GWD", fontsize=20)
plt.xscale('log')

plt.tick_params(axis='x', which='both', labelsize=20, rotation=0)
plt.tick_params(axis='y', which='major', labelsize=20)
plt.tight_layout()
plt.savefig(f"../results/{dataset}/eps_gwd.svg")
plt.show()

# %%
_emb = alignment.visualize_embedding(dim=3, visualization_config=vis_emb3d, name_list=["Embedding X", "Embedding Y"])

# %%
df = alignment.calc_accuracy(top_k_list=[1, 3, 5], eval_type="ot_plan", return_dataframe=True)
print(df)
# %%
df_plot = df
df_plot.index = ["Top 1", "Top 3", "Top 5"]
df_plot.T.plot(kind='bar', figsize=(8, 6), fontsize=20, rot=0)
plt.xticks([])
plt.xlabel("Top k",fontsize=20)
plt.ylabel("Matching Rate (%)", fontsize=20)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(f"../results/{dataset}/top_k_matching_rate.svg")
plt.show()

# %%
pair = alignment.pairwise_list[0]

#%%
df = pair.random_test_gwot_no_entropy(num_seed = 200)

#%%
plt.scatter(df["acc"], df["gwd"])
plt.title(f"GWOT no entropy \n {pair.data_name} / {pair.pair_name.replace('_', ' ')}")
plt.xlabel("Accuracy (%)")
plt.ylabel("GWD")
plt.xlim(-5, 105)
plt.grid(True)
plt.show()

#%%
df.sort_values("gwd", ascending=True).head(10)


# %%
# new_method
pair = alignment.pairwise_list[0]

#%%
ot = pair.OT
plt.imshow(ot)
plt.show()
# %%
source = pair.source.embedding
target = pair.target.embedding

# %%
print(source.shape)
print(target.shape)

# %%
new_source = pair.OT.T @ target * len(target)
new_target = pair.OT @ source * len(source)

# %%
print(new_source.shape)
print(new_target.shape)

# %%
new_rep_list = [
    Representation(name="Embedding X", embedding=source),
    Representation(name="Embedding Y", embedding=new_target),
]

# %%
ar = AlignRepresentations(
    config=config,
    representations_list=new_rep_list,
    main_results_dir=f"../results/{dataset}",
    data_name=dataset,
)

# %%
# embedding alignment
emb_name = "PCA" #"TSNE", "PCA", "MDS"
dim=3
ar.new_visualize_emb(dim=dim, method=emb_name, visualization_config=vis_emb3d)
#%%
