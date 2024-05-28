#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../../'))

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.spatial import distance

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

#def shuffle_matrix(matrix, N_divide):
#    lower_triangle_elements = _extract_lower_triangle(matrix)
#    for i in range(N_divide):
#        lower_triangle_elements = _rearrange_and_shuffle(lower_triangle_elements, N=N_divide, m=i)
#    shuffled_matrix = _create_symmetric_matrix(lower_triangle_elements, size=matrix.shape[0])
#    return shuffled_matrix
#    
#def _extract_lower_triangle(matrix):
#    lower_triangle_idx = np.tril_indices(matrix.shape[0], k=-1)
#    return matrix[lower_triangle_idx].flatten()
#
#def _create_symmetric_matrix(lower_triangle_array, size):
#    lower_triangle = np.zeros((size, size))
#    lower_triangle[np.tril_indices(size, k=-1)] = lower_triangle_array
#    symmetric_matrix = lower_triangle + lower_triangle.T
#    return symmetric_matrix
#
#
#def _rearrange_and_shuffle(original_array, N, m, seed=0):
#    sorted_array = np.sort(original_array)[::-1]
#
#    array_length = len(sorted_array)
#    segment_length = array_length // N
#
#    start_index = m * segment_length
#    end_index = (m + 1) * segment_length if m < N - 1 else array_length
#    extracted_segment = sorted_array[start_index:end_index]
#
#    np.random.seed(seed)
#    np.random.shuffle(extracted_segment)
#
#    shuffled_array = np.copy(sorted_array)
#    shuffled_array[start_index:end_index] = extracted_segment
#
#    original_order_indices = np.argsort(original_array)[::-1]
#    final_array = shuffled_array[original_order_indices]
#
#    return final_array

#%%
N_points = 50
# get matrices
circle = create_circle(N=N_points)
circle = rotate_points_around_center(circle, max_offset=0.8)
embedding_1 = add_random_offset(circle, max_offset=3)
embedding_2 = add_independent_noise(embedding_1, max_noise=3)
#sim_mat_1 = distance.cdist(embedding, embedding, "euclidean")
#sim_mat_2 = shuffle_matrix(sim_mat_1, N_divide=500)
#np.save("../../data/simulation1_embedding1.npy", embedding_1)
#np.save("../../data/simulation1_embedding2.npy", embedding_2)

Group1 = Representation(
    name="Group1",
    metric="euclidean",
    embedding=embedding_1,
    )

Group2 = Representation(
    name="Group2",
    metric="euclidean",
    embedding=embedding_2,
    
)

config = OptimizationConfig(
    eps_list=[0.1, 1],
    num_trial=50,
    db_params={"drivername": "sqlite"},
    n_iter=1,
)

vis_config = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 0, 
    cmap = "rocket_r",
    cbar_ticks_size=10,
    font="Arial"
)

vis_emb = VisualizationConfig(
    figsize=(8, 8), 
    legend_size=12,
    marker_size=60
)

alignment = AlignRepresentations(
    config=config,
    representations_list=[Group1, Group2],
    main_results_dir="../results/",
    data_name="Simulation_1"
)

# RSA
alignment.show_sim_mat(
    visualization_config=vis_config, 
    show_distribution=False,
    fig_dir="../results/Simulation_1"
    )
alignment.RSA_get_corr()

# show embeddings
Group1.show_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_1", fig_name="Group1")
Group2.show_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_1", fig_name="Group2")
# %%
# GW
alignment.gw_alignment(
    compute_OT=False,
    delete_results=False,
    visualization_config=vis_config
    )

alignment.visualize_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_1")

alignment.calc_accuracy(top_k_list=[1, 3, 5], eval_type="ot_plan")
alignment.calc_accuracy(top_k_list=[1, 3, 5], eval_type="k_nearest")

#%%

### Pattern 2
def generate_uniform_matrix(rows, cols, value):
    # Create an array of the specified size and fill it with the specified value
    matrix = np.full((rows, cols), value)
    return matrix

def add_noise_to_off_diagonal(matrix, mean=0, std=1, seed = None):
    if seed is not None:
        np.random.seed(seed)
    # Create a copy of the input matrix
    noisy_matrix = np.copy(matrix)
    
    # Add random noise to each off-diagonal entry
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if i != j:  # only add noise to off-diagonal entries
                noisy_matrix[i, j] += np.random.normal(mean, std)
    
    noisy_matrix = np.tril(noisy_matrix, k=-1) + np.tril(noisy_matrix, k=-1).T
    
    return noisy_matrix

def get_block_mat(block_sizes):
    matrix = np.zeros((sum(block_sizes), sum(block_sizes)))
    cum_block_sizes = np.cumsum(block_sizes)
    
    for i, size_1 in enumerate(cum_block_sizes):
        for j, size_2 in enumerate(cum_block_sizes[:i + 1]):
            row = block_sizes[i]
            col = block_sizes[j]
            
            start1 = cum_block_sizes[i-1] if i !=0 else 0
            start2 = cum_block_sizes[j-1] if j !=0 else 0
            end1 = cum_block_sizes[i]
            end2 = cum_block_sizes[j]
            # Get the indices of the current block
            block_indices = np.ix_(range(start1, end1), range(start2, end2))
            if i == j:
                value = 0.01
                uniform_mat = np.tril(generate_uniform_matrix(row, col, value), k = -1)
            else:
                value = np.random.uniform(low = 0.4, high = 1)
                uniform_mat = generate_uniform_matrix(row, col, value)
            matrix[block_indices] = uniform_mat
    matrix = matrix + matrix.T
    return matrix

#%%
# get matrices
num_categories = 5
block_sizes = [N_points//num_categories for i in range(num_categories)]
matrix = get_block_mat(block_sizes)
sim_mat_1 = add_noise_to_off_diagonal(matrix, mean=0.2, std=0.08, seed=0)
sim_mat_2 = add_noise_to_off_diagonal(matrix, mean=0.2, std=0.08, seed=1)
#np.save("../../data/simulation2_sim_mat1.npy", sim_mat_1)
#np.save("../../data/simulation2_sim_mat2.npy", sim_mat_2)

Group1 = Representation(
    name="Group1",
    metric="euclidean",
    sim_mat=sim_mat_1,
    get_embedding=True)

Group2 = Representation(
    name="Group2",
    metric="euclidean",
    sim_mat=sim_mat_2,
    get_embedding=True
)

config = OptimizationConfig(
    eps_list=[0.1, 1],
    num_trial=50,
    db_params={"drivername": "sqlite"},
    n_iter=1,
)

vis_config = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 0, 
    cmap = "rocket_r",
    cbar_ticks_size=10,
    font="Arial"
)

alignment = AlignRepresentations(
    config=config,
    representations_list=[Group1, Group2],
    main_results_dir="../results/",
    data_name="Simulation_2"
)

# RSA
alignment.show_sim_mat(
    visualization_config=vis_config, 
    show_distribution=False,
    fig_dir="../results/Simulation_2"
    )
alignment.RSA_get_corr()

# show embeddings
Group1.show_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_2", fig_name="Group1")
Group2.show_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_2", fig_name="Group2")

#np.save("../../data/simulation2_embedding1.npy", Group1.embedding)
#np.save("../../data/simulation2_embedding2.npy", Group2.embedding)
# %%
# GW
alignment.gw_alignment(
    compute_OT=False,
    delete_results=False,
    visualization_config=vis_config
    )

alignment.visualize_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_2")

alignment.calc_accuracy(top_k_list=[1, 3, 5], eval_type="ot_plan")
alignment.calc_accuracy(top_k_list=[1, 3, 5], eval_type="k_nearest")
# %%

### Pattern 3

#def rotate_point(point, angle_degrees, center=[0, 0]):
#    angle_radians = np.radians(angle_degrees)
#
#    cos_theta = np.cos(angle_radians)
#    sin_theta = np.sin(angle_radians)
#    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
#
#    translated_point = np.array(point) - np.array(center)
#
#    rotated_point = np.dot(translated_point, rotation_matrix)
#
#    result_point = rotated_point + np.array(center)
#
#    return result_point
#
#def generate_cardioid_points_with_start(N, t_start=0):
#    t_values = np.linspace(t_start, t_start + 2*np.pi, N, endpoint=False)
#    x_values = (1 - np.cos(t_values)) * np.cos(t_values)
#    y_values = (1 - np.cos(t_values)) * np.sin(t_values)
#    points = np.column_stack((x_values, y_values))
#    return points

#%%
N_points = 50
circle = create_circle(N_points, radius=10)
embedding_1 = rotate_points_around_center(circle, max_offset=0.35, start=0, seed=0)
embedding_2 = rotate_points_around_center(circle, max_offset=0.35, start=N_points//2, seed=0)
#cardioid_1 = generate_cardioid_points_with_start(N=N_points, t_start=0)
#cardioid_2 = generate_cardioid_points_with_start(N=N_points, t_start=np.pi)
#embedding_1 = add_random_offset(cardioid_1, max_offset=0.01, seed=0)
#embedding_2 = add_random_offset(cardioid_2, max_offset=0.01, seed=0)

#new_points = np.array([[10, 10], [-10, -10]])
#embedding_1 = np.append(embedding_1, new_points, axis=0)
#embedding_2 = np.append(embedding_2, new_points, axis=0)
#np.save("../../data/simulation3_embedding1.npy", embedding_1)
#np.save("../../data/simulation3_embedding2.npy", embedding_2)

Group1 = Representation(
    name="Group1",
    metric="euclidean",
    embedding=embedding_1)

Group2 = Representation(
    name="Group2",
    metric="euclidean",
    embedding=embedding_2
)

config = OptimizationConfig(
    eps_list=[0.1, 1],
    num_trial=50,
    db_params={"drivername": "sqlite"},
    n_iter=1,
)

alignment = AlignRepresentations(
    config=config,
    representations_list=[Group1, Group2],
    main_results_dir="../results/",
    data_name="Simulation_3"
)

vis_config = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 0, 
    cmap = "rocket_r",
    cbar_ticks_size=10,
    font="Arial"
)

# RSA
alignment.show_sim_mat(
    visualization_config=vis_config, 
    show_distribution=False,
    fig_dir="../results/Simulation_3"
    )
alignment.RSA_get_corr()

# show embeddings
Group1.show_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_3", fig_name="Group1")
Group2.show_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_3", fig_name="Group2")

# %%
# GW
alignment.gw_alignment(
    compute_OT=False,
    delete_results=False,
    visualization_config=vis_config
    )

alignment.visualize_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_3")

alignment.calc_accuracy(top_k_list=[1, 3, 5], eval_type="ot_plan")
alignment.calc_accuracy(top_k_list=[1, 3, 5], eval_type="k_nearest")

# %%



### Pattern 4
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

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
Group1 = Representation(
    name="1",
    metric="euclidean",
    embedding=embedding_1,
    )

Group2 = Representation(
    name="2",
    metric="euclidean",
    embedding=embedding_2,
    
)

config = OptimizationConfig(
    eps_list=[0.1, 1],
    num_trial=50,
    db_params={"drivername": "sqlite"},
    n_iter=1,
)

vis_config = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 0, 
    cmap = "rocket_r",
    cbar_ticks_size=10,
    font="Arial",
    color_labels=colors,
    color_label_width=3
)

vis_emb = VisualizationConfig(
    figsize=(8, 8), 
    legend_size=12,
    marker_size=60,
    color_labels=colors,
    fig_ext='svg',
    markers_list=['o', 'X']
)
vis_emb_2 = VisualizationConfig(
    figsize=(8, 8), 
    legend_size=12,
    marker_size=60,
    color_labels=colors,
    fig_ext='svg',
    markers_list=['X']
)

vis_log = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 0, 
    cmap = "viridis",
    cbar_ticks_size=15,
    font="Arial",
    xlabel_size=20,
    xticks_size=15,
    ylabel_size=20,
    yticks_size=15,
    cbar_label_size=15,
    plot_eps_log=True,
    fig_ext='svg'
)

alignment = AlignRepresentations(
    config=config,
    representations_list=[Group1, Group2],
    main_results_dir="../results/",
    data_name="Simulation_4"
)

#%%
# RSA
alignment.show_sim_mat(
    visualization_config=vis_config, 
    show_distribution=False,
    fig_dir="../results/Simulation_4"
    )
alignment.RSA_get_corr()

# show embeddings
Group1.show_embedding(dim=3, visualization_config=vis_emb, fig_dir="../results/Simulation_4", fig_name="Group1", legend=False)
Group2.show_embedding(dim=3, visualization_config=vis_emb_2, fig_dir="../results/Simulation_4", fig_name="Group2", legend=False)
# %%
# GW
alignment.gw_alignment(
    compute_OT=False,
    delete_results=False,
    visualization_config=vis_config
    )

alignment.show_optimization_log(
    fig_dir="../results/Simulation_4",
    visualization_config=vis_log
    )
alignment.visualize_embedding(dim=3, visualization_config=vis_emb, fig_dir="../results/Simulation_4")

alignment.calc_accuracy(top_k_list=[1, 3, 5], eval_type="ot_plan")
# %%


### Simulation for the colors data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# load colors
colors = np.load("../../data/color/new_color_order.npy")

#%%

def plot_circles_and_points(radius1, num_points1, radius2, num_points2, distance):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    
    # Circle 1
    theta1 = np.linspace(0, 2 * np.pi, num_points1, endpoint=False)
    x1 = radius1 * np.cos(theta1) - distance / 2
    y1 = radius1 * np.sin(theta1)
    #circle1 = plt.Circle((-distance / 2, 0), radius1, edgecolor='b', facecolor='none')
    circle_1 = np.column_stack((x1, y1))
    
    # Circle 2
    theta2 = np.linspace(0, 2 * np.pi, num_points2, endpoint=False)
    x2 = radius2 * np.cos(theta2) + distance / 2
    y2 = radius2 * np.sin(theta2)
    #circle2 = plt.Circle((distance / 2, 0), radius2, edgecolor='r', facecolor='none')
    circle_2 = np.column_stack((x2, y2))

    #ax.add_patch(circle1)
    #ax.add_patch(circle2)
    ax.plot(x1, y1, 'bo')  # Points for circle 1
    ax.plot(x2, y2, 'ro')  # Points for circle 2

    ax.set_xlim(-radius1 - distance, radius2 + distance)
    ax.set_ylim(-max(radius1, radius2) - 1, max(radius1, radius2) + 1)

    plt.grid(True)
    plt.show()
    
    return circle_1, circle_2

# Parameters
radius1 = 4
num_points1 = 60
radius2 = 3
num_points2 = 33
distance = 3.5

circle1, circle2 = plot_circles_and_points(radius1, num_points1, radius2, num_points2, distance)

# %%
# concatenate circles
embedding_1 = np.concatenate((circle1, circle2), axis=0)
# add noise
np.random.seed(0)
noise = np.random.normal(0, 0.3, embedding_1.shape)
embedding_1 = embedding_1 + noise

# shuffle the order of the points of the second circle and save the new embedding
np.random.seed(0)
np.random.shuffle(circle2)
embedding_2 = np.concatenate((circle1, circle2), axis=0)
# add the same noise
embedding_2 = embedding_2 + noise

# %%
Group1 = Representation(
    name="1",
    metric="euclidean",
    embedding=embedding_1,
    )

Group2 = Representation(
    name="2",
    metric="euclidean",
    embedding=embedding_2,
    
)

config = OptimizationConfig(
    eps_list=[0.1, 1],
    num_trial=10, #50
    db_params={"drivername": "sqlite"},
    n_iter=1,
)

vis_config = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 0, 
    cmap = "rocket_r",
    cbar_ticks_size=10,
    font="Arial",
    color_labels=colors,
    color_label_width=3
)

vis_emb = VisualizationConfig(
    figsize=(8, 8), 
    legend_size=12,
    marker_size=60,
    color_labels=colors,
    fig_ext='svg',
    markers_list=['o', 'X']
)
vis_emb_2 = VisualizationConfig(
    figsize=(8, 8), 
    legend_size=12,
    marker_size=60,
    color_labels=colors,
    fig_ext='svg',
    markers_list=['X']
)

vis_log = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 0, 
    cmap = "viridis",
    cbar_ticks_size=15,
    font="Arial",
    xlabel_size=20,
    xticks_size=15,
    ylabel_size=20,
    yticks_size=15,
    cbar_label_size=15,
    plot_eps_log=True,
    fig_ext='svg'
)

alignment = AlignRepresentations(
    config=config,
    representations_list=[Group1, Group2],
    main_results_dir="../results/",
    data_name="Simulation_colors"
)

#%%
# RSA
alignment.show_sim_mat(
    visualization_config=vis_config, 
    show_distribution=False,
    fig_dir="../results/Simulation_colors"
    )
alignment.RSA_get_corr()

# show embeddings
Group1.show_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_colors", fig_name="Group1", legend=False)
Group2.show_embedding(dim=2, visualization_config=vis_emb_2, fig_dir="../results/Simulation_colors", fig_name="Group2", legend=False)
# %%
# GW
alignment.gw_alignment(
    compute_OT=True,
    delete_results=False,
    visualization_config=vis_config
    )

alignment.show_optimization_log(
    fig_dir="../results/Simulation_colors",
    visualization_config=vis_log
    )
alignment.visualize_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_colors")

alignment.calc_accuracy(top_k_list=[1, 3, 5], eval_type="ot_plan")
# %%



### Simulation for the colors data number 2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load colors
colors = np.load("../../data/color/new_color_order.npy")

def plot_circle(radius, num_points):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    circle = np.column_stack((x, y))

    ax.plot(x, y, 'bo')  # Points for circle

    ax.set_xlim(-radius - 1, radius + 1)
    ax.set_ylim(-radius - 1, radius + 1)

    plt.grid(True)
    plt.show()
    
    return circle

# Parameters
radius = 4
num_points = 93

circle1 = plot_circle(radius, num_points)
circle2 = plot_circle(radius, num_points)

# add correlated noise
np.random.seed(0)
noise = np.random.normal(0, 0.4, circle1.shape)
circle1 = circle1 + noise*1
circle2 = circle2 + noise*1

# shuffle the order of the points of the second circle and save the new embedding
np.random.seed(0)
np.random.shuffle(circle2[-38:])

embedding_1 = circle1
embedding_2 = circle2
# %%
# %%
Group1 = Representation(
    name="1",
    metric="euclidean",
    embedding=embedding_1,
    )

Group2 = Representation(
    name="2",
    metric="euclidean",
    embedding=embedding_2,
    
)

config = OptimizationConfig(
    eps_list=[0.01, 0.1],
    num_trial=10, #50
    db_params={"drivername": "sqlite"},
    n_iter=1,
)

vis_config = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 0, 
    cmap = "rocket_r",
    cbar_ticks_size=10,
    font="Arial",
    color_labels=colors,
    color_label_width=3
)

vis_emb = VisualizationConfig(
    figsize=(8, 8), 
    legend_size=12,
    marker_size=60,
    color_labels=colors,
    fig_ext='svg',
    markers_list=['o', 'X']
)
vis_emb_2 = VisualizationConfig(
    figsize=(8, 8), 
    legend_size=12,
    marker_size=60,
    color_labels=colors,
    fig_ext='svg',
    markers_list=['X']
)

vis_log = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 0, 
    cmap = "viridis",
    cbar_ticks_size=15,
    font="Arial",
    xlabel_size=20,
    xticks_size=15,
    ylabel_size=20,
    yticks_size=15,
    cbar_label_size=15,
    plot_eps_log=True,
    fig_ext='svg'
)

alignment = AlignRepresentations(
    config=config,
    representations_list=[Group1, Group2],
    main_results_dir="../results/",
    data_name="Simulation_colors"
)

#%%
# RSA
alignment.show_sim_mat(
    visualization_config=vis_config, 
    show_distribution=False,
    fig_dir="../results/Simulation_colors"
    )
alignment.RSA_get_corr()

# show embeddings
Group1.show_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_colors", fig_name="Group1", legend=False)
Group2.show_embedding(dim=2, visualization_config=vis_emb_2, fig_dir="../results/Simulation_colors", fig_name="Group2", legend=False)
# %%
# GW
alignment.gw_alignment(
    compute_OT=True,
    delete_results=True,
    visualization_config=vis_config
    )

alignment.show_optimization_log(
    fig_dir="../results/Simulation_colors",
    visualization_config=vis_log
    )
alignment.visualize_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_colors")

alignment.calc_accuracy(top_k_list=[1, 3, 5], eval_type="ot_plan")
# %%

### Simulation for the colors data number 3
import numpy as np
import matplotlib.pyplot as plt

def generate_cluster_centers(n_clusters, size=1.0, seed=42):
    np.random.seed(seed)  # 乱数シードの設定（再現性のため）
    
    centers = []
    for _ in range(n_clusters):
        # クラスターの中心をランダムに決定
        center_x, center_y = np.random.normal((0, 0), size, 2)
        centers.append((center_x, center_y))
    
    return centers

def generate_clusters(n_points_per_cluster, centers, spread=1.0, seed=42):
    np.random.seed(seed)  # 乱数シードの設定（再現性のため）

    # set the emnpty 2d array
    clusters = np.empty((0, 2))
    for center in centers:
        # クラスターの中心からの距離が spread 以内の範囲にランダムに点を生成
        points_x = np.random.normal(center[0], spread, n_points_per_cluster)
        points_y = np.random.normal(center[1], spread, n_points_per_cluster)
        points = np.column_stack((points_x, points_y))
        # stack the points
        clusters = np.vstack((clusters, points))
    return clusters

def plot_clusters(clusters):
    plt.figure(figsize=(8, 6))
    for points_x, points_y in clusters:
        plt.scatter(points_x, points_y, alpha=0.6)
    plt.title("Randomly Generated Clusters")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()

# クラスターを生成しプロットする例
n_points_per_cluster = 5
n_clusters = 18
interpolation_points = 3
size = 20
spread = 10
centers = generate_cluster_centers(n_clusters, size=size, seed=2)
clusters1 = generate_clusters(n_points_per_cluster, centers, spread=spread, seed=42)
clusters2 = generate_clusters(n_points_per_cluster, centers, spread=spread, seed=43)
plot_clusters(clusters1)
plot_clusters(clusters2)

# convert to numpy array
# reshape (31, 2, 3) -> (93, 2)
embedding_1 = clusters1
embedding_2 = clusters2

if interpolation_points > 0:
    interp = np.random.uniform(-size, size, (interpolation_points, 2))
    embedding_1 = np.vstack((embedding_1, interp))
    embedding_2 = np.vstack((embedding_2, interp))
# %%
# load colors
colors = np.load("../../data/color/new_color_order.npy")

Group1 = Representation(
    name="1",
    metric="euclidean",
    embedding=embedding_1,
    )

Group2 = Representation(
    name="2",
    metric="euclidean",
    embedding=embedding_2,
    
)

config = OptimizationConfig(
    eps_list=[0.1, 10],
    num_trial=30, #50
    db_params={"drivername": "sqlite"},
    n_iter=1,
)

vis_config = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 0, 
    cmap = "rocket_r",
    cbar_ticks_size=10,
    font="Arial",
    color_labels=colors,
    color_label_width=3
)

vis_emb = VisualizationConfig(
    figsize=(8, 8), 
    legend_size=12,
    marker_size=60,
    color_labels=colors,
    fig_ext='svg',
    markers_list=['o', 'X']
)
vis_emb_2 = VisualizationConfig(
    figsize=(8, 8), 
    legend_size=12,
    marker_size=60,
    color_labels=colors,
    fig_ext='svg',
    markers_list=['X']
)

vis_log = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 0, 
    cmap = "viridis",
    cbar_ticks_size=15,
    font="Arial",
    xlabel_size=20,
    xticks_size=15,
    ylabel_size=20,
    yticks_size=15,
    cbar_label_size=15,
    plot_eps_log=True,
    fig_ext='svg'
)

alignment = AlignRepresentations(
    config=config,
    representations_list=[Group1, Group2],
    main_results_dir="../results/",
    data_name="Simulation_colors_cluster"
)

#%%
# RSA
alignment.show_sim_mat(
    visualization_config=vis_config, 
    show_distribution=False,
    fig_dir="../results/Simulation_colors_cluster"
    )
alignment.RSA_get_corr()

# show embeddings
Group1.show_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_colors_cluster", fig_name="Group1", legend=False)
Group2.show_embedding(dim=2, visualization_config=vis_emb_2, fig_dir="../results/Simulation_colors_cluster", fig_name="Group2", legend=False)
# %%
# GW
alignment.gw_alignment(
    compute_OT=True,
    delete_results=False,
    visualization_config=vis_config
    )

alignment.show_optimization_log(
    fig_dir="../results/Simulation_colors_cluster",
    visualization_config=vis_log
    )
alignment.visualize_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_colors_cluster")

alignment.calc_accuracy(top_k_list=[1, 3, 5], eval_type="ot_plan")
# %%