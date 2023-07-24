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

def shuffle_matrix(matrix, N_divide):
    lower_triangle_elements = _extract_lower_triangle(matrix)
    for i in range(N_divide):
        lower_triangle_elements = _rearrange_and_shuffle(lower_triangle_elements, N=N_divide, m=i)
    shuffled_matrix = _create_symmetric_matrix(lower_triangle_elements, size=matrix.shape[0])
    return shuffled_matrix
    
def _extract_lower_triangle(matrix):
    lower_triangle_idx = np.tril_indices(matrix.shape[0], k=-1)
    return matrix[lower_triangle_idx].flatten()

def _create_symmetric_matrix(lower_triangle_array, size):
    lower_triangle = np.zeros((size, size))
    lower_triangle[np.tril_indices(size, k=-1)] = lower_triangle_array
    symmetric_matrix = lower_triangle + lower_triangle.T
    return symmetric_matrix


def _rearrange_and_shuffle(original_array, N, m, seed=0):
    sorted_array = np.sort(original_array)[::-1]

    array_length = len(sorted_array)
    segment_length = array_length // N

    start_index = m * segment_length
    end_index = (m + 1) * segment_length if m < N - 1 else array_length
    extracted_segment = sorted_array[start_index:end_index]

    np.random.seed(seed)
    np.random.shuffle(extracted_segment)

    shuffled_array = np.copy(sorted_array)
    shuffled_array[start_index:end_index] = extracted_segment

    original_order_indices = np.argsort(original_array)[::-1]
    final_array = shuffled_array[original_order_indices]

    return final_array

#%%
N_points = 50
# get matrices
circle = create_circle(N=N_points)
circle = rotate_points_around_center(circle, max_offset=0.8)
embedding_1 = add_random_offset(circle, max_offset=3)
embedding_2 = add_independent_noise(embedding_1, max_noise=3)
#sim_mat_1 = distance.cdist(embedding, embedding, "euclidean")
#sim_mat_2 = shuffle_matrix(sim_mat_1, N_divide=500)

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
    data_name="Simulation_1"
)

vis_config = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 15, 
    cmap = "rocket",
    cbar_ticks_size=20,
)

vis_emb = VisualizationConfig(
    figsize=(8, 8), 
    legend_size=12,
    marker_size=60
)

alignment = AlignRepresentations(
    config=config,
    representations_list=[Group1, Group2]
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
    results_dir="../results",
    compute_OT=False,
    delete_results=False,
    visualization_config=vis_config
    )

alignment.visualize_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_1")

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
    data_name="Simulation_2"
)

vis_config = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 15, 
    cmap = "rocket",
    cbar_ticks_size=20,
)

alignment = AlignRepresentations(
    config=config,
    representations_list=[Group1, Group2]
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
# %%
# GW
alignment.gw_alignment(
    results_dir="../results",
    compute_OT=False,
    delete_results=False,
    visualization_config=vis_config
    )

alignment.visualize_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_2")
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
    data_name="Simulation_3"
)

alignment = AlignRepresentations(
    config=config,
    representations_list=[Group1, Group2]
)

vis_config = VisualizationConfig(
    figsize=(8, 6), 
    title_size = 15, 
    cmap = "rocket",
    cbar_ticks_size=20,
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
    results_dir="../results",
    compute_OT=False,
    delete_results=False,
    visualization_config=vis_config
    )

alignment.visualize_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_3")

# %%
