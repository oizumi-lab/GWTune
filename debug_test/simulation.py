#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def create_starfish_3d_fixed_width_clean(num_arms=5, sym_deg=0.0, n_points=100, arm_length=1.0, arm_width=0.2):
    """
    Generates a starfish-like shape with fixed arm width and slight length perturbation.
    """
    points = []
    points_per_arm = n_points // num_arms
    remainder = n_points % num_arms

    for i in range(num_arms):
        theta = 2.0 * np.pi * i / num_arms
        rand_factor_len = np.random.uniform(-sym_deg, sym_deg)

        arm_len_i = max(0.1, arm_length + rand_factor_len)

        root = np.array([0, 0, 0])
        tip_left = np.array([arm_len_i, -arm_width / 2.0, 0])
        tip_right = np.array([arm_len_i, arm_width / 2.0, 0])

        rot_mat = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])

        root = rot_mat @ root
        tip_left = rot_mat @ tip_left
        tip_right = rot_mat @ tip_right

        num_samples = points_per_arm + (1 if i < remainder else 0)
        for _ in range(num_samples):
            u, v = np.random.rand(), np.random.rand()
            if u + v > 1.0:
                u, v = 1.0 - u, 1.0 - v
            p = root + u * (tip_left - root) + v * (tip_right - root)
            points.append(p)

    return np.array(points)

def add_independent_noise_to_all_dimensions(points, noise_deg=0.0001):
    """
    Adds independent Gaussian noise to all dimensions of all points.
    """
    noise = np.random.normal(loc=0.0, scale=noise_deg, size=points.shape)
    return points + noise

def add_noise_to_one_dimension(points, noise_deg=0.0001, dimension=0):
    """
    Adds Gaussian noise to only one dimension of each point in the point cloud.
    This is a sanity check to observe the effect of noise on a single dimension.
    """
    noise = np.zeros_like(points)
    noise[:, dimension] = np.random.normal(loc=0.0, scale=noise_deg, size=points.shape[0])
    return points + noise

#%%
# Parameters for starfish generation
sym_deg = 0.02  # Very small symmetry perturbation
n_points = 100  # Total number of points
noise_deg_all = 0.0001  # Very small noise for all dimensions
noise_deg_one = 0.0001  # Very small noise for one dimension

# Generate starfish shapes
shape1 = create_starfish_3d_fixed_width_clean(num_arms=5, sym_deg=sym_deg, n_points=n_points)
shape2_all_noise = add_independent_noise_to_all_dimensions(shape1, noise_deg=noise_deg_all)
shape2_one_noise = add_noise_to_one_dimension(shape1, noise_deg=noise_deg_one, dimension=0)

#%%
# Visualize the shapes
fig = plt.figure(figsize=(15, 5))

# Shape 1
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(shape1[:, 0], shape1[:, 1], shape1[:, 2], c='b', label='Shape 1')
ax1.set_title("Shape 1 (Original Starfish)")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.legend()

# Shape 2 with noise in all dimensions
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(shape2_all_noise[:, 0], shape2_all_noise[:, 1], shape2_all_noise[:, 2], c='r', label="Shape 2 (All Dimensions)")
ax2.set_title("Shape 2 (Noise in All Dimensions)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.legend()

# Shape 2 with noise in one dimension
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(shape2_one_noise[:, 0], shape2_one_noise[:, 1], shape2_one_noise[:, 2], c='g', label="Shape 2 (X Dimension Only)")
ax3.set_title("Shape 2 (Noise in One Dimension)")
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_zlabel("Z")
ax3.legend()

plt.tight_layout()
plt.show()

#%%