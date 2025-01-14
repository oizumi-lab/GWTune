#%%
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import optuna
import glob
from tqdm import tqdm

from src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig

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
# GWOT parameters
eps_list = [1e-2, 1]
num_trial = 100

compute_OT = True
delete_results = True

# optuna.logging.set_verbosity(optuna.logging.WARNING)
    
# Parameters for starfish generation
n_points = 50  # Total number of points
sym_deg_list = np.linspace(0, 0.5, 20)
noise_deg_list = np.linspace(0, 0.1, 5)
sampler_initilizations = ["random_tpe", "random_grid", "uniform_grid"]

#%%
for sym_deg in sym_deg_list:
    for noise_deg in noise_deg_list:
        # Generate starfish shapes
        shape1 = create_starfish_3d_fixed_width_clean(num_arms=5, sym_deg=sym_deg, n_points=n_points)
        shape2_one_noise = add_noise_to_one_dimension(shape1, noise_deg=noise_deg, dimension=0)


        # Visualize the shapes
        fig = plt.figure(figsize=(15, 5))

        # Shape 1
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(shape1[:, 0], shape1[:, 1], shape1[:, 2], c='b', label='Shape 1')
        ax1.set_title("Shape 1 (Original Starfish)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.legend()

        # Shape 2 with noise in one dimension
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(shape2_one_noise[:, 0], shape2_one_noise[:, 1], shape2_one_noise[:, 2], c='r', label='Shape 2 (One Dimension Noise)')
        ax2.set_title("Shape 2 (One Dimension Noise)")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.legend()


        plt.tight_layout()
        plt.show()

        for sampler_init in tqdm(sampler_initilizations):
            if "random" in sampler_init:
                initialization = "random"
            elif "uniform" in sampler_init:
                initialization = "uniform"

            if "tpe" in sampler_init:
                sampler = "tpe"
            elif "grid" in sampler_init:
                sampler = "grid"
            
            main_results_dir = f"../results/simulation_starfish/{sampler_init}"
                
            # Create representations
            rep1 = Representation(
                name="1",
                metric="euclidean",
                embedding=shape1)

            rep2 = Representation(
                name="2",
                metric="euclidean",
                embedding=shape2_one_noise)

            config = OptimizationConfig(
                                eps_list=eps_list,
                                num_trial=num_trial,
                                db_params={"drivername": "sqlite"},
                                n_iter=1,
                                to_types="numpy",  # user can choose "numpy" or "torch". please set "torch" if one wants to use GPU.
                                device="cpu",  # "cuda" or "cpu"; for numpy, only "cpu" can be used.
                                sampler_name=sampler,
                                init_mat_plan=initialization,
                                show_progress_bar=False,
                            )

            vis_config = VisualizationConfig(
                                figsize=(8, 6), 
                                title_size = 0, 
                                cmap = "rocket_r",
                                cbar_ticks_size=10,
                                font="Arial",
                                # color_labels=colors,
                                color_label_width=3,
                                xlabel=f"{n_points} items",
                                ylabel=f"{n_points} items",
                                xlabel_size=40,
                                ylabel_size=40,
                            )

            vis_config_ot = VisualizationConfig(
                                figsize=(8, 6), 
                                title_size = 0, 
                                cmap = "rocket_r",
                                cbar_ticks_size=10,
                                font="Arial",
                                # color_labels=colors,
                                color_label_width=3,
                                xlabel=f"{n_points} items of X",
                                ylabel=f"{n_points} items of Y",
                                xlabel_size=40,
                                ylabel_size=40,
                            )

            vis_log = VisualizationConfig(
                                figsize=(8, 6), 
                                title_size = 0, 
                                cmap = "viridis",
                                cbar_ticks_size=15,
                                # font="Arial",
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
                                representations_list=[rep1, rep2],
                                main_results_dir=main_results_dir,
                                data_name=f"Starfish_{n_points}_points_sym{sym_deg}_noise{noise_deg}",
                            )

            # RSA
            fig_dir = f"../results/figs/simulation_Starfish"
            os.makedirs(fig_dir, exist_ok=True)
            alignment.show_sim_mat(
                visualization_config=vis_config, 
                show_distribution=False,
                fig_dir=fig_dir
                )
            alignment.RSA_get_corr(metric="pearson")

            # GW
            alignment.gw_alignment(
                compute_OT=compute_OT,
                delete_results=delete_results,
                visualization_config=vis_config_ot,
                fig_dir=fig_dir,
                delete_confirmation=False
                )

            alignment.show_optimization_log(
                fig_dir=fig_dir,
                visualization_config=vis_log
                )
            #alignment.visualize_embedding(dim=2, visualization_config=vis_emb, fig_dir=fig_dir)

            df = alignment.calc_accuracy(top_k_list=[1, 3, 5], eval_type="ot_plan", return_dataframe=True)
# %%

# plot the results

def get_min_values(df):
    min_values = []
    current_min = df['value'][0]
    for i in range(len(df)):
        value = df['value'][i]

        if value < current_min:
            current_min = value
            min_values.append(current_min)
        else:
            min_values.append(current_min)
    
    return min_values


min_values = []
min_indices = []
for sampler_init in sampler_initilizations:
    for sym_deg in sym_deg_list:
        for noise_deg in noise_deg_list:
            main_results_dir = f"../results/simulation_starfish/{sampler_init}"
            data_name = f"Starfish_{n_points}_points_sym{sym_deg}_noise{noise_deg}_1_vs_2"
            
            df_path = glob.glob(f"{main_results_dir}/{data_name}/*/*.db")[0]
            
            df = optuna.load_study(study_name = os.path.basename(df_path).split(".db")[0], storage = f"sqlite:///{df_path}").trials_dataframe()
            
            min_gwds = get_min_values(df)
            
            # get the min value in the min_gwds
            min_value = min(min_gwds)
            min_values.append(min_value)
            
            # get the index of when the min value is reached
            min_index = min_gwds.index(min_value)
            min_indices.append(min_index)

# plot min_values and min_indices in 2d heatmap
min_values = np.array(min_values).reshape((len(sampler_initilizations), len(sym_deg_list), len(noise_deg_list)))
min_indices = np.array(min_indices).reshape((len(sampler_initilizations), len(sym_deg_list), len(noise_deg_list)))

#%%
for sampler_init in sampler_initilizations:
    plt.figure(figsize=(15, 5))
    plt.imshow(min_values[sampler_initilizations.index(sampler_init), :, :], cmap='viridis')
    plt.colorbar()
    plt.yticks(ticks=np.arange(len(sym_deg_list)), labels=sym_deg_list)
    plt.ylabel("Symmetry Degree")
    plt.xticks(ticks=np.arange(len(noise_deg_list)), labels=noise_deg_list, rotation=45)
    plt.xlabel("Noise Degree")
    plt.title(f"Min gwds for {sampler_init}")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15, 5))
    plt.imshow(min_indices[sampler_initilizations.index(sampler_init), :, :], cmap='viridis')
    plt.colorbar()
    plt.yticks(ticks=np.arange(len(sym_deg_list)), labels=sym_deg_list)
    plt.ylabel("Symmetry Degree")
    plt.xticks(ticks=np.arange(len(noise_deg_list)), labels=noise_deg_list, rotation=45)
    plt.xlabel("Noise Degree")
    plt.title(f"Min indices for {sampler_init}")
    plt.tight_layout()
    plt.show()


# %%
# plot the min_gwds for the fixed noise degree
noise_deg = noise_deg_list[2]
plt.figure(figsize=(15, 5))
for sampler_init in sampler_initilizations:
    min_values = []
    for sym_deg in sym_deg_list:
        main_results_dir = f"../results/simulation_starfish/{sampler_init}"
        data_name = f"Starfish_{n_points}_points_sym{sym_deg}_noise{noise_deg}_1_vs_2"
        
        df_path = glob.glob(f"{main_results_dir}/{data_name}/*/*.db")[0]
        
        df = optuna.load_study(study_name = os.path.basename(df_path).split(".db")[0], storage = f"sqlite:///{df_path}").trials_dataframe()
        
        min_gwds = get_min_values(df)
        
        # get the min value in the min_gwds
        min_value = min(min_gwds)
        min_values.append(min_value)
        
    plt.plot(sym_deg_list, min_values, label=sampler_init)
plt.xlabel("Symmetry Degree")
plt.ylabel("Min GWD")
plt.title(f"Min GWDs for {sampler_init} with noise degree {noise_deg}")
plt.tight_layout()
plt.legend()
plt.show()
# %%
