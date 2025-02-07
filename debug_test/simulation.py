#%%
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import optuna
import glob
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from src.align_representations import Representation, AlignRepresentations, OptimizationConfig
import pandas as pd
import copy
import scipy as sp
#%%
def add_independent_noise_to_all_dimensions(points, noise_deg=0.0001, except_point_index=None):
    """
    Adds independent Gaussian noise to all dimensions of all points.
    """
    noise = np.random.normal(loc=0.0, scale=noise_deg, size=points.shape)
    
    if except_point_index is not None:
        if isinstance(except_point_index, int):
            noise[except_point_index] = 0
        elif isinstance(except_point_index, list):
            for index in except_point_index:
                noise[index] = 0
        else:
            raise ValueError("except_point_index must be either int or list.")
    
    return points + noise

#%%
def add_noise_to_one_point(points, noise_deg=0.0001, point_index=0, seed=None):
    """
    Adds Gaussian noise to all dimensions of a single point in the point cloud.
    This is a sanity check to observe the effect of noise on a single point.
    """
    if seed is not None:
        np.random.seed(seed)
    noise = np.zeros_like(points)
    
    if isinstance(point_index, int):
        noise[point_index, :] = np.random.normal(loc=0.0, scale=noise_deg, size=points.shape[1])
    elif isinstance(point_index, list):
        for index in point_index:
            noise[index, :] = np.random.normal(loc=0.0, scale=noise_deg, size=points.shape[1])
            
    return points + noise

#%%
def add_noise_to_one_dimension(points, noise_deg=0.0001, dimension=0, seed=None):
    """
    Adds Gaussian noise to only one dimension of each point in the point cloud.
    This is a sanity check to observe the effect of noise on a single dimension.
    """
    if seed is not None:
        np.random.seed(seed)
    noise = np.zeros_like(points)
    noise[:, dimension] = np.random.normal(loc=0.0, scale=noise_deg, size=points.shape[0])
    return points + noise

#%%
def create_circle_data(n_points):
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = np.cos(t)
    y = np.sin(t)
    one_data = np.vstack((x, y)).T
    return one_data

#%%
def detect_diagonal_direction(matrix):
    """
    行列が右斜め下向きまたは左斜め下向きに値が集中しているかを判定する関数。
    対角線およびその近傍に値が集中している場合も考慮する。
    
    Args:
        matrix (np.ndarray): 判定対象の行列（正方行列を想定）。
    Returns:
        str: 判定結果を表す文字列。
             'right': 右斜め下向き。
             'left': 左斜め下向き。
             'none': 特定の方向に集中していない。
    """
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("この関数は正方行列にのみ対応しています。")
    
    def _sum_diagonal(mat):
        sum_diag = []
        k_range = range(-len(mat)+1, len(mat))
        
        for k in k_range:
            element_sum = np.sum(np.diag(mat, k))
            sum_diag.append(element_sum)
        
        unique_score = np.unique(sum_diag)
        score = sum(unique_score[-2:])
        
        return score

    # 主対角線およびその近傍の値を集計
    right_diagonal_score = _sum_diagonal(matrix)
    
    # 左斜め下向きの場合の行列を作成（左右反転）
    flipped_matrix = np.fliplr(matrix)
    left_diagonal_score = _sum_diagonal(flipped_matrix)
    
    # print("right_diagonal_score", right_diagonal_score)
    # print("left_diagonal_score", left_diagonal_score)
    
    # 0行目の最大値のインデックスを取得
    max_idx = np.argmax(matrix[0])
    
    if np.isclose(right_diagonal_score, 1.0, atol=1e-1):
        return f"R{max_idx}"
    
    elif np.isclose(left_diagonal_score, 1.0, atol=1e-1):
        return f"L{max_idx}"

    else:
        return "none"

# %%
def get_result_from_database(data_name, main_results_dir):
    df_path = glob.glob(f"{main_results_dir}/{data_name}/*/*.db")[0]
    study = optuna.load_study(study_name = os.path.basename(df_path).split(".db")[0], storage = f"sqlite:///{df_path}")
    return study

# %%
def get_ot(data_name, main_results_dir, min_index):
    npy_path = glob.glob(f"{main_results_dir}/{data_name}/*/data/gw_{min_index}.npy")[0]
    ot = np.load(npy_path)
    return ot

# %%
class CircleDataExperiment:
    def __init__(
        self, 
        n_points, 
        common_noise_deg=1e-6, 
        common_noise_list :list=[], 
        independent_noise_deg=0,
        dimension_noise_deg=0,
        rotation_index=0,
    ):
        self.n_points = n_points
        self.shape1 = create_circle_data(n_points)
        self.shape2 = create_circle_data(n_points)
        
        self.common_noise_deg = common_noise_deg
        self.common_noise_list  = common_noise_list 
        self.rotation_index = rotation_index
        
        if len(common_noise_list) == 0:
            self.data_name = f"circle_{n_points}points"
        
        else:
            self.shape1 = add_noise_to_one_point(self.shape1, common_noise_deg, point_index=common_noise_list, seed=0)
            self.data_name = f"circle_{n_points}points_common_noise({len(common_noise_list)}_deg:{common_noise_deg:.2e})"
        
        self.shape2 = copy.deepcopy(self.shape1)
        
        if rotation_index > 0:
            self.shape2 = np.roll(self.shape2, rotation_index, axis=0)
            self.data_name += f"_rotation_index:{rotation_index}"

        if independent_noise_deg > 0:
            if common_noise_list == []:
                self.shape2 = add_independent_noise_to_all_dimensions(self.shape2, noise_deg=independent_noise_deg)
            else:
                self.shape2 = add_independent_noise_to_all_dimensions(self.shape2, noise_deg=independent_noise_deg, except_point_index=common_noise_list)
            
            self.data_name += f"_independent_noise_deg:{independent_noise_deg:.2e}"
        
        if dimension_noise_deg > 0:
            self.shape2 = add_noise_to_one_dimension(self.shape2, noise_deg=dimension_noise_deg, dimension=0, seed=0)
            self.data_name += f"_dimension_noise_deg:{dimension_noise_deg:.2e}"
                
    def run_experiment(self, sampler_init):
        # define the main results directory and the representation names
        initialization, sampler = sampler_init.split("_")
        main_results_dir = f"../results/circle/{sampler_init}"
        
        # Create representations
        rep1 = Representation(name="shape1", metric="euclidean", embedding=self.shape1)
        rep2 = Representation(name="shape2", metric="euclidean", embedding=self.shape2)

        config = OptimizationConfig(
            eps_list=eps_list,
            num_trial=num_trial,
            db_params={"drivername": "sqlite"},
            sinkhorn_method="sinkhorn",
            n_iter=1,
            max_iter = 200,
            to_types="numpy",  
            device="cpu",
            data_type="double", 
            sampler_name=sampler,
            init_mat_plan=initialization,
            show_progress_bar=False,
        )

        alignment = AlignRepresentations(
            config=config,
            representations_list=[rep1, rep2],
            main_results_dir=main_results_dir,
            data_name=self.data_name,
        )
        
        # GW
        data_path = f"{main_results_dir}/{alignment.data_name}_{rep1.name}_vs_{rep2.name}/*/data/*.npy"
        npy_list = glob.glob(data_path)
        
        if len(npy_list) >= num_trial:
            print(f"{alignment.data_name} was already computed.")
            
        else:
            alignment.gw_alignment(
                compute_OT=True,
                delete_results=False,
                show_log=False,
                fix_random_init_seed=False,
                sampler_seed=42,
                delete_confirmation=False,
            )
    
    def main_test(self):
        pool = ProcessPoolExecutor(len(sampler_initilizations))
        
        with pool:
            processes = []
            for _, sampler_init in enumerate(sampler_initilizations):                    
                future = pool.submit(
                    self.run_experiment,
                    sampler_init=sampler_init,
                )

                processes.append(future)

            for future in as_completed(processes):
                future.result()
    
    def visualize_raw_data(self, test=False):
        plt.style.use("default")
        # plt.rcParams["grid.color"] = "black"
        plt.rcParams['font.family'] = "Arial"
        
        # Visualize the shapes
        fig = plt.figure(figsize=(10, 6))

        # Shape 1
        ax1 = fig.add_subplot(121)
        ax1.axis("equal")
        ax1.scatter(self.shape1[:, 0], self.shape1[:, 1], c="C0", label='Shape 1', s=60)
        ax1.set_title("Shape 1", fontsize=30)
        for i in range(self.n_points):
            ax1.text(self.shape1[i, 0], self.shape1[i, 1] + 2e-2, str(i), ha='center', fontsize=20, color="black")
        # ax1.set_xlabel("X", fontsize=20)
        # ax1.set_ylabel("Y", fontsize=20)

        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        # ax1.grid()
        ax1.set_axisbelow(True)

        # Shape 2
        ax2 = fig.add_subplot(122)
        ax2.axis("equal")
        ax2.scatter(self.shape2[:, 0], self.shape2[:, 1], c="C1", label=f'Shape 2', s=60)
        for i in range(self.n_points):
            if i == 5:
                ax2.text(self.shape2[i, 0] + 2e-2, self.shape2[i, 1], str(i), ha='left', fontsize=20, color="black")
            elif i == 6:
                ax2.text(self.shape2[i, 0] - 2e-2, self.shape2[i, 1] -4e-2, str(i), ha='right', fontsize=20, color="black")
            else:
                ax2.text(self.shape2[i, 0], self.shape2[i, 1] + 2e-2, str(i), ha='center', fontsize=20, color="black")
        ax2.set_title("Shape 2", fontsize=30)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        # ax2.grid()
        ax2.set_axisbelow(True)

        plt.tight_layout()
        
        raw_save_fig_dir = f"../results/circle/fig/raw"
        os.makedirs(raw_save_fig_dir, exist_ok=True)
        if test:
            plt.show()
        else:
            plt.savefig(f"{raw_save_fig_dir}/{self.data_name}.svg")
        plt.close()
        
        mat1 = sp.spatial.distance.cdist(self.shape1, self.shape1)
        mat2 = sp.spatial.distance.cdist(self.shape2, self.shape2)
        
        cmap = "rocket"
        plt.figure(figsize=(10, 6))
        plt.suptitle(f"RDM, rotation index:{self.rotation_index}")
        plt.subplot(121)
        plt.imshow(mat1, cmap=cmap)
        plt.colorbar(orientation="horizontal")
        plt.subplot(122)
        plt.imshow(mat2, cmap=cmap)
        plt.colorbar(orientation="horizontal")
        raw_save_fig_dir = f"../results/circle/fig/rdm"
        os.makedirs(raw_save_fig_dir, exist_ok=True)
        plt.tight_layout()
        if test:
            plt.show()
        else:
            plt.savefig(f"{raw_save_fig_dir}/{self.data_name}.png")
        plt.close()

def main_test_with_independent_noise(independent_noise_deg, max_workers=3):
    main_pool = ProcessPoolExecutor(max_workers=max_workers)
    
    with main_pool:
        main_processes = []
        for _, common_noise_deg in enumerate(common_noise_deg_list):
            
            if common_noise_deg == 0:common_noise_list = []
            else:common_noise_list = main_common_noise_list  
            
            experiment = CircleDataExperiment(
                n_points, 
                common_noise_deg=common_noise_deg, 
                common_noise_list=common_noise_list,
                independent_noise_deg=independent_noise_deg,
                dimension_noise_deg=0.1,
                rotation_index=main_rot_index,
            )
            experiment.visualize_raw_data()

            future = main_pool.submit(experiment.main_test)

            main_processes.append(future)

        for future in tqdm(as_completed(main_processes), total=len(common_noise_deg_list), desc=f"independent noise deg:{independent_noise_deg:.2e}", leave=True):
            future.result()


# %%
main_compute = False
main_visualize = True

# GWOT parameters
eps_list = [1e-2, 1e-0]
num_trial = 100

# optuna.logging.set_verbosity(optuna.logging.WARNING)

# Parameters
n_points = 20 # Total number of points
common_noise_deg_list = [1e-1]
independent_noise_deg_list = [0]
sampler_initilizations = ["random_tpe", "random_grid", "uniform_grid"]

graph_name_list = ["Random + TPE", "Random + Grid Search", "Uniform + Grid Search"]

#%%
main_common_noise_list = [0, 1, 4, 6, 8, 11, 14, 18]
main_rot_index = 0

#%%
test = True
if test:
    experiment = CircleDataExperiment(
        n_points, 
        common_noise_deg=0.1, 
        common_noise_list=[0, 1, 4, 6, 8, 11, 14, 18], 
        independent_noise_deg=0,
        dimension_noise_deg=0.1,
        rotation_index=0,
    )
    experiment.visualize_raw_data(test=False)
    
#%%
if main_compute:
    for independent_noise_deg in independent_noise_deg_list:
        main_test_with_independent_noise(independent_noise_deg)

#%%
# plot the results
if main_visualize:
    os.makedirs("../results/circle/fig/main_fig/log/opt_ot", exist_ok=True)
    os.makedirs("../results/circle/fig/main_fig/log/opt_log", exist_ok=True)
    
    #%%
    for independent_noise_deg in tqdm(independent_noise_deg_list):
        for common_noise_deg in common_noise_deg_list:
            if common_noise_deg == 0:common_noise_list = 0
            else:common_noise_list = main_common_noise_list  
            
            experiment = CircleDataExperiment(
                n_points, 
                common_noise_deg=common_noise_deg, 
                common_noise_list=common_noise_list,
                independent_noise_deg=independent_noise_deg,
                dimension_noise_deg=0.1,
                rotation_index=main_rot_index,
            )
            data_name = f"{experiment.data_name}_shape1_vs_shape2"
            
            plt.figure(figsize=(10, 10))
            
            min_values = []
            for idx_, sampler_init in enumerate(sampler_initilizations):
                main_results_dir = f"../results/circle/{sampler_init}"
                study = get_result_from_database(data_name, main_results_dir)
                df = study.trials_dataframe()
                
                plt.subplot(3, 1, sampler_initilizations.index(sampler_init) + 1)
                plt.scatter(df["params_eps"], df["value"], c = 100 * df["user_attrs_best_acc"], s=20)

                plt.xlabel("eps")
                plt.ylabel("GWD")
                plt.title(f"{graph_name_list[idx_]}")
                cbar = plt.colorbar(label="Top-1 Accuracy")
                cbar.mappable.set_clim([0, 100])
                plt.xscale("log")
                plt.yscale("log")
                plt.ylim(7e-3, 1e-0)
                plt.grid(True)
                
                min_value = df.index[df["value"] == df["value"].min()]
                min_values.append(min_value[0])
            
            plt.tight_layout()
            
            plt.savefig(f"../results/circle/fig/main_fig/log/opt_log/comparison_log_{data_name}.svg")
            plt.close()
            

            plt.figure(figsize=(10, 4))
            for i, sampler_init in enumerate(sampler_initilizations):
                main_results_dir = f"../results/circle/{sampler_init}"
                ot = get_ot(data_name, main_results_dir, min_values[i])
                
                plt.subplot(1, 3, i + 1)
                plt.imshow(ot, cmap="rocket_r")

                plt.xlabel(f"{n_points} points")
                plt.ylabel(f"{n_points} points")
                plt.title(f"OT {sampler_init}")
                plt.colorbar(shrink=0.6)
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"../results/circle/fig/main_fig/log/opt_ot/comparison_ot_{data_name}.svg")
            plt.close()
            
            
            save_fig_path = f"../results/circle/fig/main_fig/"
            os.makedirs(save_fig_path, exist_ok=True)
            
            plt.style.use("default")
            plt.rcParams['font.family'] = "Arial"
            
            fig = plt.figure(figsize=(10, 10))  
            outer = gridspec.GridSpec(3, 1, wspace=0.2, hspace=0.1)
            # plt.suptitle(f"OT {sampler_init} (ascending sorted by GWD)", size=20, y=0.99)
            adjust_list = [0.89, 0.62, 0.36]
            
            num_ot = 5
            
            for i, sampler_init in enumerate(sampler_initilizations[:]):
                main_results_dir = f"../results/circle/{sampler_init}"
                
                study = get_result_from_database(data_name, main_results_dir)
                df = study.trials_dataframe()
                
                inner = gridspec.GridSpecFromSubplotSpec(1, num_ot, subplot_spec=outer[i], wspace=0.2)

                ax = []
                
                if i == 0:
                    ot_list = [0,1,2,5,6]
                elif i == 1:
                    ot_list = range(num_ot)
                else:
                    ot_list = range(num_ot)
                
                
                for _, idx in enumerate(df.sort_values(by="value").index[ot_list]):
                    ax.append(fig.add_subplot(inner[_]))
                    
                    # ヒートマップのデータを取得
                    ot = get_ot(data_name, main_results_dir, idx)
                    ax[-1].imshow(ot, cmap="rocket_r")
                    
                    # res = detect_diagonal_direction(ot)
                    
                    # GWDの値を取得してタイトルに表示
                    gwd = df.loc[idx, "value"]
                    ax[-1].set_title(f"GWD:{gwd:.2e}", fontsize=16)
                    ax[-1].axis('off')  # 軸を非表示にする

                # 各段のタイトルを設定（中央寄せ）
                fig.text(0.5, adjust_list[i], graph_name_list[i], ha='center', fontsize=23)

            # 全体のタイトルを追加
            # fig.suptitle("bottom 10 OT (ascending sorted by GWD)", fontsize=20, y=0.98)

            # レイアウト調整（余白を削減）
            # plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.1)
                
            plt.tight_layout()
            # plt.show()}
            plt.savefig(f"{save_fig_path}/heatmap_ot.svg")
            plt.close() 


# %%

        