#%%
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import torch
from scipy.spatial import distance
from scipy.stats import spearmanr, pearsonr
from sklearn import manifold
import ot
import sys
import os
from typing import List

from utils.utils_functions import get_category_idx, sort_matrix_with_categories
from utils import visualize_functions, backend, init_matrix, gw_optimizer
from gw_alignment import GW_Alignment
from histogram_matching import SimpleHistogramMatching


class Optimization_Config:
    def __init__(
        self,
        data_name="THINGS",
        delete_study=False,
        device="cpu",
        to_types="numpy",
        n_jobs=1,
        init_plans_list=["random"],
        num_trial=4,
        n_iter=1,
        max_iter=200,
        sampler_name="tpe",
        eps_list=[1, 10],
        eps_log=True,
        pruner_name="hyperband",
        pruner_params={"n_startup_trials": 1, "n_warmup_steps": 2, "min_resource": 2, "reduction_factor": 3},
    ) -> None:

        self.data_name = data_name
        self.delete_study = delete_study
        self.device = device
        self.to_types = to_types
        self.n_jobs = n_jobs
        self.init_plans_list = init_plans_list
        self.num_trial = num_trial
        self.n_iter = n_iter
        self.max_iter = max_iter
        self.sampler_name = sampler_name
        self.eps_list = eps_list
        self.eps_log = eps_log
        self.pruner_name = pruner_name
        self.pruner_params = pruner_params
        
class Visualization_Config():
    def __init__(self, 
                 figsize = (8, 6),
                 cbar_size = 1.0,
                 cbar_ticks_size = 5,
                 ticks_size = 5,
                 xticks_rotation = 90,
                 yticks_rotation = 0,
                 title_size = 20,
                 legend_size = 5, 
                 xlabel = None,
                 xlabel_size = 15,
                 ylabel = None,
                 ylabel_size = 15,
                 zlabel = None,
                 zlabel_size = 15, 
                 color_labels = None, 
                 color_hue = None, 
                 markers_list = None, 
                 marker_size = 30
                 ) -> None:
        
        self.visualization_params = {
            'figsize': figsize,
            'cbar_size': cbar_size,
            'cbar_ticks_size': cbar_ticks_size,
            'ticks_size': ticks_size,
            'xticks_rotation': xticks_rotation,
            'yticks_rotation': yticks_rotation,
            'title_size': title_size,
            'legend_size' : legend_size, 
            'xlabel': xlabel,
            'xlabel_size': xlabel_size,
            'ylabel': ylabel,
            'ylabel_size': ylabel_size,
            'zlabel' : zlabel,
            'zlabel_size' : zlabel_size, 
            'color_labels': color_labels, 
            'color_hue' : color_hue, 
            'markers_list' : markers_list, 
            'marker_size' : marker_size
        }
        
    def __call__(self):
        return self.visualization_params



class Representation:
    """
    A class object that has information of a representation, such as embeddings and similarity matrices
    """

    def __init__(
        self,
        name,
        sim_mat = None,
        get_embedding = True,
        MDS_dim = 3, 
        embedding = None,
        metric = "cosine",
        shuffle = False,
        category_mat: pd.DataFrame = None,
        category_name_list = ["all"],
    ) -> None:
        """
        Args:
            name (_type_): The name of Representation (e.g. "Group 1")
            sim_mat (_type_, optional): RDM (Representational Dissimilarity Matrix) of the representation. Defaults to None.
            embedding (_type_, optional): The embedding of the representaion. Defaults to None.
            metric (str, optional): The distance metric for computing dissimilarity matrix. Defaults to "cosine".
        """

        self.name = name
        self.metric = metric
        self.shuffle = shuffle

        if sim_mat is None:
            self.embedding = embedding
            self.sim_mat = self._get_sim_mat()
        else:
            self.sim_mat = sim_mat

        if embedding is None:
            self.sim_mat = sim_mat
            if get_embedding:
                self.embedding = self._get_embedding(dim = MDS_dim)
        else:
            self.embedding = embedding

        if self.shuffle:
            self.shuffled_sim_mat = self._get_shuffled_sim_mat()

        self.category_mat = category_mat
        (
            self.object_labels,
            self.category_idx_list,
            self.num_category_list,
            self.category_name_list,
        ) = self._get_index_data(category_mat, category_name_list)

    def _get_shuffled_sim_mat(self):  # ここも、torchでも対応できるようにする必要がある。
        """
        The function for shuffling the lower trianglar matrix.
        """
        # Get the lower triangular elements of the matrix
        lower_tri = self.sim_mat[np.tril_indices(self.sim_mat.shape[0], k=-1)]

        # Shuffle the lower triangular elements
        np.random.shuffle(lower_tri)

        # Create a new matrix with the shuffled lower triangular elements
        shuffled_matrix = np.zeros_like(self.sim_mat)
        shuffled_matrix[np.tril_indices(shuffled_matrix.shape[0], k=-1)] = lower_tri
        shuffled_matrix = shuffled_matrix + shuffled_matrix.T

        return shuffled_matrix

    def _get_sim_mat(self):
        if self.metric == "dot":
            metric = "cosine"
        else:
            metric = self.metric

        return distance.cdist(
            self.embedding, self.embedding, metric=metric
        )  # ここも、torchでも対応できるようにする必要がある。backendにdistを定義すたら良いと思う。

    def _get_embedding(self, dim):  # ここも、torchでも対応できるようにする必要がある。sklearnはtorchで使えない。
        MDS_embedding = manifold.MDS(n_components = dim, dissimilarity = "precomputed", random_state = 0)
        embedding = MDS_embedding.fit_transform(self.sim_mat)
        return embedding

    def _get_index_data(self, category_mat: pd.DataFrame = None, category_name_list=None):
        if category_mat is None:
            object_labels, category_idx_list, category_num_list, new_category_name_list = None, None, None, None

        else:
            if category_name_list == ["all"]:
                new_category_name_list = category_mat.columns.tolist()
            else:
                new_category_name_list = category_name_list

            category_idx_list, category_num_list = get_category_idx(
                category_mat, new_category_name_list, show_numbers=True
            )

            object_labels = list()
            for i in range(len(category_idx_list)):
                object_labels += category_mat.index[category_idx_list[i]].tolist()

        return object_labels, category_idx_list, category_num_list, new_category_name_list
        
    def show_sim_mat(self, returned = "figure", sim_mat_format = "default", visualization_config : Visualization_Config = Visualization_Config(), fig_dir = None, ticks = None):
        """_summary_

        Args:
            returned (str, optional): "figure", "row_data" or "both" . Defaults to "figure".
            sim_mat_format (str, optional): "default", "sorted" or "both". Defaults to "default".
            visualize_matrix (Visualize_Matrix, optional): The instance of Visualize_Matrix. Defaults to None.
            fig_dir (_type_, optional): _description_. Defaults to None.
            ticks : "None", "object", "category" or "numbers". Defaults to None.
        """
        if self.category_idx_list is not None:
            sim_mat_sorted = sort_matrix_with_categories(self.sim_mat, category_idx_list=self.category_idx_list)

        if returned == "figure" or returned == "both":
            if fig_dir is not None:
                fig_path = os.path.join(fig_dir, f"RDM_{self.name}")
            else:
                fig_path = None

            if sim_mat_format == "default":
                visualize_functions.show_heatmap(self.sim_mat, title = self.name, file_name = fig_path, **visualization_config())
            
            elif sim_mat_format == "sorted":
                visualize_functions.show_heatmap(sim_mat_sorted, 
                                                 title = self.name, 
                                                 file_name = fig_path, 
                                                 ticks = ticks, 
                                                 category_name_list = self.category_name_list, 
                                                 num_category_list = self.num_category_list, 
                                                 object_labels = self.object_labels, 
                                                 **visualization_config()
                                                 )
            
            elif sim_mat_format == "both":
                sim_mat_list = [self.sim_mat, sim_mat_sorted]
                for sim_mat in sim_mat_list:
                    visualize_functions.show_heatmap(sim_mat, title = self.name, file_name = fig_path, **visualization_config())
                
            else:
                raise ValueError("sim_mat_format must be either 'default', 'sorted', or 'both'.")
        
            
        elif returned == "row_data" or returned == "both":
            if sim_mat_format == "default":
                return self.sim_mat

            elif sim_mat_format == "sorted":
                return sim_mat_sorted

            elif sim_mat_format == "both":
                return self.sim_mat, sim_mat_sorted

            else:
                raise ValueError("sim_mat_format must be either 'default', 'sorted', or 'both'.")

    def show_sim_mat_distribution(self):  # ここも、torchでも対応できるようにする必要がある。
        lower_triangular = np.tril(self.sim_mat)
        lower_triangular = lower_triangular.flatten()
        plt.hist(lower_triangular)
        plt.title(f"Distribution of RDM ({self.name})")
        plt.show()
        
    def show_embedding(self, 
                       dim, 
                       visualization_config : Visualization_Config = Visualization_Config(),
                       category_name_list = None, 
                       num_category_list = None, 
                       category_idx_list = None, 
                       title = None, 
                       legend = True, 
                       fig_dir = None, 
                       fig_name = "Aligned_embedding.png"
                       ):
        
        if fig_dir is not None:
            fig_path = os.path.join(fig_dir, fig_name)  
        else: 
            fig_path = None
        
        if category_idx_list is None:
            if self.category_idx_list is not None:
                category_name_list = self.category_name_list
                num_category_list = self.num_category_list
                category_idx_list = self.category_idx_list
            
        visualize_embedding = visualize_functions.Visualize_Embedding(
            embedding_list = [self.embedding],
            dim = dim, 
            category_name_list = category_name_list,
            num_category_list = num_category_list,
            category_idx_list = category_idx_list
        )
        
        visualize_embedding.plot_embedding(name_list = [self.name], title = title, legend = legend, save_dir = fig_path, **visualization_config())
    
class Pairwise_Analysis():
    """
    A class object that has methods conducting gw-alignment and corresponding results
    This object has information of a pair of Representations.
    """

    def __init__(self, config: Optimization_Config, source: Representation, target: Representation) -> None:
        """
        Args:
            config (Optimization_Config) : instance of Optimization_Config
            source (Representation): instance of Representation
            target (Representation): instance of Representation
        """
        self.source = source
        self.target = target
        self.config = config

        assert self.source.shuffle == self.target.shuffle, "please use the same 'shuffle' both for source and target."

        if self.source.shuffle:
            self.RDM_source = self.source.shuffled_sim_mat
            self.RDM_target = self.target.shuffled_sim_mat
            self.pair_name = f"{target.name} vs {source.name} (shuffle)"

        else:
            self.RDM_source = self.source.sim_mat
            self.RDM_target = self.target.sim_mat
            self.pair_name = f"{target.name} vs {source.name}"

        assert self.RDM_source.shape == self.RDM_target.shape, "the shape of sim_mat is not the same."

        self.backend = backend.Backend(device=self.config.device, to_types=self.config.to_types)

    def show_both_sim_mats(self):

        a = self.RDM_source
        b = self.RDM_target

        plt.figure()
        plt.subplot(121)

        plt.title("source : " + self.source.name)
        plt.imshow(a, cmap=plt.cm.jet)
        plt.colorbar(orientation="horizontal")

        plt.subplot(122)
        plt.title("target : " + self.target.name)
        plt.imshow(b, cmap=plt.cm.jet)
        plt.colorbar(orientation="horizontal")

        plt.tight_layout()
        plt.show()

        a_hist, a_bin = np.histogram(a, bins=100)
        b_hist, b_bin = np.histogram(b, bins=100)

        # print(a_hist, a_bin)
        # print(a_hist.shape, a_bin.shape)

        # plt.figure()
        # plt.suptitle('histogram')
        # plt.subplot(121)
        # plt.title('source : ' + self.source.name)
        # plt.hist(a_bin[:-1], a_bin, weights = a_hist, color = 'C0', alpha = 0.5)
        # plt.grid(True)

        # plt.subplot(122)
        # plt.title('target : ' + self.target.name)
        # plt.hist(b_bin[:-1], b_bin, weights = b_hist, color = 'C1', alpha = 0.5)
        # plt.grid(True)

        plt.figure()
        plt.title("histogram source : " + self.source.name + ", target : " + self.target.name)
        plt.hist(a_bin[:-1], a_bin, weights=a_hist, label=self.source.name, alpha=0.5)
        plt.hist(b_bin[:-1], b_bin, weights=b_hist, label=self.target.name, alpha=0.5)
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    def RSA(self, metric="spearman"):  # ここも、torchでも対応できるようにする必要がある。
        upper_tri_source = self.RDM_source[np.triu_indices(self.RDM_source.shape[0], k=1)]
        upper_tri_target = self.RDM_target[np.triu_indices(self.RDM_target.shape[0], k=1)]

        if metric == "spearman":
            corr, _ = spearmanr(upper_tri_source, upper_tri_target)
        elif metric == "pearson":
            corr, _ = pearsonr(upper_tri_source, upper_tri_target)

        return corr

    def match_sim_mat_distribution(self):
        matching = SimpleHistogramMatching(self.RDM_source, self.RDM_target)

        self.RDM_target = matching.simple_histogram_matching()
        
    
    def run_gw(self, 
               results_dir, 
               load_OT = False, 
               returned = "figure", 
               OT_format = "default", 
               visualization_config : Visualization_Config = Visualization_Config(), 
               show_log = False,
               fig_dir = None, 
               ticks = None):
        """
        Main computation
        """            
        self.OT, df_trial = self._gw_alignment(results_dir, load_OT = load_OT)
        OT = self.show_OT(title = f"$\Gamma$ ({self.pair_name})", 
                      returned = returned, 
                      OT_format = OT_format, 
                      visualization_config = visualization_config, 
                      fig_dir = fig_dir, 
                      ticks = ticks)
        
        if show_log:
            self._get_optimization_log(df_trial, fig_dir = fig_dir)
        
        return OT

    def _gw_alignment(self, results_dir, load_OT=False):

        filename = self.config.data_name + " " + self.pair_name

        save_path = os.path.join(results_dir, filename)
        
        storage = "sqlite:///" + save_path + "/" + filename + ".db"

        # distribution in the source space, and target space
        p = ot.unif(len(self.RDM_source))
        q = ot.unif(len(self.RDM_target))

        # generate instance solves gw_alignment
        gw = GW_Alignment(
            self.RDM_source,
            self.RDM_target,
            p,
            q,
            save_path,
            max_iter=self.config.max_iter,
            n_iter=self.config.n_iter,
            to_types=self.config.to_types,
        )

        # generate instance optimize gw_alignment
        opt = gw_optimizer.load_optimizer(
            save_path,
            n_jobs=self.config.n_jobs,
            num_trial=self.config.num_trial,
            to_types=self.config.to_types,
            method="optuna",
            sampler_name=self.config.sampler_name,
            pruner_name=self.config.pruner_name,
            pruner_params=self.config.pruner_params,
            n_iter=self.config.n_iter,
            filename=filename,
            storage=storage,
            delete_study=self.config.delete_study,
        )

        if not load_OT:
            ### optimization
            # 1. choose the initial matrix for GW alignment computation.
            init_plans = init_matrix.InitMatrix().implemented_init_plans(self.config.init_plans_list)

            # used only in grid search sampler below the two lines
            eps_space = opt.define_eps_space(self.config.eps_list, self.config.eps_log, self.config.num_trial)
            search_space = {"eps": eps_space, "initialize": init_plans}

            # 2. run optimzation
            study = opt.run_study(
                gw,
                self.config.device,
                init_plans_list=init_plans,
                eps_list=self.config.eps_list,
                eps_log=self.config.eps_log,
                search_space=search_space,
            )

            best_trial = study.best_trial
            df_trial = study.trials_dataframe()
            
            if self.config.to_types == 'numpy':
                OT = np.load(save_path + f"/{self.config.init_plans_list[0]}/gw_{best_trial.number}.npy")

            elif self.config.to_types == "torch":
                OT = torch.load(save_path + f"/{self.config.init_plans_list[0]}/gw_{best_trial.number}.pt")
                OT = OT.to("cpu").numpy()

        else:
            study = opt.load_study()
            best_trial = study.best_trial
            df_trial = study.trials_dataframe()
            
            if self.config.to_types == 'numpy':
                OT = np.load(save_path + f"/{self.config.init_plans_list[0]}/gw_{best_trial.number}.npy")

            elif self.config.to_types == "torch":
                OT = torch.load(save_path + f"/{self.config.init_plans_list[0]}/gw_{best_trial.number}.pt")
                
                OT = OT.to('cpu').numpy()
        
        return OT, df_trial
          
    def _get_optimization_log(self, df_trial, fig_dir):
        # figure plotting epsilon as x-axis and GWD as y-axis
        sns.scatterplot(data = df_trial, x = "params_eps", y = "value", s = 50)
        plt.xlabel("$\epsilon$")
        plt.ylabel("GWD")
        if fig_dir is not None:
            fig_path = os.path.join(fig_dir, f"Optim_log_eps_GWD_{self.pair_name}.png")
            plt.savefig(fig_path)
        plt.tight_layout()
        plt.show()
        
        # 　figure plotting GWD as x-axis and accuracy as y-axis
        sns.scatterplot(data = df_trial, x = "value", y = "user_attrs_best_acc", s = 50)
        plt.xlabel("GWD")
        plt.ylabel("accuracy")
        if fig_dir is not None:
            fig_path = os.path.join(fig_dir, f"Optim_log_acc_GWD_{self.pair_name}.png")
            plt.savefig(fig_path)
        plt.tight_layout()
        plt.show()
    
    def show_OT(self, 
                 title, 
                 returned = "figure",
                 OT_format = "default",
                 visualization_config : Visualization_Config = Visualization_Config(),
                 fig_dir = None,
                 ticks = None):
        
        if self.source.category_idx_list is not None:
            OT_sorted = sort_matrix_with_categories(self.OT, category_idx_list=self.source.category_idx_list)

        if returned == "figure" or returned == "both":
            if fig_dir is not None:
                fig_path = os.path.join(fig_dir, f"{title}.png")
            else:
                fig_path = None

            if OT_format == "default":
                visualize_functions.show_heatmap(self.OT, title = title, file_name = fig_path, **visualization_config())
            
            elif OT_format == "sorted":
                visualize_functions.show_heatmap(OT_sorted, 
                                                 title = title, 
                                                 file_name = fig_path, 
                                                 ticks = ticks, 
                                                 category_name_list = self.source.category_name_list, 
                                                 num_category_list = self.source.num_category_list, 
                                                 object_labels = self.source.object_labels, 
                                                 **visualization_config()
                                                 )
            
            elif OT_format == "both":
                OT_list = [self.OT, OT_sorted]
                for OT in OT_list:
                    visualize_functions.show_heatmap(OT, title = title, file_name = fig_path, **visualization_config())
                    
            else:
                raise ValueError("OT_format must be either 'default', 'sorted', or 'both'.")
            
            
        elif returned == "row_data" or returned == "both":
            if OT_format == "default":
                return self.OT

            elif OT_format == "sorted":
                return OT_sorted

            elif OT_format == "both":
                return self.OT, OT_sorted

            else:
                raise ValueError("OT_format must be either 'default', 'sorted', or 'both'.")

    def calc_category_level_accuracy(self, category_mat=None):
        if category_mat is None:
            category_mat = self.source.category_mat.values
        else:
            category_mat = category_mat.values
        count = 0

        for i in range(self.OT.shape[0]):
            max_index = np.argmax(self.OT[i])

            if np.array_equal(category_mat[i], category_mat[max_index]):
                count += 1

        accuracy = count / self.OT.shape[0] * 100

        return accuracy

    def eval_accuracy(self, top_k_list, eval_type="ot_plan", metric="cosine", barycenter=False, supervised=False):
        df = pd.DataFrame()
        df["top_n"] = top_k_list

        if supervised:
            OT = np.diag([1 / len(self.target.sim_mat) for _ in range(len(self.target.sim_mat))])
        else:
            OT = self.OT

        acc_list = list()
        for k in top_k_list:
            if eval_type == "k_nearest":
                if not barycenter:
                    new_embedding_source = self.procrustes(self.target.embedding, self.source.embedding, OT)
                else:
                    new_embedding_source = self.source.embedding
                    
                # Compute distances between each points
                dist_mat = distance.cdist(self.target.embedding, new_embedding_source, metric)

                acc = self._calc_accuracy_with_topk_diagonal(dist_mat, k=k, order="minimum")

            elif eval_type == "ot_plan":
                acc = self._calc_accuracy_with_topk_diagonal(OT, k=k, order="maximum")

            acc_list.append(acc)

        df[self.pair_name] = acc_list

        return df

    def _calc_accuracy_with_topk_diagonal(self, matrix, k, order="maximum"):
        # Get the diagonal elements
        diagonal = np.diag(matrix)

        # Get the top k values for each row
        if order == "maximum":
            topk_values = np.partition(matrix, -k)[:, -k:]
        elif order == "minimum":
            topk_values = np.partition(matrix, k - 1)[:, :k]
        else:
            raise ValueError("Invalid order parameter. Must be 'maximum' or 'minimum'.")

        # Count the number of rows where the diagonal is in the top k values
        count = np.sum(np.isin(diagonal, topk_values))

        # Calculate the accuracy as the proportion of counts to the total number of rows
        accuracy = count / matrix.shape[0]
        accuracy *= 100

        return accuracy

    def procrustes(self, embedding_target, embedding_sourse, OT):
        """
        Function that brings embedding_sourse closest to embedding_target by orthogonal matrix

        Args:
            embedding_target : shape (n_target, m)
            embedding_sourse : shape (n_sourse, m)
            OT : shape (n_sourse, n_target)
                Transportation matrix of sourse→target

        Returns:
            new_embedding_sourse : shape (n_sourse, m)
        """
        assert self.source.shuffle == False, "you cannot use procrustes method if 'shuffle' is True."

        U, S, Vt = np.linalg.svd(np.matmul(embedding_sourse.T, np.matmul(OT, embedding_target)))
        Q = np.matmul(U, Vt)
        new_embedding_sourse = np.matmul(embedding_sourse, Q)

        return new_embedding_sourse

    def wasserstein_alignment(self, metric):
        a = ot.unif(len(self.source.embedding))
        b = ot.unif(len(self.target.embedding))
        
        M = distance.cdist(self.source.embedding, self.target.embedding, metric=metric)

        self.OT, log = ot.emd(a, b, M, log=True)     
        
        return log["cost"]
           
        
    def get_new_source_embedding(self):
        return self.procrustes(self.target.embedding, self.source.embedding, self.OT)


class Align_Representations:
    """
    This object has methods for conducting N groups level analysis and corresponding results.
    This has information of all pairs of representations.
    """

    def __init__(
        self,
        config: Optimization_Config,
        representations_list: List[Representation],
        pair_number_list="all",
        metric="cosine",
        shuffle=False,
    ) -> None:
        """
        Args:
            representations_list (list): a list of Representations
        """
        self.config = config

        self.metric = metric
        self.representations_list = representations_list
        self.pairwise_list = self._get_pairwise_list()

        self.RSA_corr = dict()

        self.shuffle = shuffle

        if pair_number_list == "all":
            pair_number_list = range(len(self.pairwise_list))

        self.pair_number_list = pair_number_list

    def _get_pairwise_list(self) -> List[Pairwise_Analysis]:
        pairs = list(itertools.combinations(self.representations_list, 2))

        pairwise_list = list()
        for i, pair in enumerate(pairs):
            pairwise = Pairwise_Analysis(config=self.config, source=pair[1], target=pair[0])
            pairwise_list.append(pairwise)
            print(f"Pair number {i} : {pairwise.pair_name}")

        return pairwise_list

    def RSA_get_corr(self, metric="spearman"):
        for pairwise in self.pairwise_list:
            corr = pairwise.RSA(metric=metric)
            self.RSA_corr[pairwise.pair_name] = corr
            print(f"Correlation {pairwise.pair_name} : {corr}")
    
    def show_sim_mat(self, returned = "figure", sim_mat_format = "default", visualization_config : Visualization_Config = Visualization_Config(), fig_dir = None, show_distribution = True, ticks = None):
        """_summary_

        Args:
            returned (str, optional): "figure", "row_data" or "both" . Defaults to "figure".
            sim_mat_format (str, optional): "default", "sorted" or "both". Defaults to "default".
            visualization_config (Visualization_Config, optional): The instance of Visualization_Config. Defaults to None.
            fig_dir (_type_, optional): _description_. Defaults to None.
        """
        for representation in self.representations_list:
            representation.show_sim_mat(returned = returned,
                                        sim_mat_format = sim_mat_format,
                                        visualization_config = visualization_config,
                                        fig_dir = fig_dir, 
                                        ticks = ticks
                                        )
            if show_distribution:
                representation.show_sim_mat_distribution()
    
    def gw_alignment(self, results_dir, load_OT = False, returned = "figure", OT_format = "default", visualization_config : Visualization_Config = Visualization_Config(), show_log = False, fig_dir = None, ticks = None):
        OT_list = []
        for pair_number in self.pair_number_list:
            pairwise = self.pairwise_list[pair_number]
            OT = pairwise.run_gw(results_dir = results_dir,
                            load_OT = load_OT,
                            returned = returned,
                            OT_format = OT_format,
                            visualization_config = visualization_config,
                            show_log = show_log,
                            fig_dir = fig_dir, 
                            ticks = ticks
                            )
            OT_list.append(OT)

        if returned == "row_data":
            return OT_list
    
    def calc_barycenter(self, X_init=None):
        embedding_list = [representation.embedding for representation in self.representations_list]
        
        if X_init is None:
            X_init = np.mean(embedding_list, axis=0) # initial Dirac locations
            
        b = ot.unif(len(X_init)) # weights of the barycenter
        
        weights_list = []# measures weights
        for representation in self.representations_list:
            weights = ot.unif(len(representation.embedding))
            weights_list.append(weights)
        
        X = ot.lp.free_support_barycenter(embedding_list, weights_list, X_init, b) # # new location of the barycenter
        
        return X
        
    def barycenter_alignment(self, 
                             pivot, 
                             n_iter, 
                             results_dir, 
                             load_OT=False, 
                             returned="figure", 
                             OT_format="default", 
                             visualization_config:Visualization_Config=Visualization_Config(), 
                             show_log=False, 
                             fig_dir=None, 
                             ticks=None):
        
        assert self.pair_number_list == range(len(self.pairwise_list))
        
        ### Select the pivot
        pivot_representation = self.representations_list[pivot]
        others_representaions = self.representations_list[:pivot] + self.representations_list[pivot + 1:]
        
        ### GW alignment to the pivot
        # ここの部分はあとでself.gw_alignmentの中に組み込む
        for representation in others_representaions:
            pairwise = Pairwise_Analysis(config=self.config, source=representation, target=pivot_representation)

            pairwise.run_gw(results_dir=results_dir,
                            load_OT=load_OT,
                            returned=returned,
                            OT_format=OT_format,
                            visualization_config=visualization_config,
                            show_log=show_log,
                            fig_dir=fig_dir, 
                            ticks=ticks
                            )

            pairwise.source.embedding = pairwise.get_new_source_embedding()

        ### Set up barycenter
        init_embedding = self.calc_barycenter()
        self.barycenter = Representation(name="barycenter",
                                         embedding=init_embedding,
                                         category_mat=self.representations_list[0].category_mat,
                                         category_name_list=self.representations_list[0].category_name_list
                                         )
        
        ### Set pairwises whose target is the barycenter
        pairwise_barycenters = []
        for representation in self.representations_list:
            pairwise = Pairwise_Analysis(config=self.config, source=representation, target=self.barycenter)
            pairwise_barycenters.append(pairwise)
        
        ### Barycenter alignment
        loss_list = []
        embedding_barycenter = init_embedding
        for i in range(n_iter):   
            embedding_barycenter = self.calc_barycenter(X_init=embedding_barycenter)
            
            loss = 0
            for pairwise in pairwise_barycenters:
                # update the embedding of the barycenter
                pairwise.target.embedding = embedding_barycenter
                
                # OT to the barycenter
                loss += pairwise.wasserstein_alignment(metric=self.metric)
                
                # update the embeddings of each representation
                pairwise.source.embedding = pairwise.get_new_source_embedding()
                
            loss /= len(pairwise_barycenters)
            loss_list.append(loss)
        
        plt.figure()
        plt.plot(loss_list)
        plt.xlabel("iteration")
        plt.ylabel("Mean Wasserstein distance")
        
        ### replace OT of each pairwise by the product of OTs to the barycenter
        self._get_OT_all_pair(pairwise_barycenters)

        ### visualize
        OT_list = []
        for pairwise in self.pairwise_list:
            OT = pairwise.show_OT(title = f"$\Gamma$ ({pairwise.pair_name})",
                                returned = returned,
                                OT_format = OT_format,
                                visualization_config = visualization_config,
                                fig_dir = fig_dir,
                                ticks = ticks)
            OT_list.append(OT)
        
        if returned == "row_data":
            return OT_list
        
    def _get_OT_all_pair(self, pairwise_barycenters):
        # replace OT of each pairwise by the product of OTs to the barycenter
        pairs = list(itertools.combinations(pairwise_barycenters, 2))
        
        for i, (pairwise_1, pairwise_2) in enumerate(pairs):
            OT = np.matmul(pairwise_2.OT, pairwise_1.OT.T)
            OT *= len(OT) # normalize
            self.pairwise_list[i].OT = OT
            
    def calc_accuracy(self, top_k_list, eval_type="ot_plan", barycenter=False):
        accuracy = pd.DataFrame()
        accuracy["top_n"] = top_k_list

        for pair_number in self.pair_number_list:
            pairwise = self.pairwise_list[pair_number]
            df = pairwise.eval_accuracy(top_k_list, eval_type=eval_type, metric=self.metric, barycenter=barycenter)

            accuracy = pd.merge(accuracy, df, on="top_n")

        accuracy = accuracy.set_index("top_n")

        if eval_type == "ot_plan":
            self.top_k_accuracy = accuracy
            print("Top k accuracy : \n", accuracy)

        elif eval_type == "k_nearest":
            self.k_nearest_matching_rate = accuracy
            print("K nearest matching rate : \n", accuracy)

        print("Mean : \n", accuracy.iloc[:, 1:].mean(axis="columns"))

    def calc_category_level_accuracy(self, 
                                     make_hist=False, 
                                     fig_dir=None, 
                                     fig_name="Category_level_accuracy.png", 
                                     category_mat=None):
        acc_list = []
        for pairnumber in self.pair_number_list:
            pairwise = self.pairwise_list[pairnumber]
            acc = pairwise.calc_category_level_accuracy(category_mat=category_mat)
            print(f"{pairwise.pair_name} :  {acc}")
            acc_list.append(acc)

        if make_hist:
            plt.figure()
            plt.hist(acc_list)
            plt.xlabel("Accuracy")
            plt.savefig(os.path.join(fig_dir, fig_name))
            plt.show()

    def _get_dataframe(self, eval_type="ot_plan", shuffle=False, concat=True):
        df = self.top_k_accuracy if eval_type == "ot_plan" else self.k_nearest_matching_rate

        if not shuffle:
            cols = [col for col in df.columns if "shuffle" not in col and "top_n" not in col]
        else:
            cols = [col for col in df.columns if "shuffle" in col]

        df = df[cols]
        if concat:
            df = pd.concat([df[i] for i in df.columns], axis=0)
            df = df.rename("matching rate")
        return df

    def plot_accuracy(
        self, eval_type="ot_plan", shuffle=False, fig_dir=None, fig_name="Accuracy_ot_plan.png", scatter=True
    ):
        plt.figure(figsize=(5, 3))

        if scatter:
            df = self._get_dataframe(eval_type, shuffle=shuffle, concat=True)
            sns.set_style("darkgrid")
            sns.set_palette("pastel")
            sns.swarmplot(data=pd.DataFrame(df), x="top_n", y="matching rate", size=5, dodge=True)

        else:
            df = self._get_dataframe(eval_type, shuffle=shuffle, concat=False)
            for group in df.columns:
                plt.plot(df.index, df[group], c="blue")

        plt.ylim(0, 100)
        plt.xlabel("k")
        plt.ylabel("Matching rate")
        # plt.legend(loc = "best")
        plt.tick_params(axis="both", which="major")
        plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2)
        if fig_dir is not None:
            plt.savefig(os.path.join(fig_dir, fig_name))
        plt.show()
    
    def procrustes_to_pivot(self):
        pass
    
    def visualize_embedding(self, 
                            dim, 
                            pivot = 0,
                            returned = "figure", 
                            visualization_config : Visualization_Config = Visualization_Config(), 
                            category_name_list = None, 
                            num_category_list = None, 
                            category_idx_list = None, 
                            title = None, 
                            legend = True, 
                            fig_dir = None, 
                            fig_name = "Aligned_embedding.png"):
        """_summary_

        Args:
            dim (_type_): The number of dimensions the points are embedded.  
            pivot (str, optional) : The pivot or "barycenter" to which all embeddings are aligned. Defaults to 0.
            returned (str, optional): "figure" or "row_data. Defaults to "figure".
            visualization_config (Visualization_Config, optional): _description_. Defaults to Visualization_Config().
            category_name_list (_type_, optional): _description_. Defaults to None.
            num_category_list (_type_, optional): _description_. Defaults to None.
            category_idx_list (_type_, optional): _description_. Defaults to None.
            title (_type_, optional): _description_. Defaults to None.
            legend (bool, optional): _description_. Defaults to True.
            fig_dir (_type_, optional): _description_. Defaults to None.
            fig_name (str, optional): _description_. Defaults to "Aligned_embedding.png".

        Returns:
            _type_: _description_
        """
        
        if fig_dir is not None:
            fig_path = os.path.join(fig_dir, fig_name)
        else:
            fig_path = None

        if pivot != "barycenter":
            # self.procrustes_to_pivot()
            for i in range(len(self.pairwise_list) // 2):
                pair = self.pairwise_list[i]
                pair.source.embedding = pair.get_new_source_embedding()
        
        else:
            assert(self.barycenter is not None)

        name_list = []
        embedding_list = []
        for i in range(len(self.representations_list)):
            embedding_list.append(self.representations_list[i].embedding)
            name_list.append(self.representations_list[i].name)
        
        if returned == "figure":
            if category_idx_list is None:
                if self.representations_list[0].category_idx_list is not None:
                    category_name_list = self.representations_list[0].category_name_list
                    num_category_list = self.representations_list[0].num_category_list
                    category_idx_list = self.representations_list[0].category_idx_list

            visualize_embedding = visualize_functions.Visualize_Embedding(
                embedding_list = embedding_list,
                dim = dim, 
                category_name_list = category_name_list,
                num_category_list = num_category_list,
                category_idx_list = category_idx_list
            )

            visualize_embedding.plot_embedding(name_list = name_list, title = title, legend = legend, save_dir = fig_path, **visualization_config())

        elif returned == "row_data":
            return embedding_list
