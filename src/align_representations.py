#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.spatial import distance
from scipy.stats import spearmanr, pearsonr
from sklearn import manifold
import ot
import sys
import os
from typing import List

from utils.utils_functions import get_category_idx
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

class Representation():
    """
    A class object that has information of a representation, such as embeddings and similarity matrices
    """
    def __init__(self, name, sim_mat = None, embedding = None, metric = "cosine", shuffle = False) -> None:
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
            self.embedding = self._get_embedding()
        else:
            self.embedding = embedding
        
        if self.shuffle:
            self.shuffled_sim_mat = self._get_shuffled_sim_mat()

    def _get_shuffled_sim_mat(self):# ここも、torchでも対応できるようにする必要がある。
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
        
        return distance.cdist(self.embedding, self.embedding, metric = metric)# ここも、torchでも対応できるようにする必要がある。backendにdistを定義すたら良いと思う。
    
    def _get_embedding(self):# ここも、torchでも対応できるようにする必要がある。sklearnはtorchで使えない。
        MDS_embedding = manifold.MDS(n_components = 3, dissimilarity = 'precomputed', random_state = 0)
        embedding = MDS_embedding.fit_transform(self.sim_mat)
        return embedding
        
    def show_sim_mat(self, ticks_size = None, label = None, fig_dir = None):
        if fig_dir is not None:
            fig_path = os.path.join(fig_dir, f"RDM_{self.name}.png")
        else:
            fig_path = None
        
        visualize_functions.show_heatmap(
            self.sim_mat, 
            title = self.name, 
            ticks_size = ticks_size, 
            xlabel = label, 
            ylabel = label, 
            file_name = fig_path,
        )
        
    def show_sim_mat_distribution(self):# ここも、torchでも対応できるようにする必要がある。
        lower_triangular = np.tril(self.sim_mat)
        lower_triangular = lower_triangular.flatten()
        plt.hist(lower_triangular)
        plt.title(f"Distribution of RDM ({self.name})")
        plt.show()
        
    def show_embedding(self, dim = 3):
        visualize_embedding = visualize_functions.Visualize_Embedding(
            embedding_list = [self.embedding], 
            name_list = [self.name],
        )
        
        visualize_embedding.plot_embedding(dim = dim)
    
class Pairwise_Analysis():
    """
    A class object that has methods conducting gw-alignment and corresponding results
    This object has information of a pair of Representations.
    """
    def __init__(self, config : Optimization_Config, source : Representation, target : Representation) -> None:
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
        
        
        self.backend = backend.Backend(device=self.config.device, to_types=self.config)
        
        self.RDM_source, self.RDM_target = self.backend(self.RDM_source, self.RDM_target)
        
    
    def show_both_sim_mats(self):
        
        if self.config.to_types == 'torch':
            a = self.RDM_source.to('cpu').numpy()
            b = self.RDM_target.to('cpu').numpy()
        else:
            a = self.RDM_source
            b = self.RDM_target
        
        plt.figure()
        plt.subplot(121)

        plt.title('source : ' + self.source.name)
        plt.imshow(a, cmap=plt.cm.jet)
        plt.colorbar(orientation='horizontal')

        plt.subplot(122)
        plt.title('target : ' + self.target.name)
        plt.imshow(b , cmap=plt.cm.jet)
        plt.colorbar(orientation='horizontal')

        plt.tight_layout()
        plt.show()
        
    
    def RSA(self, metric = "spearman"):# ここも、torchでも対応できるようにする必要がある。
        upper_tri_source = self.RDM_source[np.triu_indices(self.RDM_source.shape[0], k=1)]
        upper_tri_target = self.RDM_target[np.triu_indices(self.RDM_target.shape[0], k=1)]
        
        if metric == "spearman":
            corr, _ = spearmanr(upper_tri_source, upper_tri_target)
        elif metric == "pearson":
            corr, _ = pearsonr(upper_tri_source, upper_tri_target)
        
        return corr
    
    def match_sim_mat_distribution(self):
        matching = SimpleHistogramMatching(self.RDM_source, self.RDM_target, self.backend)
        
        self.RDM_target = matching.simple_histogram_matching()
        
    
    def run_gw(self, ticks_size = None, load_OT = False, fig_dir = None):
        """
        Main computation
        """            
        self.OT = self._gw_alignment(load_OT = load_OT)
        self._show_OT(title = f"$\Gamma$ ({self.pair_name})", ticks_size = ticks_size, fig_dir = fig_dir)
        
    def _gw_alignment(self, results_dir = "../results/", load_OT = False):
        
        filename = self.config.data_name + " " + self.pair_name
        
        sql_name = "sqlite"
        storage = "sqlite:///" + results_dir + "/" + filename + ".db"
        
        save_path = results_dir + filename
        
        # distribution in the source space, and target space
        p = ot.unif(len(self.RDM_source))
        q = ot.unif(len(self.RDM_target))

        # generate instance solves gw_alignment
        test_gw = GW_Alignment(
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
            sql_name=sql_name,
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
                test_gw,
                self.config.device,
                init_plans_list=init_plans,
                eps_list=self.config.eps_list,
                eps_log=self.config.eps_log,
                search_space=search_space,
            )
            
            best_trial = study.best_trial
            OT = np.load(save_path + f"/{self.config.init_plans_list[0]}/gw_{best_trial.number}.npy")
        
        else:
            study = opt.load_study()
            best_trial = study.best_trial
            OT = np.load(save_path + f"/{self.config.init_plans_list[0]}/gw_{best_trial.number}.npy")
        
        return OT
          
    def _get_optimization_log(self):
        pass
    
    def _show_OT(self, title, ticks_size = None, fig_dir = None):
        if fig_dir is not None:
            fig_path = os.path.join(fig_dir, f"OT_{self.pair_name}.png")  
        else: 
            fig_path = None
            
        visualize_functions.show_heatmap(matrix = self.OT, title = title, ticks_size = ticks_size, file_name = fig_path)
    
    def eval_accuracy(self, top_k_list, eval_type = "ot_plan",  metric = "cosine", supervised = False):
        df = pd.DataFrame()
        
        df["top_n"] = top_k_list

        if supervised:
            OT = self.backend.nx.diag([1/len(self.target.sim_mat) for _ in range(len(self.target.sim_mat))]) # ここも、torchでも対応できるようにする必要がある。
        else:
            OT = self.OT
        
        acc_list = list()
        for k in top_k_list:
            if eval_type == "k_nearest":
                """
                2023.5.15 佐々木
                ここのtop_kの算出方法は直すべき箇所で間違いないが、
                全く本質的ではないので、全部公開するときに直した方がいい。
                """
                new_embedding_source = self.procrustes(self.target.embedding, self.source.embedding, OT)
                
                # Compute distances between each points
                dist_mat = distance.cdist(self.target.embedding, new_embedding_source, metric) # ここも、torchでも対応できるようにする必要がある。

                # Get sorted indices 
                sorted_idx = self.backend.nx.argsort(dist_mat, axis = 1)
                # sorted_idx = np.argpartition(dist_mat, )

                # Get the same colors and count k-nearest
                acc = 0
                for i in range(self.target.embedding.shape[0]):
                    acc += (sorted_idx[i, :k]  == i).sum() 
                acc /= self.target.embedding.shape[0]
                acc *= 100
            
            elif eval_type == "ot_plan":
                acc = 0
                for i in range(OT.shape[0]):
                    idx = self.backend.nx.argsort(-OT[i, :])
                    acc += (idx[:k] == i).sum()    
                acc /= OT.shape[0]
                acc *= 100
            
            acc_list.append(acc)
        
        df[self.pair_name] = acc_list
        
        return df
    
    # def procrustes(self):
    #     self.source.embedding = self._procrustes(self.target.embedding, self.source.embedding, self.OT)

    def procrustes(self, embedding_1, embedding_2, Pi):
        """
        embedding_2をembedding_1に最も近づける回転行列Qを求める

        Args:
            embedding_1 : shape (n_1, m)
            embedding_2 : shape (n_2, m)
            Pi : shape (n_2, n_1) 
                Transportation matrix of 2→1
            
        Returns:
            new_embedding_2 : shape (n_2, m)
        """
        assert self.source.shuffle == False, "you cannot use procrustes method if 'shuffle' is True."
        
        # ここも、torchでも対応できるようにする必要がある。
        U, S, Vt = np.linalg.svd(np.matmul(embedding_2.T, np.matmul(Pi, embedding_1)))
        Q = np.matmul(U, Vt)
        new_embedding_2 = np.matmul(embedding_2, Q)
        
        return new_embedding_2
    
    def get_new_source_embedding(self):
        return self.procrustes(self.target.embedding, self.source.embedding, self.OT)



class Align_Representations():
    """
    This object has methods for conducting N groups level analysis and corresponding results.
    This has information of all pairs of representations.
    """
    def __init__(self, config : Optimization_Config, representations_list : List[Representation], pair_number_list = "all", metric = 'cosine', shuffle = False) -> None:
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
            pairwise = Pairwise_Analysis(config = self.config, source = pair[1], target = pair[0])
            pairwise_list.append(pairwise)    
            print(f"Pair number {i} : {pairwise.pair_name}")
        
        return pairwise_list

    def RSA_get_corr(self, metric = "spearman"):
        for pairwise in self.pairwise_list:
            corr = pairwise.RSA(metric = metric)
            self.RSA_corr[pairwise.pair_name] = corr
            print(f"Correlation {pairwise.pair_name} : {corr}")
    
    def show_sim_mat(self, ticks_size = None, label = None, fig_dir = None, show_distribution = True):
        for representation in self.representations_list:
            representation.show_sim_mat(ticks_size = ticks_size, label = label, fig_dir = fig_dir)
            if show_distribution:
                representation.show_sim_mat_distribution()
    
    def gw_alignment(self, ticks_size = None, load_OT = False, fig_dir = None):
        for pair_number in self.pair_number_list:
            pairwise = self.pairwise_list[pair_number]
            pairwise.run_gw(ticks_size = ticks_size, load_OT = load_OT, fig_dir = fig_dir)
            
    def barycenter_alignment(self):
        pass
       
    def calc_accuracy(self, top_k_list, eval_type = "ot_plan"):
        accuracy = pd.DataFrame()
        accuracy["top_n"] = top_k_list
        
        for pair_number in self.pair_number_list:
            pairwise = self.pairwise_list[pair_number]
            df = pairwise.eval_accuracy(top_k_list, eval_type = eval_type, metric = self.metric)

            accuracy = pd.merge(accuracy, df, on = "top_n")
        
        accuracy = accuracy.set_index("top_n")
        
        if eval_type == "ot_plan":
            self.top_k_accuracy = accuracy
            print("Top k accuracy : \n", accuracy)
        
        elif eval_type == "k_nearest":
            self.k_nearest_matching_rate = accuracy  
            print("K nearest matching rate : \n", accuracy)
        
        print("Mean : \n", accuracy.iloc[:, 1:].mean(axis = "columns"))
        
    
    def _get_dataframe(self, eval_type = "ot_plan", shuffle = False, concat = True):
        df = self.top_k_accuracy if eval_type == "ot_plan" else self.k_nearest_matching_rate         
        
        if not shuffle:
            cols = [col for col in df.columns if "shuffle" not in col and "top_n" not in col]
        else:
            cols = [col for col in df.columns if "shuffle" in col]
        
        df = df[cols]
        if concat:
            df = pd.concat([df[i] for i in df.columns], axis = 0)
            df = df.rename("matching rate")
        return df
        
    def plot_accuracy(self, eval_type = "ot_plan", shuffle = False, fig_dir = None, fig_name = None, scatter = True):
        plt.figure(figsize = (5, 3)) 
        
        if scatter:
            df = self._get_dataframe(eval_type, shuffle = shuffle, concat = True)
            sns.set_style("darkgrid")
            sns.set_palette("pastel")
            sns.swarmplot(data = pd.DataFrame(df), x = "top_n", y = "matching rate", size = 5, dodge = True)
        
        else:
            df = self._get_dataframe(eval_type, shuffle = shuffle, concat = False)
            for group in df.columns:
                plt.plot(df.index, df[group], c = "blue")
        
        plt.ylim(0, 100)
        plt.xlabel("k")
        plt.ylabel("Matching rate")
        #plt.legend(loc = "best")
        plt.tick_params(axis = "both", which = "major")
        plt.subplots_adjust(left=0.2, right=0.9, bottom = 0.2)
        if fig_dir is not None:
            plt.savefig(os.path.join(fig_dir, fig_name))
        plt.show()
    
    def visualize_embedding(self, 
                            dim = 3, 
                            color_labels = None, 
                            category_name_list = None, 
                            category_num_list = None, 
                            category_idx_list = None, 
                            fig_dir = None):
        
        if fig_dir is not None:
            fig_path = os.path.join(fig_dir, "Aligned_embedding.png")  
        else: 
            fig_path = None
            
        name_list = []
        embedding_list = []
        for i in range(len(self.pairwise_list) // 2):
            pair = self.pairwise_list[i]
            embedding_list.append(pair.get_new_source_embedding())
            name_list.append(pair.pair_name)
            
        visualize_embedding = visualize_functions.Visualize_Embedding(
            embedding_list = embedding_list,
            name_list = name_list,
            color_labels = color_labels,
            category_name_list = category_name_list,
            category_num_list = category_num_list,
            category_idx_list = category_idx_list
        )
        
        visualize_embedding.plot_embedding(dim = dim, save_dir = fig_path)


#%%
if __name__ == "__main__":
    '''
    parameters
    '''
    n_group = 4
    metric = "euclidean"
    
    #%%
    '''
    Create subject groups list
    '''
    representations = []
    for i in range(n_group):
        name = f"Group{i+1}"
        embedding = np.load(f"../data/THINGS_embedding_Group{i+1}.npy")[0]
        representation = Representation(name = name, embedding = embedding, metric = metric, shuffle = False)
        representations.append(representation)
    
    #%%
    '''
    Unsupervised alignment between Representations
    '''
    test_config = Optimization_Config(delete_study=False, n_jobs=1)
    align_representations = Align_Representations(config = test_config, representations_list = representations)
    
    #%%
    # RSA
    align_representations.show_sim_mat()
    align_representations.RSA_get_corr()
    
    #%%
    # Run gw
    align_representations.gw_alignment(load_OT = True)
    #%%
    '''
    Evaluate the accuracy
    '''
    ## Accuracy of the optimized OT matrix
    align_representations.calc_accuracy(top_k_list = [1, 5, 10], eval_type = "ot_plan")
    
    # %%
    align_representations.plot_accuracy(eval_type = "ot_plan", scatter = True)
    # %%
    ## Matching rate of k-nearest neighbors 
    align_representations.calc_accuracy(top_k_list = [1, 5, 10], eval_type = "k_nearest")
    # %%
    align_representations.plot_accuracy(eval_type = "k_nearest", scatter = True)
    
    #%%
    '''
    Visualize embedding
    '''
    ## Load the coarse categories data
    category_name_list = ["bird", "insect", "plant", "clothing",  "furniture", "fruit", "drink", "vehicle"]
    category_mat = pd.read_csv("../data/category_mat_manual_preprocessed.csv", sep = ",", index_col = 0)   
    category_idx_list, category_num_list = get_category_idx(category_mat, category_name_list, show_numbers = True)  
    
    align_representations.visualize_embedding(dim = 3, category_name_list = category_name_list, category_idx_list = category_idx_list, category_num_list = category_num_list)
# %%
