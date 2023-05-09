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

from utils.utils_functions import procrustes, get_category_idx
from utils import visualize_functions, evaluation
from gw_alignment import run_main_gw, Optimization_Config


class Representation:
    """
    A class object that has information of a representation, such as embeddings and similarity matrices
    """
    def __init__(self, name, sim_mat = None, embedding = None, metric = "cosine") -> None:
        """_summary_

        Args:
            name (_type_): The name of Representation (e.g. "Group 1")
            sim_mat (_type_, optional): RDM (Representational Dissimilarity Matrix) of the representation. Defaults to None.
            embedding (_type_, optional): The embedding of the representaion. Defaults to None.
            metric (str, optional): The distance metric for computing dissimilarity matrix. Defaults to "cosine".
        """
        self.name = name
        self.metric = metric
        if sim_mat is None:
            self.embedding = embedding
            self.sim_mat = self._get_sim_mat()
        elif embedding is None:
            self.sim_mat = sim_mat
            self.embedding = self._get_embedding()
        else:
            self.embedding = embedding
            self.sim_mat = sim_mat
        self.shuffled_sim_mat = self._get_shuffled_sim_mat()

    def _get_sim_mat(self):
        if self.metric == "dot":
            metric = "cosine"
        else:
            metric = self.metric
        return distance.cdist(self.embedding, self.embedding, metric = metric)
    
    def _get_shuffled_sim_mat(self):
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
    
    def _get_embedding(self):
        MDS_embedding = manifold.MDS(n_components = 3, dissimilarity = 'precomputed', random_state = 0)
        embedding = MDS_embedding.fit_transform(self.sim_mat)
        return embedding
        
    def show_sim_mat(self, ticks_size = None, label = None, fig_dir = None):
        fig_path = os.path.join(fig_dir, f"RDM_{self.name}.png") if fig_dir is not None else None
        visualize_functions.show_heatmap(self.sim_mat, title = self.name, ticks_size = ticks_size, xlabel = label, ylabel = label, file_name = fig_path)
        
    def show_sim_mat_distribution(self):
        lower_triangular = np.tril(self.sim_mat)
        lower_triangular = lower_triangular.flatten()
        plt.hist(lower_triangular)
        plt.title(f"Distribution of RDM ({self.name})")
        plt.show()
        
    def show_embedding(self, dim = 3):
        visualize_embedding = visualize_functions.Visualize_Embedding(embedding_list = [self.embedding], name_list = [self.name])
        visualize_embedding.plot_embedding(dim = dim)
    
    
class Pairwise_Analysis:
    """
    A class object that has methods conducting gw-alignment and corresponding results
    This object has information of a pair of Representations.
    """
    
    def __init__(self, target : Representation, source : Representation, config : Optimization_Config) -> None:
        """
        Args:
            source (Representation): instance of Representation
            target (Representation): instance of Representation
            config (Optimization_Config) : instance of Optimization_Config
        """
        self.target = target
        self.source = source
        self.config = config
        self.OT = 0
        self.shuffled_OT = 0
        self.top_k_accuracy = pd.DataFrame()
        self.k_nearest_matching_rate = pd.DataFrame()
        self.pair_name = f"{target.name} vs {source.name}"
    
    def RSA(self, shuffle = False, metric = "spearman"):
        RDM_source, RDM_target = (self.source.sim_mat, self.target.sim_mat) if not shuffle else (self.source.shuffled_sim_mat, self.target.shuffled_sim_mat)
        upper_tri_source = RDM_source[np.triu_indices(RDM_source.shape[0], k=1)]
        upper_tri_target = RDM_target[np.triu_indices(RDM_target.shape[0], k=1)]
        if metric == "spearman":
            corr, _ = spearmanr(upper_tri_source, upper_tri_target)
        elif metric == "pearson":
            corr, _ = pearsonr(upper_tri_source, upper_tri_target)
        return corr
    
    def match_sim_mat_distribution(self):
        pass
    
    def run_gw(self, shuffle, ticks_size = None, load_OT = False, fig_dir = None):
        """
        Main computation
        """            
        if shuffle:     
                self.shuffled_OT = self._gw_alignment(shuffle = shuffle, load_OT = load_OT)
        else:
                self.OT = self._gw_alignment(shuffle = shuffle, load_OT = load_OT)
        self._show_OT(title = f"$\Gamma$ ({self.pair_name}) {'(shuffle)' if shuffle else ''} ", shuffle = shuffle, ticks_size = ticks_size, fig_dir = fig_dir)
        
    def _gw_alignment(self, results_dir = "../results/", shuffle = False, load_OT = False):
        filename = self.config.data_name + " " + self.pair_name
        RDM_source, RDM_target = (self.source.sim_mat, self.target.sim_mat) if not shuffle else (self.source.shuffled_sim_mat, self.target.shuffled_sim_mat)
        OT = run_main_gw(self.config, RDM_source, RDM_target, results_dir, filename, load_OT)
        return OT
          
    def _get_optimization_log(self):
        pass
    
    def _show_OT(self, title, shuffle : bool, ticks_size = None, fig_dir = None):
        OT = self.OT if not shuffle else self.shuffled_OT
        fig_path = os.path.join(fig_dir, f"OT_{self.pair_name}.png") if fig_dir is not None else None
        visualize_functions.show_heatmap(matrix = OT, title = title, ticks_size = ticks_size, file_name = fig_path)
    
    def calc_top_k_accuracy(self, k_list, shuffle : bool):
        OT = self.OT if not shuffle else self.shuffled_OT
        name = self.pair_name if not shuffle else self.pair_name + " shuffle"
        
        self.top_k_accuracy["top_n"] = k_list
        acc_list = self._eval_accuracy(OT = OT, k_list = k_list, eval_type = "ot_plan")
        self.top_k_accuracy[name] = acc_list
    
    def calc_k_nearest_matching_rate(self, k_list, metric):
        self.k_nearest_matching_rate["top_n"] = k_list
        acc_list = self._eval_accuracy(OT = self.OT, k_list = k_list, eval_type = "k_nearest", metric = metric)
        self.k_nearest_matching_rate[self.pair_name] = acc_list
        
    def _eval_accuracy(self, OT, k_list, eval_type = "ot_plan", supervised = False, metric = "cosine"):
        top_n_list = k_list

        OT = OT if not supervised else np.diag([1/len(self.target.sim_mat) for i in range(len(self.target.sim_mat))])
        acc_list = list()
        for k in top_n_list:
            if eval_type == "k_nearest":
                Q, new_embedding_source = procrustes(self.target.embedding, self.source.embedding, OT)
                acc = evaluation.pairwise_k_nearest_matching_rate(self.target.embedding, new_embedding_source, top_n = k, metric = metric)
            elif eval_type == "ot_plan":
                acc = evaluation.calc_correct_rate_ot_plan(OT, top_n = k)
            acc_list.append(acc)
        return acc_list
    
    def procrustes(self):
        Q, self.source.embedding = procrustes(self.target.embedding, self.source.embedding, self.OT)
        
    def visualize_embedding(self, dim = 3, category_name_list = None, category_num_list = None, category_idx_list = None):
        self.procrustes()
        embedding_list = [self.target.embedding, self.source.embedding]
        name_list = [self.target.name, self.source.name]
        
        visualize_embedding = visualize_functions.Visualize_Embedding(embedding_list = embedding_list, name_list = name_list, category_name_list = category_name_list, category_num_list = category_num_list, category_idx_list = category_idx_list)
        visualize_embedding.plot_embedding(dim = dim)
  
  
class Align_Representations:
    """
    This object has methods for conducting N groups level analysis and corresponding results.
    This has information of all pairs of representations.
    """
    def __init__(self, representations_list : List[Representation], config : Optimization_Config) -> None:
        """
        Args:
            representations_list (list): a list of Representations
        """
        self.representations_list = representations_list
        self.config = config
        self.pairwise_list = self._get_pairwise_list()
        
        self.RSA_corr = dict()
        self.top_k_accuracy = pd.DataFrame()
        self.k_nearest_matching_rate = pd.DataFrame()

    def _get_pairwise_list(self) -> List[Pairwise_Analysis]:
        pairs = list(itertools.combinations(self.representations_list, 2))
        pairwise_list = list()
        for i, pair in enumerate(pairs):
            pairwise = Pairwise_Analysis(target = pair[0], source = pair[1], config = self.config)
            pairwise_list.append(pairwise)    
            print(f"Pair number {i} : {pairwise.pair_name}")
        return pairwise_list

    def RSA_get_corr(self, shuffle = False, metric = "spearman"):
        for pairwise in self.pairwise_list:
            corr = pairwise.RSA(shuffle = shuffle, metric = metric)
            self.RSA_corr[pairwise.pair_name] = corr
            print(f"Correlation {pairwise.pair_name} : {corr}")
    
    def show_sim_mat(self, ticks_size = None, label = None, fig_dir = None, show_distribution = True):
        for representation in self.representations_list:
            representation.show_sim_mat(ticks_size = ticks_size, label = label, fig_dir = fig_dir)
            if show_distribution:
                representation.show_sim_mat_distribution()
    
    def gw_alignment(self, pairnumber_list = "all", shuffle = False, ticks_size = None, load_OT = False, fig_dir = None):
        if pairnumber_list == "all":
            pairnumber_list = [i for i in range(len(self.pairwise_list))]
        self.pairnumber_list = pairnumber_list
        for pairnumber in self.pairnumber_list:
            pairwise = self.pairwise_list[pairnumber]
            pairwise.run_gw(shuffle = shuffle, ticks_size = ticks_size, load_OT = load_OT, fig_dir = fig_dir)
            
    def barycenter_alignment(self):
        pass
       
    def calc_top_k_accuracy(self, k_list : int, shuffle : bool):
        self.top_k_accuracy["top_n"] = k_list
        for pairnumber in self.pairnumber_list:
            pairwise = self.pairwise_list[pairnumber]
            pairwise.calc_top_k_accuracy(k_list, shuffle = shuffle)
            #if shuffle:
            #    pairwise.calc_top_k_accuracy(k_list, shuffle = True)
            self.top_k_accuracy = pd.merge(self.top_k_accuracy, pairwise.top_k_accuracy, on = "top_n")
        print("Top k accuracy : \n", self.top_k_accuracy)
        print("Mean : \n", self.top_k_accuracy.iloc[:, 1:].mean(axis = "columns"))
            
    def calc_k_nearest_matching_rate(self, k_list, metric):
        self.k_nearest_matching_rate["top_n"] = k_list
        for pairnumber in self.pairnumber_list:
            pairwise = self.pairwise_list[pairnumber]
            pairwise.calc_k_nearest_matching_rate(k_list, metric)
            self.k_nearest_matching_rate = pd.merge(self.k_nearest_matching_rate, pairwise.k_nearest_matching_rate, on = "top_n")
        print("K nearest matching rate : \n", self.k_nearest_matching_rate)
        print("Mean : \n", self.k_nearest_matching_rate.iloc[:, 1:].mean(axis = "columns"))
    
    def _get_dataframe(self, eval_type = "ot_plan", shuffle = False, concat = True):
        df = self.top_k_accuracy if eval_type == "ot_plan" else self.k_nearest_matching_rate         
        df = df.set_index("top_n")
        if not shuffle:
            cols = [col for col in df.columns if "shuffle" not in col and "top_n" not in col]
        else:
            cols = [col for col in df.columns if "shuffle" in col]
        df = df[cols]
        if concat:
            df = pd.concat([df[i] for i in df.columns], axis = 0)
            df = df.rename("matching rate")
        return df
        
    def plot_accuracy(self, eval_type = "ot_plan", shuffle = False, fig_dir = None, scatter = True):
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
            plt.savefig(os.path.join(fig_dir, f"Accuracy_{eval_type}.png"))
        plt.show()
    
    def visualize_embedding(self, dim = 3, color_labels = None, category_name_list = None, category_num_list = None, category_idx_list = None, fig_dir = None):
        for i in range(len(self.pairwise_list) // 2):
            pair = self.pairwise_list[i]
            pair.procrustes()
        embedding_list = [self.representations_list[i].embedding for i in range(len(self.representations_list))]
        name_list = [self.representations_list[i].name for i in range(len(self.representations_list))]
        fig_path = os.path.join(fig_dir, "Aligned_embedding.png") if fig_dir is not None else None
        
        visualize_embedding = visualize_functions.Visualize_Embedding(embedding_list = embedding_list, name_list = name_list, color_labels = color_labels, category_name_list = category_name_list, category_num_list = category_num_list, category_idx_list = category_idx_list)
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
        representation = Representation(name = name, embedding = embedding, metric = metric)
        representations.append(representation)
    #%%
    '''
    Unsupervised alignment between Representations
    '''
    align_representations = Align_Representations(representations_list = representations, config = Optimization_Config())
    #%%
    # RSA
    align_representations.show_sim_mat()
    align_representations.RSA_get_corr(shuffle = False)
    
    #%%
    # Run gw
    align_representations.gw_alignment(shuffle = False, load_OT = True)
    #%%
    '''
    Evaluate the accuracy
    '''
    ## Accuracy of the optimized OT matrix
    align_representations.calc_top_k_accuracy(k_list = [1, 5, 10], shuffle = False)
    align_representations.plot_accuracy(eval_type = "ot_plan", shuffle = False, scatter = True)
    #%%
    ## Matching rate of k-nearest neighbors 
    align_representations.calc_k_nearest_matching_rate(k_list = [1, 5, 10], metric = metric)
    align_representations.plot_accuracy(eval_type = "k_nearest", shuffle = False, scatter = True)
    
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
