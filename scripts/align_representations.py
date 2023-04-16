#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.spatial import distance
from sklearn.manifold import MDS
import ot
import sys
import os
from typing import List

from src.utils.utils_functions import procrustes, shuffle_RDM, RSA_get_corr, get_category_idx, shuffle_symmetric_block_mat
from src.utils.visualize_functions import show_heatmap, Visualize_Embedding
from src.utils.evaluation import pairwise_k_nearest_matching_rate, calc_correct_rate_ot_plan
from src.gw_alignment import GW_Alignment
from src.utils.gw_optimizer import load_optimizer
from src.utils.init_matrix import InitMatrix


class Optimization_Config:
    def __init__(self, 
                 delete_study = True, 
                 device = 'cpu',
                 to_types = 'numpy',
                 n_jobs = 4,
                 init_plans_list = ['random'],
                 num_trial = 4,
                 n_iter = 1,
                 max_iter = 200,
                 sampler_name = 'tpe',
                 eps_list = [1, 10],
                 eps_log = True,
                 pruner_name = 'hyperband',
                 pruner_params = {'n_startup_trials': 1, 'n_warmup_steps': 2, 'min_resource': 2, 'reduction_factor' : 3}
                 ) -> None:
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


class Representation:
    """
    A class object that has information of a representation, such as embeddings and similarity matrices
    """
    def __init__(self, name, sim_mat = None, embedding = None, metric = "cosine") -> None:
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
        return shuffle_RDM(self.sim_mat)
    
    def _get_embedding(self):
        MDS_embedding = MDS(n_components = 3, dissimilarity = 'precomputed', random_state = 0)
        embedding = MDS_embedding.fit_transform(self.sim_mat)
        return embedding
        
    def show_sim_mat(self, ticks_size = None, label = None):
        plt.figure(figsize = (20, 20))
        ax = sns.heatmap(self.sim_mat, square = True, cbar_kws = {"shrink": .80})
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize = ticks_size)    
        plt.xlabel(label, size = 40)
        plt.ylabel(label, size = 40)
        plt.title(self.name, size = 60)
        plt.show()
        
    def show_sim_mat_distribution(self):
        pass
    
    def show_embedding(self, dim = 3):
        visualize_embedding = Visualize_Embedding(embedding_list = [self.embedding], name_list = [self.name])
        visualize_embedding.plot_embedding(dim = dim)
    
    def procrustes(self, target, Pi):
        Q, self.embedding = procrustes(target.embedding, self.embedding, Pi)  
    
    
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
        """
        self.target = target
        self.source = source
        self.config = config
        self.OT = 0
        self.shuffled_OT = 0
        self.top_k_accuracy = pd.DataFrame()
        self.k_nearest_matching_rate = pd.DataFrame()
        self.pair_name = f"{target.name} vs {source.name}"
    
    def RSA(self, shuffle = False):
        if not shuffle:
            corr = RSA_get_corr(self.source.sim_mat, self.target.sim_mat)
        else:
            corr = RSA_get_corr(self.source.shuffled_sim_mat, self.target.shuffled_sim_mat)
        return corr
    
    def match_sim_mat_distribution(self):
        pass
    
    def run_gw(self, shuffle, ticks_size = None, load_OT = False):
        """
        Main computation
        """            
        if shuffle:     
                self.shuffled_OT = self._gw_alignment(shuffle = shuffle, load_OT = load_OT)
        else:
                self.OT = self._gw_alignment(shuffle = shuffle, load_OT = load_OT)
        self._show_OT(title = f"$\Gamma$ ({self.pair_name}) {'(shuffle)' if shuffle else ''} ", shuffle = shuffle, ticks_size = ticks_size)
        
    def _gw_alignment(self, shuffle : bool, load_OT = False):
        filename = self.pair_name
        save_path = '../results/gw_alignment/' + filename
        sql_name = 'sqlite'
        storage = "sqlite:///" + save_path +  '/' + filename + '.db'
        RDM_source, RDM_target = (self.source.sim_mat, self.target.sim_mat) if not shuffle else (self.source.shuffled_sim_mat, self.target.shuffled_sim_mat)

        # distribution in the source space, and target space
        p = ot.unif(len(RDM_source))
        q = ot.unif(len(RDM_target))
        save_path = "../results/" + filename

        # generate instance solves gw_alignment　
        test_gw = GW_Alignment(RDM_source, RDM_target, p, q, save_path, max_iter = self.config.max_iter, n_iter = self.config.n_iter, to_types = self.config.to_types)

        # generate instance optimize gw_alignment　
        opt = load_optimizer(save_path,
                                n_jobs = self.config.n_jobs,
                                num_trial = self.config.num_trial,
                                to_types = self.config.device,
                                method = 'optuna',
                                sampler_name = self.config.sampler_name,
                                pruner_name = self.config.pruner_name,
                                pruner_params = self.config.pruner_params,
                                n_iter = self.config.n_iter,
                                filename = filename,
                                sql_name = sql_name,
                                storage = storage,
                                delete_study = False
        )

        ### optimization
        # 1. 初期値の選択。実装済みの初期値条件の抽出をgw_optimizer.pyからinit_matrix.pyに移動しました。
        init_plans = InitMatrix().implemented_init_plans(self.config.init_plans_list)

        # used only in grid search sampler below the two lines
        eps_space = opt.define_eps_space(self.config.eps_list, self.config.eps_log, self.config.num_trial)
        search_space = {"eps": eps_space, "initialize": init_plans}
        if not load_OT:
            # 2. run optimzation
            study = opt.run_study(test_gw, self.config.device, init_plans_list = init_plans, eps_list = self.config.eps_list, eps_log = self.config.eps_log, search_space = search_space)
            best_trial = study.best_trial
            OT = np.load(save_path+f'/{self.config.init_plans_list[0]}/gw_{best_trial.number}.npy')
        else:
            study = opt.load_study()
            best_trial = study.best_trial
            OT = np.load(save_path+f'/{self.config.init_plans_list[0]}/gw_{best_trial.number}.npy')
        return OT
          
    def _get_optimization_log(self):
        pass
    
    def _show_OT(self, title, shuffle : bool, ticks_size = None):
        OT = self.OT if not shuffle else self.shuffled_OT
        plt.figure(figsize = (20, 20))
        ax = sns.heatmap(OT, square = True, cbar_kws = {"shrink": .80})
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize = ticks_size)    
        plt.xlabel(self.target.name, size = 40)
        plt.ylabel(self.source.name, size = 40)
        plt.title(title, size = 60)
        plt.show()
    
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

        OT = OT if not supervised else np.diag([1/1854 for i in range(1854)])
        acc_list = list()
        for k in top_n_list:
            if eval_type == "k_nearest":
                Q, new_embedding_source = procrustes(self.target.embedding, self.source.embedding, OT)
                acc = pairwise_k_nearest_matching_rate(self.target.embedding, new_embedding_source, top_n = k, metric = metric)
            elif eval_type == "ot_plan":
                acc = calc_correct_rate_ot_plan(OT, top_n = k)
            acc_list.append(acc)
        return acc_list
    
    def procrustes(self):
        self.source.procrustes(self.target, self.OT)
        
    def visualize_embedding(self, dim = 3, category_name_list = None, category_num_list = None, category_idx_list = None):
        self.procrustes()
        embedding_list = [self.target.embedding, self.source.embedding]
        name_list = [self.target.name, self.source.name]
        
        visualize_embedding = Visualize_Embedding(embedding_list = embedding_list, name_list = name_list, category_name_list = category_name_list, category_num_list = category_num_list, category_idx_list = category_idx_list)
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
        for pair in pairs:
            pairwise = Pairwise_Analysis(target = pair[0], source = pair[1], config = self.config)
            pairwise_list.append(pairwise)
        return pairwise_list

    def RSA_get_corr(self, shuffle = False):
        for pairwise in self.pairwise_list:
            corr = pairwise.RSA(shuffle)
            self.RSA_corr[pairwise.pair_name] = corr
            print(f"Correlation {pairwise.pair_name} : {corr}")
    
    def show_sim_mat(self, ticks_size = None, label = None, title = None):
        for representation in self.representations_list:
            representation.show_sim_mat(ticks_size = ticks_size, label = label)
    
    def gw_alignment(self, shuffle : bool, ticks_size = None, load_OT = False):
        for pairwise in self.pairwise_list:
            pairwise.run_gw(shuffle = shuffle, ticks_size = ticks_size, load_OT = load_OT)
            
    def barycenter_alignment(self):
        pass
       
    def calc_top_k_accuracy(self, k_list : int, shuffle : bool):
        self.top_k_accuracy["top_n"] = k_list
        for pairwise in self.pairwise_list:
            pairwise.calc_top_k_accuracy(k_list, shuffle = shuffle)
            #if shuffle:
            #    pairwise.calc_top_k_accuracy(k_list, shuffle = True)
            self.top_k_accuracy = pd.merge(self.top_k_accuracy, pairwise.top_k_accuracy, on = "top_n")
        print("Top k accuracy : \n", self.top_k_accuracy)
        print("Mean : \n", self.top_k_accuracy.iloc[:, 1:].mean(axis = "columns"))
            
    def calc_k_nearest_matching_rate(self, k_list, metric):
        self.k_nearest_matching_rate["top_n"] = k_list
        for pairwise in self.pairwise_list:
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
        
    def plot_accuracy(self, eval_type = "ot_plan", shuffle = False, save_fig = True, scatter = True):
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
                         
        #if shuffle:    
        #    df = self._get_dataframe(eval_type, shuffle = True, concat = False)
        #    y_mean = df.mean(axis = "columns")
        #    y_std = df.std(axis = "columns")
        #    plt.plot(df.index, y_mean, label = "shuffle", linewidth = 2)
        #    plt.fill_between(df.index, (y_mean - y_std), (y_mean + y_std), alpha = 0.3)
        #plt.xticks(ticks = [i + 1 for i in range(10)], labels = [i + 1 for i in range(10)])
        plt.ylim(0, 100)
        plt.xlabel("k")
        plt.ylabel("Matching rate")
        #plt.legend(loc = "best")
        plt.tick_params(axis = "both", which = "major")
        plt.subplots_adjust(left=0.2, right=0.9, bottom = 0.2)
        plt.show()
    
    def visualize_embedding(self, dim = 3, color_labels = None, category_name_list = None, category_num_list = None, category_idx_list = None):
        for i in range(len(self.pairwise_list) // 2):
            pair = self.pairwise_list[i]
            pair.procrustes()
        embedding_list = [self.representations_list[i].embedding for i in range(len(self.representations_list))]
        name_list = [self.representations_list[i].name for i in range(len(self.representations_list))]

        visualize_embedding = Visualize_Embedding(embedding_list = embedding_list, name_list = name_list, color_labels = color_labels, category_name_list = category_name_list, category_num_list = category_num_list, category_idx_list = category_idx_list)
        visualize_embedding.plot_embedding(dim = dim)

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
