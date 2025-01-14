# %% [markdown]
#  # Tutorial for Gromov-Wassserstein unsupervised alignment 

# %%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../../'))

import numpy as np
import pandas as pd
import pickle as pkl
import torch

from src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig

# %% [markdown]
# # Step1: Prepare dissimilarity matrices or embeddings from the data
# First, you need to prepare dissimilarity matrices or embeddings from your data.  
# To store dissimilarity matrices or embeddings, an instance of the class `Representation` is used.   
# Please put your dissimilarity matrices or embeddings into the variables `sim_mat` or `embedding` in this instance.   
# 
# ## Load data
# `simulation`: Synthetic data illustrating difference between supervised an unsupervised alignment    

# n_clusters_list = [4, 8, 16]
symmetry_levels = ['high', 'medium', 'low']
init_mat_plan_list = ["uniform", "random"]

# for n_clusters in n_clusters_list:
for symmetry_level in symmetry_levels:
    for init_mat_plan in init_mat_plan_list:
        if init_mat_plan == "uniform":
            sampler_name_list = ["grid"]
        else:
            sampler_name_list = ["tpe", "grid"]
        
        for sampler_name in sampler_name_list:
                        
            # %%
            # list of representations where the instances of "Representation" class are included
            representations = list()

            # select data : "simulation", "AllenBrain", "THINGS", "DNN", "color"
            data_select = "simulation"


            # %% [markdown]
            # ### Dataset No.1 `difficulty`

            # %%
            # correlation = 0.8 # 0.9, 0.8, 0.7

            # data_select += f"_corr{correlation}"
            # data_select += f"_clusters{n_clusters}"
            data_select += f"_{symmetry_level}_symmetry"

            n_representations = 2 # Set the number of the instanses of "Representation". This number must be equal to or less than the number of the groups. 2 is the maximum for this data.
            metric = "euclidean" # Please set the metric that can be used in "scipy.spatical.distance.cdist()".

            for i in range(n_representations):
                name = f"Group{i+1}" # the name of the representation
                # embedding = np.load(f"../../data/difficulty/corr{correlation}/embedding_{i+1}.npy") # the dissimilarity matrix will be computed with this embedding based on the metric
                # embedding = np.load(f"../../data/difficulty/clusters{n_clusters}/embedding_{i+1}.npy") # the dissimilarity matrix will be computed with this embedding based on the metric
                embedding = np.load(f"../../data/difficulty/{symmetry_level}_symmetry/embedding_{i+1}.npy") # the dissimilarity matrix will be computed with this embedding based on the metric

                representation = Representation(
                    name=name,
                    embedding=embedding,
                    metric=metric,
                    get_embedding=False, # If there is the embeddings, plese set this variable "False".
                )
                
                representations.append(representation)

            # %% [markdown]
            # # Step 2: Set the parameters for the optimazation of GWOT
            # Second, you need to set the parameters for the optimization of GWOT.    
            # For most of the parameters, you can start with the default values.   
            # However, there are some essential parameters that you need to check for your original applications.  
            # 
            # ## Optimization Config  
            # 
            # #### Most important parameters to check for your application:
            # `eps_list`: The range of the values of epsilon for entropic GWOT.   
            # If epsilon is not in appropriate ranges (if it is too low), the optimization may not work properly.   
            # Although the algorithm will find good epsilon values after many trials, it is a good practice to narrow down the range beforehand.   
            # 
            # `num_trial`: The number of trials to test epsilon values from the specified range.   
            # This number directly determines the quality of the unsupervised alignment.   
            # You should set this number high enough to find good local minima. 

            # %%
            eps_list_tutorial = [1e-2, 10]
            device = 'cpu'
            to_types = 'numpy'
            multi_gpu = False

            # whether epsilon is sampled at log scale or not
            eps_log = True

            # set the number of trials, i.e., the number of epsilon values evaluated in optimization. default : 4
            num_trial = 100

            main_results_dir = "../../results/" + data_select
            if sampler_name == "grid" and init_mat_plan == "random":
                main_results_dir = main_results_dir + f"/random+grid/"

            # %%
            config = OptimizationConfig(    
                eps_list = eps_list_tutorial,
                eps_log = eps_log,
                num_trial = num_trial,
                sinkhorn_method='sinkhorn', 
                
                ### Set the device ('cuda' or 'cpu') and variable type ('torch' or 'numpy')
                to_types = to_types, # user can choose "numpy" or "torch". please set "torch" if one wants to use GPU.
                device = device, # "cuda" or "cpu"; for numpy, only "cpu" can be used. 
                data_type = "double", # user can define the dtypes both for numpy and torch, "float(=float32)" or "double(=float64)". For using GPU with "sinkhorn", double is storongly recommended.
                
                ### Parallel Computation (requires n_jobs > 1, available both for numpy and torch)
                n_jobs = 1, # n_jobs : the number of worker to compute. if n_jobs = 1, normal computation will start. "Multithread" is used for Parallel computation.
                multi_gpu = multi_gpu, # This parameter is only for "torch". # "True" : all the GPU installed in your environment are used, "list (e.g.[0,2,3])"" : cuda:0,2,3, and "False" : single gpu (or cpu for numpy) will use.

                db_params={"drivername": "sqlite"},
                # db_params={"drivername": "mysql+pymysql", "username": "root", "password": "****", "host": "localhost"},
                
                ### Set the parameters for optimization
                # 'uniform': uniform matrix, 'diag': diagonal matrix', random': random matrix
                init_mat_plan = init_mat_plan,
                
                n_iter = 1,
                max_iter = 1000,
                
                sampler_name = sampler_name,
                pruner_name = 'hyperband',
                pruner_params = {'n_startup_trials': 1, 
                                'n_warmup_steps': 2, 
                                'min_resource': 2, 
                                'reduction_factor' : 3
                                },
            )

            # %% [markdown]
            # ## Step 3 : Gromov-Wasserstein Optimal Transport (GWOT) between Representations
            # Third, you perform GWOT between the instanses of "Representation", by using the class `AlignRepresentations`.  
            # This class has methods for the optimization of entropic Gromov-Wasserstein distance, and the evaluation of the GWOT (Step 4).  
            # This class also has a method to perform conventional Representation Similarity Analysis (RSA).   

            # %%
            # Create an "AlignRepresentations" instance
            align_representation = AlignRepresentations(
                config=config,
                representations_list=representations,   
            
                # histogram matching : this will adjust the histogram of target to that of source.
                histogram_matching=False,

                # main_results_dir : folder or file name when saving the result
                main_results_dir = main_results_dir,
            
                # data_name : Please rewrite this name if users want to use their own data.
                data_name = data_select,
            )

            # %% [markdown]
            # ## Show dissimilarity matrices

            # %%
            sim_mat_format = "default"

            visualize_config = VisualizationConfig(
                figsize=(8, 6), 
                title_size = 15, 
                ot_object_tick=True,
            )

            visualize_hist = VisualizationConfig(figsize=(8, 6), color='C0')

            sim_mat = align_representation.show_sim_mat(
                sim_mat_format = sim_mat_format, 
                visualization_config = visualize_config,
                visualization_config_hist = visualize_hist,
                show_distribution=False, # if True, the histogram figure of the sim_mat will be shown. visualization_config_hist will be used for adjusting this figure.
            )

            # %% [markdown]
            # ## Reperesentation Similarity Aanalysis (RSA)
            # This performs a conventional representation similarity analysis.

            # %%
            ### parameters for computing RSA
            # metric = "pearson" or "spearman" by scipy.stats
            # The result of RSA for each pair will be stored in align_representation.RSA_corr
            align_representation.RSA_get_corr(metric = "pearson")

            # print(align_representation.RSA_corr)

            # %% [markdown]
            # ## GWOT
            # The optimization results are saved in the folder named "config.data_name" + "representations.name" vs "representation.name".  
            # If you want to change the name of the saved folder, please make changes to "config.data_name" and "representations.name" (or change the "filename" in the code block below).

            # %% [markdown]
            # GWOT is performed by appling the method `gw_alignment` to the instance of `AlignRepresentations` class.
            # 
            # We show all the parameters to run GWOT computation as an example with THINGS or DNN dataset because these dataset have category information label.
            # 
            # For the dataset of color, AllenBrain, and simulation (these doesn’t have the category information), we show how to do this in next cell. 

            # %% [markdown]
            # Here is the example to compute the GWOT for each pair for color, AllenBrain, and simulation datasets below.

            # %%
            compute_OT = True

            # %%
            visualize_config = VisualizationConfig(
                show_figure=True,
                figsize=(8, 6), 
                title_size = 15, 
                ot_object_tick=True,
                plot_eps_log=eps_log,
            )

            sim_mat_format = "default"

            align_representation.gw_alignment(
                compute_OT = compute_OT,
                delete_results = True,
                return_data = False,
                return_figure = False,
                OT_format = sim_mat_format,
                visualization_config = visualize_config,
                delete_confirmation=False,
            )

            # %% [markdown]
            # # Step 4: Evaluation and Visualization
            # Finally, you can evaluate and visualize the unsupervise alignment of GWOT.   

            # %% [markdown]
            # ## Show how the GWD was optimized
            # `show_optimization_log` will make two figures to show both the relationships between epsilons (x-axis) and GWD (y-axis), and between accuracy (x-axis) and GWD (y-axis).
            # 
            # 

            # %%
            ### Show how the GWD was optimized (evaluation figure)
            # show both the relationships between epsilons and GWD, and between accuracy and GWD
            align_representation.show_optimization_log(fig_dir=None, visualization_config=visualize_config) 

            # %% [markdown]
            # ## Evaluation of the accuracy of the unsupervised alignment
            # There are two ways to evaluate the accuracy.  
            # 1. Calculate the accuracy based on the OT plan. 
            # - For using this method, please set the parameter `eval_type = "ot_plan"` in "calc_accuracy()".
            #   
            # 2. Calculate the matching rate based on the k-nearest neighbors of the embeddings.
            # -  For using this method, please set the parameter `eval_type = "k_nearest"` in "calc_accuracy()".
            # 
            # For both cases, the accuracy evaluation criterion can be adjusted by considering "top k".  
            # By setting "top_k_list", you can observe how the accuracy increases as the criterion is relaxed.

            # %%
            ## Calculate the accuracy based on the OT plan. 
            align_representation.calc_accuracy(top_k_list = [1, 5, 10], eval_type = "ot_plan")
            align_representation.plot_accuracy(eval_type = "ot_plan", scatter = True)

            top_k_accuracy = align_representation.top_k_accuracy # you can get the dataframe directly 

            # %%
            ## Calculate the matching rate based on the k-nearest neighbors of the embeddings.
            align_representation.calc_accuracy(top_k_list = [1, 5, 10], eval_type = "k_nearest", metric=metric)
            align_representation.plot_accuracy(eval_type = "k_nearest", scatter = True)

            k_nearest_matching_rate = align_representation.k_nearest_matching_rate # you can get the dataframe directly 

            # %% [markdown]
            # ## Procrustes Analysis
            # Using optimized transportation plans, you can align the embeddings of each representation to a shared space in an unsupervised manner.  
            # The `"pivot"` refers to the target embeddings space to which the other embeddings will be aligned.   
            # You have the option to designate the `"pivot"` as one of the representations or the barycenter.  
            # Please ensure that 'pair_number_list' includes all pairs between the pivot and the other Representations.  
            # 
            # If you wish to utilize the barycenter, please make use of the method `AlignRepresentation.barycenter_alignment()`.  
            # You can use it in the same manner as you did with `AlignRepresentation.gw_alignment()`.

            # %%
            emb_name = "PCA" #"TSNE", "PCA", "MDS"

            visualization_embedding = VisualizationConfig(
                fig_ext='svg',
                figsize=(9, 9), 
                xlabel="PC1",
                ylabel="PC2", 
                marker_size=100,
                xlabel_size=40,
                ylabel_size=40,
                legend_size=20,
            )

            align_representation.visualize_embedding(
                dim=2, # the dimensionality of the space the points are embedded in. You can choose either 2 or 3.
                pivot=0, # the number of one of the representations or the "barycenter".
                visualization_config=visualization_embedding
            )

            # %% [markdown]
            # ## (Option) Visualize the aligned embeddings
            # It is helpful to visually inspect the quality of the unsupervised alignment by plotting the aligned embeddings in 2D or 3D space. 
            # 
            # Using the optimal transportation plan $\Gamma∗$, one set of embeddings $X$, 
            # and the other set of aligned embeddings are obtained by performing matrix product of the $\Gamma*$ and $X$. 
            # 
            # This $\Gamma* X$ means the new embedding to represent the other set of embedding $Y$ in the space of $X$. 
            # 
            # Next, the embedding of $X$ and new embedding of $Y$ ($Γ∗X$) in the space of $X$ are concatenated in the axis of the number of embeddings (e.g. experimental stimuli, and so on).  
            # Then, for example, Principle Components Analysis (PCA) is used to project high-dimensional embeddings into 2D or 3D space, but other dimensionality reduction methods are also applicable. 
            # 
            # And, each one of the two embeddings were plotted in 3D figure after splitting the data into the two embeddings. 
            # If there is a known correspondence between the embeddings, the user can visually check whether the aligned embeddings are positioned close to the corresponding embeddings.

            # %%
            pair = align_representation.pairwise_list[0]

            ot = pair.OT

            # plt.imshow(ot, cmap='viridis')
            # plt.show()

            # %%
            source = pair.source.embedding
            target = pair.target.embedding

            new_source = pair.OT.T @ target * len(target)
            new_target = pair.OT @ source * len(source)

            # %%
            new_rep_list = [
                Representation(name="Group1", embedding=source),
                Representation(name="Group2", embedding=new_target),
            ]

            ar = AlignRepresentations(
                config=config,
                representations_list=new_rep_list,
                histogram_matching=False,
                # metric="cosine",
                main_results_dir="../../results/" + data_select,
                data_name=data_select,
            )

            # %%
            emb_name = "PCA" #"TSNE", "PCA", "MDS"

            visualization_embedding = VisualizationConfig(
                fig_ext='svg',
                figsize=(9, 9), 
                xlabel="PC1",
                ylabel="PC2", 
                marker_size=100,
                xlabel_size=40,
                ylabel_size=40,
                legend_size=20,
            )

            ar.visualize_embedding(
                dim=2, # the dimensionality of the space the points are embedded in. You can choose either 2 or 3.
                method=emb_name, # the method used to embed the points. You can choose either "PCA", "TSNE", or "MDS".
                pivot=None, 
                visualization_config=visualization_embedding
            )


