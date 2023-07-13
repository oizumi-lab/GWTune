# %% [markdown]
#  # Tutorial for Gromov-Wassserstein unsupervised alignment 

# %%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../'))

import numpy as np
import pandas as pd
import pickle as pkl

from src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig

# %% [markdown]
# # Step1: Prepare dissimilarity matrices or embeddings from the data
# First, you need to prepare dissimilarity matrices or embeddings from your data.  
# 
# The unit of unsupervised alignment is an instance of the "Representation" class. This class has variables such as "name" and either "sim_mat" or "embedding". You need to assign values to these variables.  
# These instances are stored in "representations" and later passed to the "AlignRepresentations" class.
# 
# ## Load data
# You have the option to select the data from the following choices:
# 1. 'color': human similarity judgements of 93 colors for 5 paricipants groups
# 2. 'THINGS' : human similarity judgements of 1854 objects for 4 paricipants groups
# 
# "data_select" in next code block determines which data is being used.

# %%
# list of representations where the instances of "Representation" class are included
representations = list()

# select data
data_select = "color"
# data_select = "THINGS"

# %% [markdown]
# ### Dataset No1. `color`
# In this case, we directly assign the dissimilarity matrices of 93 colors to "Representation".

# %%
# Load data and create "Representation" instance
if data_select == 'color':
    n_representations = 5 # Set the number of the instanses of "Representation". This number must be equal to or less than the number of the groups. 5 is the maximum for this data.
    metric = "euclidean" # Please set the metric that can be used in "scipy.spatical.distance.cdist()".
    
    data_path = '../data/num_groups_5_seed_0_fill_val_3.5.pickle'
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    sim_mat_list = data["group_ave_mat"]
    for i in range(n_representations):
        name = f"Group{i+1}" # "name" will be used as a filename for saving the results
        sim_mat = sim_mat_list[i] # the dissimilarity matrix of the i-th group
        # make an instance "Representation" with settings 
        representation = Representation(
            name=name, 
            metric=metric,
            sim_mat=sim_mat,  #: np.ndarray
            embedding=None,   #: np.ndarray 
            get_embedding=True, # If true, the embeddings are computed from the dissimilarity matrix automatically using the MDS function. Default is False. 
            MDS_dim=3, # If "get embedding" is True, please set the dimensions of the embeddings.
            object_labels=None,
            category_name_list=None,
            num_category_list=None,
            category_idx_list=None,
            func_for_sort_sim_mat=None,
       ) 
        representations.append(representation)

# %% [markdown]
# ### Dataset No.2 `THINGS`
# In this case, we assign the embeddings of each object to "Representation". This class will compute the dissimilarity matrices with the embeddings.  
# Furthermore, this dataset includes coarse category labels for each objet, and we will now demonstrate how to utilize them.

# %%
if data_select == "THINGS":
    # define the coarce category labels
    category_mat = pd.read_csv("../data/category_mat_manual_preprocessed.csv", sep = ",", index_col = 0)
    
    # calculate the parameters for the coarce category labels
    # Please prepare equivalent parameters when using other datasets.
    from src.utils.utils_functions import get_category_data, sort_matrix_with_categories # get_category_data and sort_matrix_with_categories are functions specialied for this dataset
    object_labels, category_idx_list, num_category_list, category_name_list = get_category_data(category_mat)
    
    n_representations = 4 # Set the number of the instanses of "Representation". This number must be equal to or less than the number of the groups. 4 is the maximum for this data.
    metric = "euclidean" # Please set the metric that can be used in "scipy.spatical.distance.cdist()".
    
    for i in range(n_representations):
        name = f"Group{i+1}"
        embedding = np.load(f"../data/THINGS_embedding_Group{i+1}.npy")[0]
        
        representation = Representation(
            name=name, 
            embedding=embedding, # the dissimilarity matrix will be computed with this embedding.
            metric=metric,
            get_embedding=False, # If there is the embeddings, plese set this variable "False".
            object_labels=object_labels,
            category_name_list=category_name_list,
            category_idx_list=category_idx_list,
            num_category_list=num_category_list,
            func_for_sort_sim_mat=sort_matrix_with_categories
        )
        
        representations.append(representation)

# %% [markdown]
# # Step 2: Set the parameters for the optimazation of GWOT
# 
# ## Optimization Config  
# 
# #### Most important parameters to check for your application:
# `eps_list, num_trial` are essential for computing the GW alignment.  
# You need to choose the appropriate ranges of the epsilon, `eps_list`.  
# If the epsilon is not in the appropriate ranges, the optimization may not work properly.  
# Also, the epsilon range is critical for finding good local optimum.  
# 
# For other parameters, please start by trying the default values.

# %%
### Most important parameters
# Set the range of the epsilon
# set the minimum value and maximum value for 'tpe' sampler
# for 'grid' or 'random' sampler, you can also set the step size
if data_select == "THINGS":
    eps_list_tutorial = [1, 10]
if data_select == "color":
    eps_list_tutorial = [0.02, 0.2]
    
# whether epsilon is sampled at log scale or not
eps_log = True

# set the number of trials, i.e., the number of epsilon values evaluated in optimization. default : 4
num_trial = 30

### Set the parameters for optimization
# initialization of transportation plan
# 'uniform': uniform matrix, 'diag': diagonal matrix', random': random matrix, 'permutation': permutation matrix
# Select multiple options was deprecated.
init_mat_plan = "random"

# %% [markdown]
# ### if user wants to use some user-defined init matrices...
# For ”user_define”, it is note that all the initialization plans need to be written in Numpy even when PyTorch is used for the optimization.  
# The user can define a single or multiple plans before the optimization starts.

# %%
if init_mat_plan == "user_define":
    import ot
    size = representation.sim_mat.shape[0]
    user_define_mat_random = np.random.randn(size, size)
    user_define_mat_random = user_define_mat_random / user_define_mat_random.sum()
    user_define_init_mat_list = [user_define_mat_random, np.outer(ot.unif(size), ot.unif(size))]

else:
    user_define_init_mat_list = None

# %%
config = OptimizationConfig(
    
    eps_list = eps_list_tutorial, # [1, 10] for THINGS data, [0.02, 0.2] for colors data
    eps_log = eps_log, # whether epsilon is sampled at log scale or not
    num_trial = num_trial, # set the number of trials, i.e., the number of epsilon values evaluated in optimization. default : 4
    sinkhorn_method='sinkhorn', # please choose the method of sinkhorn implemented by POT (URL : https://pythonot.github.io/gen_modules/ot.bregman.html#id87). For using GPU, "sinkhorn_log" is recommended.
    
    ### Set the device ('cuda' or 'cpu') and variable type ('torch' or 'numpy')
    to_types = 'numpy', # user can choose "numpy" or "torch". please set "torch" if one wants to use GPU.
    device = 'cpu', # "cuda" or "cpu"; for numpy, only "cpu" can be used. 
    data_type = "double", # user can define the dtypes both for numpy and torch, "float(=float32)" or "double(=float64)". For using GPU with "sinkhorn", double is storongly recommended.
    
    ### Parallel Computation (requires n_jobs > 1, available both for numpy and torch)
    n_jobs = 1, # n_jobs : the number of worker to compute. if n_jobs = 1, normal computation will start. "Multithread" is used for Parallel computation.
    multi_gpu = True, # This parameter is only for "torch". # "True" : all the GPU installed in your environment are used, "list (e.g.[0,2,3])"" : cuda:0,2,3, and "False" : single gpu (or cpu for numpy) will use.
    
    ### Set the db_params to create database URL to store the optimization results (either PyMySQL or SQLite. For using PyMySQL, some additional setting beforehand will be needed).  
    # The database URL in sqlalchemy is like "dialect+driver://username:password@host:port/database". See the following page for details. https://docs.sqlalchemy.org/en/20/core/engines.html
    # If you want to use SQLite, it's enough to set "db_params={"drivername": "sqlite"}".
    # This package generates 1 database per each study.

    # db_params={"drivername": "sqlite"},
    db_params={"drivername": "mysql+pymysql", "username": "root", "password": "", "host": "localhost"},
    
    ### Set the parameters for optimization
    # 'uniform': uniform matrix, 'diag': diagonal matrix', random': random matrix, 'permutation': permutation matrix
    init_mat_plan = init_mat_plan,
    
    user_define_init_mat_list = user_define_init_mat_list,
    
    ### Set the parameters for optimization
    # n_iter : the number of random initial matrices for 'random' or 'permutation' options：default: 1
    # max_iter : the maximum number of iteration for GW optimization: default: 200
    n_iter = 1,
    max_iter = 200,
    
    ### folder or file name when saving the result
    # The ptimization results are saved in the folder named "config.data_name" + "representations.name" vs "representation.name".  
    # If you want to change the name of the saved folder, please make changes to "config.data_name" and "representations.name".
    data_name = data_select, # Please rewrite this name if users want to use their own data.
    
    ### choose sampler implemented by Optuna
    # 1. 'random': randomly select epsilon between the range of epsilon
    # 2. 'grid': grid search between the range of epsilon
    # 3. 'tpe': Bayesian sampling
    sampler_name = 'tpe',
    
    ### choose pruner
    # 1. 'median': Pruning if the score is below the past median at a certain point in time  
    #     n_startup_trials: Do not activate the pruner until this number of trials has finished  
    #     n_warmup_steps: Do not activate the pruner for each trial below this step  
        
    # 2. 'hyperband': Use multiple SuccessiveHalvingPrunerd that gradually longer pruning decision periods and that gradually stricter criteria  
    #     min_resource: Do not activate the pruner for each trial below this step  
    #     reduction_factor: How often to check for pruning. Smaller values result in more frequent pruning checks. Between 2 to 6.  
        
    # 3. 'nop': no pruning
    pruner_name = 'hyperband',
    pruner_params = {'n_startup_trials': 1, 
                     'n_warmup_steps': 2, 
                     'min_resource': 2, 
                     'reduction_factor' : 3
                    },
)

# %% [markdown]
# ## VisualizationConfig
# You can set the parameters for the visualization of the matrices or the embeddings.
# 
# Here, we aim to introduce all the parameters that will be used for this instance, keeping in mind that some of them may be modified later for each dataset.
# 
# Please keep in mind you can also get the raw results data if you want to make the figures by yourself.  

# %%
visualization_config = VisualizationConfig(
    ### Please set the parameters below that can be used in "mttplotlib.pyplot"
    figsize=(8, 6), 
    title_size = 15, 
    cmap = 'cividis',
    cbar_ticks_size=20,
    ticks_size=5,
    xticks_rotation=90,
    yticks_rotation=0,
    legend_size=5,
    xlabel=None,
    xlabel_size=15,
    ylabel=None,
    ylabel_size=15,
    zlabel=None,
    zlabel_size=15,
    color_labels=None,
    color_hue=None,
    markers_list=None,
    marker_size=30,
    
    ### Set the parameters for showing the boundary of the coarce category labels if the dataset have them. If not, please set draw_category_line = False.
    draw_category_line=True,
    category_line_color='C2',
    category_line_alpha=0.2,
    category_line_style='dashed',
    
    ### If you want to save the figure only, but don't show them, please set show_figure = False.
    show_figure = True,
)

# %% [markdown]
# ## Step 3 : Unsupervised alignment between Representations
# "AlignRepresentations" is the class for performing the unsupervised alignment among the instanses of "Representation".  
# This class has methods for Representation Similarity Analysis (RSA), Gromov-Wasserstein (GW) alignment, and the evaluation of the GW alignment.  
# 
# By default, the instance applis GW alignment to all pairs in the `representations` defined at the begining of this notebook.   
# If you want to limit the pairs to which GW alignment is applied, please set “AlignRepresentations.pair_number_list”. (e.g. pair_number_list = [[0, 1], [0, 2]])

# %%
# Create an "AlignRepresentations" instance
align_representation = AlignRepresentations(
    representations_list=representations,
    pair_number_list="all", # If you want to limit the pairs to which GW alignment is applied, please set “AlignRepresentations.pair_number_list”. (e.g. pair_number_list = [[0, 1], [0, 2]])
    histogram_matching=False,
    config=config,
    metric="cosine", # The metric for computing the distance between the embeddings. Please set the metric tha can be used in "scipy.spatical.distance.cdist()".
)

# %% [markdown]
# ## Show dissimilarity matrices

# %%
## Dataset No.1 : color 
if data_select == "color":
    sim_mat_format = "default"
    visualize_config = VisualizationConfig(figsize=(8, 6), title_size = 15)
    visualize_hist = VisualizationConfig(figsize=(8, 6), cmap='C0')
    sim_mat = align_representation.show_sim_mat(
        sim_mat_format = sim_mat_format, 
        visualization_config = visualize_config,
        visualization_config_hist = visualize_hist,
        show_distribution=False,
    )

# %%
## Dataset No.2 : THINGS
if data_select == "THINGS":
    sim_mat_format = "sorted"
    visualize_config = VisualizationConfig(
        figsize=(8, 6), 
        title_size = 15, 
        cmap = 'Blues',
        cbar_ticks_size=20,
        
        draw_category_line=True,
        category_line_color='C4',
        category_line_alpha=0.5,
        category_line_style='dashed',
       
        )
    
    visualize_hist = VisualizationConfig(figsize=(8, 6), cmap='C0')
    
    sim_mat = align_representation.show_sim_mat(
        sim_mat_format=sim_mat_format, 
        visualization_config=visualize_config,
        visualization_config_hist=visualize_hist,
        fig_dir=None,
        show_distribution=False,
        ticks='category'
    )

# %% [markdown]
# ## Reperesentation Similarity Aanalysis (RSA)

# %%
### parameters for computing RSA
# metric = "pearson" or "spearman" by scipy.stats
# method = "normal" or "all"
#     "normal" : perform RSA with the upper-triangle matrix of sim_mat
#     "all" : perform RSA with the full matrix of sim_mat
align_representation.RSA_get_corr(metric = "pearson", method = 'all')

# %% [markdown]
# ## Perform GW Alignment
# The optimization results are saved in the folder named "config.data_name" + "representations.name" vs "representation.name".  
# If you want to change the name of the saved folder, please make changes to "config.data_name" and "representations.name" (or change the "filename" in the code block below).

# %%
# If the computation has been completed and there is no need to recompute, set "compute_OT" to False. In this case, the previously calculated OT plans will be loaded.
# If users want to compare both numpy and torch, "compute_OT" needs to be True (e.g. an expected case is that users wants to change the "to_types" once after the computation is finished)
compute_OT = False

### If the previous optimization data exists, you can delete it.
# If you are attempting the same optimization with a different epsilon search space (eps_list), it is recommended to delete the previous results.
# Setting delete_results=True will delete both the database and the directory where the results of the previous optimization are stored.
# This function only works when n_job = 1, all the computed results exist, and "compute_OT" is set to False.
# The code will prompt for confirmation before deleting all the results.
delete_results = False

### If user wants to specify a different eps range in some certain pairs, you can make an dict of eps list for them. Other pairs will use the range defined in "config".
# If there is no pair to have their specific eps range, please set the pair_eps_range = {} or None. Then, the eps range which user defined in "config" will be used for each pair.
# example : pair_eps_list = {"Group1_vs_Group2":[10, 100]} -> the pair named "Group1_vs_Group2" will use the eps range ([10, 100]). eps_log in "config" will be applied
# caucation!! : please use "_vs_" between the representations' names.
pair_eps_list = {}
# pair_eps_list = {"Group1_vs_Group2":[5, 20]}

# %%
results_dir = "../results" # results will be saved in this folder

if data_select == "THINGS":
    sim_mat_format = "sorted" # "sorted" : the rows and columns of the OT plans are sorted by the coarce categories. If there is no need for sorting, set it to "default".
    visualize_config = VisualizationConfig(
        figsize=(8, 6), 
        title_size=15,
        cbar_ticks_size=15,
        draw_category_line=True,
        category_line_color='C2',
        category_line_alpha=0.2,
        category_line_style='dashed',
    )

    ot_list = align_representation.gw_alignment(
        results_dir = results_dir, 
        pair_eps_list = pair_eps_list,
        compute_OT = compute_OT,
        delete_results = delete_results,
        
        ## return_data : If True, the "OT_format" data will be returned in `ot_list`.
        return_data = False,
        return_figure = True,
        
        OT_format = sim_mat_format,
        visualization_config = visualize_config,
        
        ## show_log : if True, this will show the figures how the GWD was optimized.
        show_log=False, 
        
        ## fig_dir : you can define the path to which you save the figures (.png). If None, the figures will be saved in the same subfolder in "results_dir"
        fig_dir=None,
        
        ## ticks : you can use "objects" or "category" or "None"
        ticks='category', 
        
        ## filename : default is None. If None, the database name and folder name to save the results will automatically made. 
        filename=None, 
        save_dataframe=False,
        
        ## change_sampler_seed : If True, the random seed will be changed for each pair, else, the same seed defined in the next parameter will be used.  Default is False.
        change_sampler_seed=True, 
        
        ## fix_sampler_seed : this seed is used mainly for random sampler and TPE samapler. you can set any int (>= 0) value for sampler's seed. Default is 42.
        fix_sampler_seed = 42, 
        
        ## parallel_method : user can change the way of parallel computation, "multiprocess" or "multithread".
        # "multithread" may be effective for most case, please choose the best one for user's environment.
        parallel_method="multithread",
    )

if data_select == "color":
    visualize_config = VisualizationConfig(figsize=(10, 10), title_size = 15, show_figure=True)

    align_representation.gw_alignment(
        results_dir = results_dir,
        compute_OT = compute_OT,
        delete_results = delete_results,
        return_data = False,
        return_figure = True,
        OT_format = sim_mat_format, # "default"
        visualization_config = visualize_config,
    )

# %%
## Show how the GWD was optimized
#  show both the relationships between epsilons and GWD, and between accuracy and GWD
visualize_config = VisualizationConfig(figsize=(8,6), cmap='C0', show_figure=False)
align_representation.show_optimization_log(
    results_dir="../results",
    filename=None,
    fig_dir=None,
    visualization_config=visualize_config,
) 

# %% [markdown]
# # Step 4: Evaluation

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
align_representation.calc_accuracy(top_k_list = [1, 5, 10], eval_type = "k_nearest")
align_representation.plot_accuracy(eval_type = "k_nearest", scatter = True)

k_nearest_matching_rate = align_representation.k_nearest_matching_rate # you can get the dataframe directly 

# %%
## Calclate the category level accuracy

# If the data has the coarse category labels, you can observe the category level accuracy.
# This accuracy is calculated based on the OT plan.
if data_select == "THINGS":
    align_representation.calc_accuracy(top_k_list = [1, 5, 10], eval_type = "category", category_mat=category_mat)
    align_representation.plot_accuracy(eval_type = "category", scatter = True)

    category_level_accuracy = align_representation.category_level_accuracy # you can get the dataframe directly 

# %% [markdown]
# ## Visualize the aligned embeddings
# Using optimized transportation plans, you can align the embeddings of each representation to a shared space in an unsupervised manner.  
# The `"pivot"` refers to the target embeddings space to which the other embeddings will be aligned.   
# You have the option to designate the `"pivot"` as one of the representations or the barycenter.  
# Please ensure that 'pair_number_list' includes all pairs between the pivot and the other Representations.  
# 
# If you wish to utilize the barycenter, please make use of the method `AlignRepresentation.barycenter_alignment()`.  
# You can use it in the same manner as you did with `AlignRepresentation.gw_alignment()`.

# %%
# Set color labels and coarse category labels if exist.
# If there are a large number of objects within each group, such as in the case of THINGS data, visualizing all the points may not be meaningful. 
# In such cases, it is necessary to specify specific coarse category labels that you would like to visualize.
if data_select == "THINGS":
    category_name_list = ["bird", "insect", "plant", "clothing",  "furniture", "fruit", "drink", "vehicle"] # please specify the categories that you would like to visualize.
    category_mat = pd.read_csv("../data/category_mat_manual_preprocessed.csv", sep = ",", index_col = 0)   
    object_labels, category_idx_list, num_category_list, category_name_list = get_category_data(category_mat, category_name_list, show_numbers = True)  
    
    visualization_embedding = VisualizationConfig(
        figsize=(8, 8), 
        xlabel="PC1",
        ylabel="PC2", 
        zlabel="PC3", 
        marker_size=6,
        legend_size=10
    )
    
    align_representation.visualize_embedding(
        dim=3,  # the dimensionality of the space the points are embedded in. You can choose either 2 or 3.
        pivot=0, # the number of one of the representations or the "barycenter".
        visualization_config=visualization_embedding,
        category_name_list=category_name_list, 
        category_idx_list=category_idx_list, 
        num_category_list=num_category_list,
    )

# %%
if data_select == 'color':
    file_path = "../data/color_dict.csv"
    data_color = pd.read_csv(file_path)
    color_labels = data_color.columns.values # Set color labels if exist
    
    visualization_embedding = VisualizationConfig(
        color_labels=color_labels, # If there is no specific color labels, please set it to "None". Color labels will be automatically generated in that case. 
        color_hue=None, # If "color_labels=None", you have the option to choose the color hue as either "cool", "warm", or "None".
        figsize=(9, 9), 
        xlabel="PC1", 
        ylabel="PC2",
        zlabel="PC3", 
        legend_size=10
    )
    
    align_representation.visualize_embedding(
        dim=3, # the dimensionality of the space the points are embedded in. You can choose either 2 or 3.
        pivot=0, # the number of one of the representations or the "barycenter".
        visualization_config=visualization_embedding
    )

# %% [markdown]
#  ## Delete Results
# 
# If you want to delete both the directory and the database where the calculation results are stored all at once, you can use drop_gw_alignment_files.  
# Please be very careful because this operation is irreversible.

# %%
# align_representation.drop_gw_alignment_files(drop_all=True)

# %% [markdown]
# # GWOT without entropy
# By using optimal transportation plan obtained with entropic GW as an initial transportation matrix, we run the optimization of GWOT without entropy.  
# This procedure further minimizes GWD and enables us to fairly compare GWD values obtained with different entropy regularization values.  

# %%
import glob
import torch
import matplotlib.pyplot as plt
import ot
import copy

def get_top_k_trials(study, k):
    trials = study.trials_dataframe()
    sorted_trials = trials.sort_values(by="value", ascending=True)
    top_k_trials = sorted_trials.head(k)
    top_k_trials = top_k_trials[['number', 'value', 'params_eps']]
    return top_k_trials

def calculate_GWD(pairwise, top_k_trials):
    GWD0_list = list()
    OT0_list = list()
    for i in top_k_trials['number']:
        ot_path = glob.glob(pairwise.data_path + f"/gw_{i}.*")[0]
        if '.npy' in ot_path:
            OT = np.load(ot_path)
        elif '.pt' in ot_path:
            OT = torch.load(ot_path).to("cpu").numpy()

        C1 = pairwise.source.sim_mat
        C2 = pairwise.target.sim_mat
        p = OT.sum(axis=1)
        q = OT.sum(axis=0)
        OT0, log0 = ot.gromov.gromov_wasserstein(C1, C2, p, q, log=True, verbose=False, G0=OT)
        GWD0 = log0['gw_dist']
        GWD0_list.append(GWD0)
        OT0_list.append(OT0)
    
    return GWD0_list, OT0_list

def plot_OT(pairwise, pairwise_after, OT0_list, GWD0_list):
    pairwise._show_OT(title=f"before {pairwise.pair_name}")    
    pairwise_after._show_OT(title=f"after {pairwise.pair_name}")

def evaluate_accuracy_and_plot(pairwise, pairwise_after, eval_type):
    df_before = pairwise.eval_accuracy(top_k_list = [1, 5, 10], eval_type=eval_type)
    df_after = pairwise_after.eval_accuracy(top_k_list = [1, 5, 10], eval_type=eval_type)
    width = 0.35  # Width of the bars
    x = np.arange(len(df_before.index))
    plt.bar(x - width/2, df_before[pairwise.pair_name], width, label='before')
    plt.bar(x + width/2, df_after[pairwise.pair_name], width, label='after')
    plt.title(f'{eval_type} accuracy')
    plt.show()

def plot_GWD_optimization(top_k_trials, GWD0_list, pair_name):
    marker_size = 10
    plt.figure(figsize=(8,6))
    plt.scatter(top_k_trials["params_eps"], top_k_trials["value"], c = 'red', s=marker_size) # before
    plt.scatter(top_k_trials["params_eps"], GWD0_list, c = 'blue', s=marker_size) # after
    plt.xlabel("$\epsilon$")
    plt.ylabel("GWD")
    plt.xticks(rotation=30)
    plt.title(f"$\epsilon$ - GWD ({pair_name})")
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()

# %%
pairwise_list = align_representation.pairwise_list
pairwise_after_list = list()
number_of_besttrials = 10

# run GWOT without entropy regularization with optimized OTs obtained by entropic GWOT as initial values
for pairwise in pairwise_list:
    study = pairwise._run_optimization(compute_OT)
    top_k_trials = get_top_k_trials(study, number_of_besttrials)
    GWD0_list, OT0_list = calculate_GWD(pairwise, top_k_trials)
    # create new instance for after optimization
    pairwise_after = copy.deepcopy(pairwise)
    pairwise_after.OT = OT0_list[np.argmin(GWD0_list)]
    pairwise_after_list.append(pairwise_after)
    
# plot results
for pairwise, pairwise_after in zip(pairwise_list, pairwise_after_list):
    plot_OT(pairwise, pairwise_after, OT0_list, GWD0_list)
    evaluate_accuracy_and_plot(pairwise, pairwise_after, 'ot_plan')
    evaluate_accuracy_and_plot(pairwise, pairwise_after, 'k_nearest')
    plot_GWD_optimization(top_k_trials, GWD0_list, pairwise.pair_name)

# %%



