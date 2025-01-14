# %%
import os, sys, glob
import optuna
import matplotlib.pyplot as plt

#%%
def get_data(data_select):
    path = f"../results/{data_select}"
    print(glob.glob(f"{path}/*/*/*.db"))
    random_path = glob.glob(f"{path}/*/random/*.db")[0]
    uniform_path = glob.glob(f"{path}/*/uniform/*.db")[0]
    
    if "simulation" in data_select:
        random_grid_path = glob.glob(f"{path}/random+grid/*/random/*.db")[0]
    else:
        random_grid_path = glob.glob(f"../results/random+grid/{data_select}/*/random/*.db")[0]

    df_random = optuna.load_study(study_name = os.path.basename(random_path).split(".db")[0], storage = f"sqlite:///{random_path}").trials_dataframe()
    df_uniform = optuna.load_study(study_name = os.path.basename(uniform_path).split(".db")[0], storage = f"sqlite:///{uniform_path}").trials_dataframe()
    df_random_grid = optuna.load_study(study_name = os.path.basename(random_grid_path).split(".db")[0], storage = f"sqlite:///{random_grid_path}").trials_dataframe()

    return df_random, df_uniform, df_random_grid

#%%
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

#%%

# data_select_list = ["simulation_corr0.9", "simulation_corr0.8", "simulation_corr0.7"] # 
# data_select_list = ["simulation_clusters4", "simulation_clusters8", "simulation_clusters16"] #,, "simulation_clusters4"
data_select_list = ["simulation_high_symmetry", "simulation_medium_symmetry", "simulation_low_symmetry"] #,, "simulation_clusters4"
title_list = ["high symmetry", "medium symmetry", "low symmetry"] # , "4 clusters"

all_data = []
for data_select, title in zip(data_select_list, title_list):
    random, uniform, random_grid = get_data(data_select)
    all_data.append((random, uniform, random_grid))

# plot
plt.figure(figsize=(10, 12))
plt.suptitle("Comparison of different search strategies")

n = len(data_select_list)
for i in range(n):
    plt.subplot(n, 1, i+1)
    plt.title(title_list[i])
    plt.plot(get_min_values(all_data[i][1]), label = "uniform with grid")
    plt.plot(get_min_values(all_data[i][2]), label = "random with grid")
    plt.plot(get_min_values(all_data[i][0]), label = "random with TPE")
    plt.xlim(0, 100)
    plt.ylim(0.0, 0.2)
    plt.xlabel("Trial")
    plt.ylabel("minimum GWD")
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.show()

#%%
things_random, things_uniform, things_random_grid = get_data("THINGS")

#%%
allen_random, allen_uniform, allen_random_grid = get_data("AllenBrain")

#%%
dnn_random, dnn_uniform, dnn_random_grid = get_data("DNN")

# %%
plt.figure(figsize=(10, 12))
plt.suptitle("Comparison of different search strategies")

plt.subplot(3, 1, 1)
plt.title("Behavioral data: Human psychological embeddings of natural objects")
plt.plot(get_min_values(things_uniform), label = "uniform with grid")
plt.plot(get_min_values(things_random_grid), label = "random with grid")
plt.plot(get_min_values(things_random), label = "random with TPE")
plt.xlabel("Trial")
plt.ylabel("minimum GWD")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.title("Neural data: Neuropixels visual coding in mice")
plt.plot(get_min_values(allen_uniform), label = "uniform with grid")
plt.plot(get_min_values(allen_random_grid), label = "random with grid")
plt.plot(get_min_values(allen_random), label = "random with TPE")
plt.xlabel("Trial")
plt.ylabel("minimum GWD")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.title("Model: Vision Deep Neural Networks")
plt.plot(get_min_values(dnn_uniform), label = "uniform with grid")
plt.plot(get_min_values(dnn_random_grid), label = "random with grid")
plt.plot(get_min_values(dnn_random), label = "random with TPE")
plt.xlabel("Trial")
plt.ylabel("minimum GWD")
plt.grid(True)
plt.legend()


plt.tight_layout()
plt.show()
# %%
