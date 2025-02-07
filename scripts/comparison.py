# %%
import os, sys, glob
import optuna
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch

#%%
def get_data(data_select, init_plan, sampler_name):
    path = f"../results/{data_select}/{sampler_name}"
    db_path = glob.glob(f"{path}/*/{init_plan}/*.db")[0]
    df = optuna.load_study(study_name = os.path.basename(db_path).split(".db")[0], storage = f"sqlite:///{db_path}").trials_dataframe()
    return df

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

def get_max_acc(df):
    max_acc = []
    current_max = df['value'][0]
    for i in range(len(df)):
        value = df['value'][i]

        if value > current_max:
            current_max = value
            max_acc.append(current_max)
        else:
            max_acc.append(current_max)
    
    return max_acc

#%%
things_random = get_data("THINGS", "random", "tpe")
things_uniform = get_data("THINGS", "uniform", "grid") 
things_random_grid = get_data("THINGS", "random", "grid")

#%%
allen_random = get_data("AllenBrain", "random", "tpe")
allen_uniform = get_data("AllenBrain", "uniform", "grid") 
allen_random_grid = get_data("AllenBrain", "random", "grid")

#%%
dnn_random = get_data("DNN", "random", "tpe")
dnn_uniform = get_data("DNN", "uniform", "grid") 
dnn_random_grid = get_data("DNN", "random", "grid")


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
min_allen = pd.DataFrame({"TPE + Random": allen_random["value"].min(), "Grid Search + Random": allen_random_grid["value"].min(), "Grid Search + Uniform": allen_uniform["value"].min()}, index = ["Minimum GWD"])
# %%
min_allen.plot(figsize = (8,6), kind = "bar", rot = 0, title = "Minimum GWD for AllenBrain data")

#%%
def get_ot(df, init_plan, sampler_name):
    idx = df[df["value"] == df["value"].min()].index[0]
    print("acc.", df[df["value"] == df["value"].min()]["user_attrs_best_acc"].values[0])
    npy_path = glob.glob(f"../results/AllenBrain/{sampler_name}/*/{init_plan}/*/gw_{idx}.npy")[0]
    ot = np.load(npy_path)
    
    return ot

#%%
ot_random = get_ot(allen_random, "random", "tpe")
ot_uniform = get_ot(allen_uniform, "uniform", "grid")
ot_random_grid = get_ot(allen_random_grid, "random", "grid")


# %%
import seaborn as sns
plt.figure(figsize=(10, 3.6))
plt.suptitle("Neural data: Neuropixels visual coding in mice")

plt.subplot(1, 3, 1)
plt.title("TPE + Random")
plt.imshow(ot_random, cmap="rocket_r")
plt.xlabel("90 short movies of VISam (pseudo mouse A)")
plt.ylabel("90 short movies of VISal (pseudo mouse B)")

plt.subplot(1, 3, 2)
plt.title("Grid Search + Random")
plt.imshow(ot_random_grid, cmap="rocket_r")
plt.xlabel("90 short movies of VISam (pseudo mouse A)")
plt.ylabel("90 short movies of VISal (pseudo mouse B)")

plt.subplot(1, 3, 3)
plt.title("Grid Search + Uniform")
plt.imshow(ot_uniform, cmap="rocket_r")
plt.xlabel("90 short movies of VISam (pseudo mouse A)")
plt.ylabel("90 short movies of VISal (pseudo mouse B)")


plt.tight_layout()
plt.show()

#%%
def get_ot(data_name, init_plan, sampler_name, idx):
    npy_path = glob.glob(f"../results/{data_name}/{sampler_name}/*/{init_plan}/*/gw_{idx}.npy")
    if len(npy_path) == 0:
        npy_path = glob.glob(f"../results/{data_name}/{sampler_name}/*/{init_plan}/*/gw_{idx}.pt")[0]
        ot = torch.load(npy_path, weights_only=False).numpy()
        return ot
    else:
        npy_path = npy_path[0]
        ot = np.load(npy_path)
        return ot

# %%
sampler = "grid"
init_plan = "random"
data_name = "THINGS"


#%%
def plot_all_ot(data_name, init_plan, sampler):
    df = get_data(data_name, init_plan, sampler)
    
    num_ot = 10
    plt.subplots(num_ot, 10, figsize=(18, 18))
    plt.suptitle(f"OT {init_plan}, {sampler} (ascending sorted by GWD)", size=20, y=0.99)

    for _, idx in enumerate(df.sort_values(by="value").index[:num_ot*10]):
        ot = get_ot(data_name, init_plan, sampler, idx)
        
        plt.subplot(num_ot, 10, _+1)
        plt.imshow(ot, cmap="rocket_r")
        
        if data_name == "THINGS":
            clim_max = 1e-10
            plt.clim(0, clim_max)
        elif data_name == "AllenBrain":
            pass
        elif data_name == "DNN":
            clim_max = 1e-5
            plt.clim(0, clim_max)
        
        gwd = df.loc[idx, "value"]
        
        plt.title(f"GWD:{gwd:.2e}")

    plt.tight_layout()
    plt.show()

# %%
plot_all_ot("THINGS", "random", "tpe")

# %%
plot_all_ot("THINGS", "random", "grid")

# %%
plot_all_ot("THINGS", "uniform", "grid")

# %%
plot_all_ot("AllenBrain", "random", "tpe")

# %%
plot_all_ot("AllenBrain", "random", "grid")

# %%
plot_all_ot("AllenBrain", "uniform", "grid")

# %%
plot_all_ot("DNN", "random", "tpe")

# %%
plot_all_ot("DNN", "random", "grid")

# %%
plot_all_ot("DNN", "uniform", "grid")
# %%
