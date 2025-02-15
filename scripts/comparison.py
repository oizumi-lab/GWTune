# %%
import os, sys, glob
import optuna
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch

#%%
def get_data(data_select, init_plan, sampler_name, index=None):
    path = f"../results/{data_select}/{sampler_name}"
    db_path = glob.glob(f"{path}/*/{init_plan}/*.db")[0]
    df = optuna.load_study(study_name = os.path.basename(db_path).split(".db")[0], storage = f"sqlite:///{db_path}").trials_dataframe()
    if index is not None:
        df = df[:index]
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
ind = None
#%%
things_random = get_data("THINGS", "random", "tpe", index=ind)
things_uniform = get_data("THINGS", "uniform", "grid", index=ind)
things_random_grid = get_data("THINGS", "random", "grid", index=ind)

#%%
allen_random = get_data("AllenBrain", "random", "tpe", index=ind)
allen_uniform = get_data("AllenBrain", "uniform", "grid", index=ind)
allen_random_grid = get_data("AllenBrain", "random", "grid", index=ind)

#%%
dnn_random = get_data("DNN", "random", "tpe", index=ind)
dnn_uniform = get_data("DNN", "uniform", "grid", index=ind)
dnn_random_grid = get_data("DNN", "random", "grid", index=ind)

# %%
plt.style.use("default")
# plt.rcParams["grid.color"] = "black"
plt.rcParams['font.family'] = "Arial"
plt.figure(figsize=(6, 8))
# plt.suptitle("Comparison of different initialization strategies")

plt.subplot(3, 1, 1)
plt.title("Behavioral data : THINGS")
plt.plot(get_min_values(things_random), label = "Random + TPE")
plt.plot(get_min_values(things_random_grid), label = "Random + Grid Search")
plt.plot(get_min_values(things_uniform), label = "Uniform + Grid Search")
plt.xlabel("Trial")
plt.ylabel("minimum GWD")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.title("Neural data : AlenBrain")
plt.plot(get_min_values(allen_random), label = "Random + TPE")
plt.plot(get_min_values(allen_random_grid), label = "Random + Grid Search")
plt.plot(get_min_values(allen_uniform), label = "Uniform + Grid Search")
plt.xlabel("Trial")
plt.ylabel("minimum GWD")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.title("Model : DNN")
plt.plot(get_min_values(dnn_random), label = "Random + TPE")
plt.plot(get_min_values(dnn_random_grid), label = "Random + Grid Search")
plt.plot(get_min_values(dnn_uniform), label = "Uniform + Grid Search")
plt.xlabel("Trial")
plt.ylabel("minimum GWD")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

#%%
plt.figure()
plt.title("Neural data : AlenBrain")
plt.plot(get_min_values(allen_random), label = "Random + TPE")
plt.plot(get_min_values(allen_random_grid), label = "Random + Grid Search")
plt.plot(get_min_values(allen_uniform), label = "Uniform + Grid Search")
plt.xlabel("Trial")
plt.ylabel("minimum GWD")
plt.grid(True)
plt.legend()

#%%
def get_ot(df, dataset, init_plan, sampler_name):
    idx = df[df["value"] == df["value"].min()].index[0]
    print("acc.", df[df["value"] == df["value"].min()]["user_attrs_best_acc"].values[0])
    npy_path = glob.glob(f"../results/{dataset}/{sampler_name}/*/{init_plan}/*/gw_{idx}.npy")
    
    if len(npy_path) == 0:
        torch_path = glob.glob(f"../results/{dataset}/{sampler_name}/*/{init_plan}/*/gw_{idx}.pt")[0]
        ot = torch.load(torch_path, weights_only=False).numpy()
    else:
        ot = np.load(npy_path[0])
    return ot

#%%
ot_things_random = get_ot(things_random, "THINGS", "random", "tpe")
ot_things_uniform = get_ot(things_uniform, "THINGS", "uniform", "grid")
ot_things_random_grid = get_ot(things_random_grid, "THINGS", "random", "grid")

#%%
ot_allen_random = get_ot(allen_random, "AllenBrain", "random", "tpe")
ot_allen_uniform = get_ot(allen_uniform, "AllenBrain", "uniform", "grid")
ot_allen_random_grid = get_ot(allen_random_grid, "AllenBrain", "random", "grid")

#%%
ot_dnn_random = get_ot(dnn_random, "DNN", "random", "tpe")
ot_dnn_uniform = get_ot(dnn_uniform, "DNN", "uniform", "grid")
ot_dnn_random_grid = get_ot(dnn_random_grid, "DNN", "random", "grid")


# %%
import seaborn as sns
plt.style.use("default")
plt.rcParams['font.family'] = "Arial"
plt.figure(figsize=(12, 16))

plt.subplot(3, 3, 1)
plt.title("THINGS : Random + TPE", fontsize=15)
plt.imshow(ot_things_random, cmap="rocket_r")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.colorbar()
plt.clim(0, 1e-4)
plt.xlabel("1854 objects", fontsize=13)
plt.ylabel("1854 objects", fontsize=13)

plt.subplot(3, 3, 2)
plt.title("THINGS : Random + Grid Search", fontsize=15)
plt.imshow(ot_things_random_grid, cmap="rocket_r")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.clim(0, 1e-4)
plt.xlabel("1854 objects", fontsize=13)
plt.ylabel("1854 objects", fontsize=13)

plt.subplot(3, 3, 3)
plt.title("THINGS : Uniform + Grid Search", fontsize=15)
plt.imshow(ot_things_uniform, cmap="rocket_r")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.clim(0, 1e-4)
plt.xlabel("1854 objects", fontsize=13)
plt.ylabel("1854 objects", fontsize=13)

plt.subplot(3, 3, 4)
plt.title("AllenBrain : Random + TPE", fontsize=15)
plt.imshow(ot_allen_random, cmap="rocket_r")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("90 short movies of VISam (pseudo mouse A)", fontsize=13)
plt.ylabel("90 short movies of VISal (pseudo mouse B)", fontsize=13)

plt.subplot(3, 3, 5)
plt.title("AllenBrain : Random + Grid Search", fontsize=15)
plt.imshow(ot_allen_random_grid, cmap="rocket_r")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("90 short movies of VISam (pseudo mouse A)", fontsize=13)
plt.ylabel("90 short movies of VISal (pseudo mouse B)", fontsize=13)

plt.subplot(3, 3, 6)
plt.title("AllenBrain : Uniform + Grid Search", fontsize=15)
plt.imshow(ot_allen_uniform, cmap="rocket_r")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("90 short movies of VISam (pseudo mouse A)", fontsize=13)
plt.ylabel("90 short movies of VISal (pseudo mouse B)", fontsize=13)

plt.subplot(3, 3, 7)
plt.title("DNN : Random + TPE", fontsize=15)
plt.imshow(ot_dnn_random, cmap="rocket_r")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.clim(0, 1e-4)
plt.xlabel("1000 images of ResNet50", fontsize=13)
plt.ylabel("1000 images of VGG19", fontsize=13)

plt.subplot(3, 3, 8)
plt.title("DNN : Random + Grid Search", fontsize=15)
plt.imshow(ot_dnn_random_grid, cmap="rocket_r")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.clim(0, 1e-4)
plt.xlabel("1000 images of ResNet50", fontsize=13)
plt.ylabel("1000 images of VGG19", fontsize=13)

plt.subplot(3, 3, 9)
plt.title("DNN : Uniform + Grid Search", fontsize=15)
plt.imshow(ot_dnn_uniform, cmap="rocket_r")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.clim(0, 1e-4)
plt.xlabel("1000 images of ResNet50", fontsize=13)
plt.ylabel("1000 images of VGG19", fontsize=13)

plt.tight_layout()
# plt.show()
plt.savefig("../results/real_data_ot.svg")

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
    # plt.suptitle(f"OT {init_plan}, {sampler} (ascending sorted by GWD)", size=20, y=0.99)

    for _, idx in enumerate(df.sort_values(by="value").index[:num_ot*10]):
        ot = get_ot(data_name, init_plan, sampler, idx)
        
        plt.subplot(num_ot, 10, _+1)
        plt.imshow(ot, cmap="rocket_r")
        
        if data_name == "THINGS":
            clim_max = 2e-6
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
