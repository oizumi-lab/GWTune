#%%
import sys
import numpy as np
import ot
from scipy.spatial.distance import squareform
from scipy.special import comb
from scipy.stats import pearsonr
import optuna

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("../")
from src.GW_alignment import my_entropic_gromov_wasserstein, my_entropic_gromov_wasserstein2

%config InlineBackend.figure_formats = {'png', 'retina'}
plt.rcParams["font.size"] = 14

#%%
# 関数定義
def sort_for_scaling(X):
    x = squareform(X)
    x_sorted = np.sort(x)
    x_inverse_idx = np.argsort(x).argsort()
    return x,x_sorted,x_inverse_idx

def sc_plot(x,y, labels):
    plt.figure()
    plt.plot(x,y,'.')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()

def im_plot(X,Y,title_list):
    fig,axes = plt.subplots(1,2,figsize=(10,5))
    for ax, dm, t in zip(axes.reshape(-1), [X,Y],title_list):
        a = ax.imshow(dm)
        ax.set_title(t)
        cbar = fig.colorbar(a,ax=ax,shrink=0.7)
    plt.show()

def gw_alignment(X,Y,epsilon=0.01):
    T = np.random.uniform(low=0,high=1,size=(n,n))
    T = T/np.sum(T)
    p,q = np.sum(T,axis=1), np.sum(T,axis=0)

    gw,log = my_entropic_gromov_wasserstein2(C1=X,C2=Y,p=p,q=q,T=T,epsilon=epsilon,loss_fun="square_loss",verbose=True,log=True)
    plt.figure(figsize=(5,5))
    sns.heatmap(gw,square=True)
    plt.show()
    return gw, log

class Adjust_Disimilarity():
    def __init__(self,X, Y):
        self.x,self.x_sorted ,_ = sort_for_scaling(X)
        self.y,self.y_sorted ,_ = sort_for_scaling(Y)

    def sort_for_scaling(self,data):
        x = squareform(data)
        x_sorted = np.sort(x)
        x_inverse_idx = np.argsort(x).argsort()
        return x,x_sorted,x_inverse_idx

    def model_normalize(self, v, lam, alpha):
        data = alpha * ((np.power(1 + v, lam) - 1) / lam)
        return data

    def __call__(self,trial):
        alpha1 = trial.suggest_float('alpha1', 1e-1, 1e2, log = True)
        lam1 = trial.suggest_float('lam1', 1e-6, 1e2, log = True)
        alpha2 = trial.suggest_float('alpha2', 1e-2, 1e2, log = True)
        lam2 = trial.suggest_float('lam2', 1e-6, 1e2, log = True)

        x_data, y_data = self.model_normalize(self.x_sorted, lam1, alpha1),self.model_normalize(self.y_sorted, lam2, alpha2)

        l = ot.emd2_1d(x_data, y_data)
        return l
#%%
n = 100  # 点数
sigma = 1.5

np.random.seed(42)
x = np.random.uniform(0,1,size=comb(n,2, exact=True))
np.random.seed(0)
y = 2*x + np.random.uniform(0,sigma,size=comb(n,2, exact=True))
X = squareform(x)
Y = squareform(y)
#%%
# RSA correlation
corr,_ = pearsonr(x,y)
print(f'pearson r = {corr}')

sc_plot(x,y,['X','Y'])
im_plot(X,Y,['X','Y'])
gw,log = gw_alignment(X,Y,epsilon=0.01)

gwd = log['gw_dist']
print(f'GW distance = {gwd}')
#%%
# scaling
x,x_sorted,x_inverse_idx = sort_for_scaling(X)
y,y_sorted,y_inverse_idx = sort_for_scaling(Y)

transformed_y =  x_sorted[y_inverse_idx]
transformed_Y = squareform(transformed_y)
#%%
# RSA correlation
transformed_corr,_ = pearsonr(x,transformed_y)
print(f'pearson r = {transformed_corr}')

sc_plot(x,transformed_y,['X','transformed_Y'])
im_plot(X,transformed_Y,['X','transformed_Y'])
transformed_gw,transformed_log = gw_alignment(X,transformed_Y,epsilon=0.01)

transformed_gwd = transformed_log['gw_dist']
print(f'GW distance = {transformed_gwd}')
#%%
# Yeo-Johnson parameter変換
# parameterの最適化
tt = Adjust_Disimilarity(X,Y)
study = optuna.create_study(direction='minimize',study_name='adjust')
study.optimize(tt,n_trials=1000)

#%%
best_params = study.best_params
yj_x, yj_y = tt.model_normalize(tt.x, lam=best_params['lam1'],alpha=best_params['alpha1']), tt.model_normalize(tt.y, lam=best_params['lam2'],alpha=best_params['alpha2'])
yj_X, yj_Y = squareform(yj_x),squareform(yj_y)

#%%
# RSA correlation
yj_corr,_ = pearsonr(yj_x,yj_y)
print(f'pearson r = {yj_corr}')

sc_plot(yj_x,yj_y,['yj_X','yj_Y'])
im_plot(yj_X,yj_Y,['yj_X','yj_Y'])
yj_gw,yj_log = gw_alignment(yj_X,yj_Y,epsilon=0.00008)

yj_gwd = yj_log['gw_dist']
print(f'GW distance = {yj_gwd}')

#%%
# x = [6,5,1]
# y = [7,3,8]

# x_sorted,x_inverse_idx = np.sort(x),  np.argsort(x).argsort()
# y_sorted,y_inverse_idx = np.sort(y),  np.argsort(y).argsort()

# transformed_x = y_sorted[x_inverse_idx]
# print(transformed_x)
#%%
