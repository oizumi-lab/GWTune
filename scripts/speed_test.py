# %%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# %%
import time
import numpy as np
import scipy
import pandas as pd
import torch
import ot
import matplotlib.pyplot as plt
import os, sys
import pickle as pkl

import warnings
warnings.simplefilter("ignore")

from src.utils.backend import Backend

# %%
class SpeedTest():
    def __init__(self, data_select) -> None:
        self.data_select = data_select
        pass

    def load_sample_data(self):
        """
        ただ、sample dataをloadするだけの関数。

        Returns:
            _type_: _description_
        """
        if self.data_select == 'DNN':
            path1 = '../data/model1.pt'
            path2 = '../data/model2.pt'
            C1 = torch.load(path1)
            C2 = torch.load(path2)
        elif self.data_select == 'color':
            data_path = '../data/num_groups_5_seed_0_fill_val_3.5.pickle'
            with open(data_path, "rb") as f:
                data = pkl.load(f)
            sim_mat_list = data["group_ave_mat"]
            C1 = sim_mat_list[0]
            C2 = sim_mat_list[1]
        elif self.data_select == 'face':
            data_path = '../data/faces_GROUP_interp.mat'
            mat_dic = scipy.io.loadmat(data_path)
            C1 = mat_dic["group_mean_ATTENDED"]
            C2 = mat_dic["group_mean_UNATTENDED"]

        p = ot.unif(len(C1))
        q = ot.unif(len(C2))
        return C1, C2, p, q

    def entropic_GW(self, epsilon, T = None, log = True, verbose = False):
        max_iter = 100
        tol = 1e-9

        C1, C2, p, q = self.pred_dist, self.target_dist, self.p, self.q

        nx = ot.backend.get_backend(C1, C2, p, q)

        # add T as an input
        if T is None:
            T = nx.outer(p, q)

        constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun = "square_loss")
        cpt = 0
        err = 1

        if log:
            log = {'err': []}

        while (err > tol and cpt < max_iter):
            Tprev = T
            # compute the gradient
            tens = ot.gromov.gwggrad(constC, hC1, hC2, T)
            T = ot.bregman.sinkhorn(p, q, tens, epsilon, method = 'sinkhorn')

            if cpt % 10 == 0:
                err = nx.norm(T - Tprev)
                if log:
                    log['err'].append(err)
                if verbose:
                    if cpt % 200 == 0:
                        print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(cpt, err))
            cpt += 1

        
        if log:
            log['gw_dist'] = ot.gromov.gwloss(constC, hC1, hC2, T)
            print(cpt)
            print(log['gw_dist'])
            return T, log

        else:
            return T

    def time_test(self, epsilon, device, to_types):
        be = Backend(device, to_types)
        pred_dist, target_dist, p, q = self.load_sample_data()
        self.pred_dist, self.target_dist, self.p, self.q = be(pred_dist, target_dist, p, q)
        
        print('Computation Test : dtype = {}, device = {}.'.format(type(self.p), device))

        start = time.time()  # ここから時間計測スタート
        T, log = self.entropic_GW(epsilon)
        end = time.time() - start  # 時間計測終了

        print('Computation Time:', end)
        print('---------------------------------------------------------------------------')

        return log['gw_dist'], end

    def _make_df(self, time_list, arg2):
        numpy_log = arg2[0]
        log_list = [numpy_log] + [a.item() for a in arg2[1:]]

        name_list = ['numpy_cpu', 'torch_cpu', 'torch_gpu', 'jax_cpu', 'jax_gpu']

        df = pd.DataFrame(data={'time': time_list, 'log': log_list}, index=name_list[: len(time_list)])

        df['time'].plot(kind='bar', title='comparison', ylabel='time (sec.)', rot=0)
        plt.show()

        df['log'].plot(kind='bar', title='comparison', ylabel='GW Loss', rot=0, color='C1')
        plt.show()

        return df

    def comparison(self):
        """
        POTによるGW計算の確認。
        3Type(numpy, torch, jax)での計算が正しくできるかを検証できる。
        
        2023.4.4 現在
        '../data/faces_GROUP_interp.mat'のデータを使うと、
        torchでの計算がepsilonの値次第で、実行不可能になることが判明。
        
        epsilon = 0.06 # この値だと、torchでの計算は全て0になる。cpu, cuda関係なく。
        epsilon = 0.08 # この値だと、torch, numpy, jaxで正しい計算結果となる。
        
        """
        if self.data_select == 'face':
            epsilon = 0.06 # この値だと、torchでの計算は全て0になる。
            epsilon = 0.08 # この値だと、torch, numpy, jaxで正しい計算結果となる。
        
        elif self.data_select == 'color':
            epsilon = 0.1
        
        elif self.data_select == 'DNN':
            epsilon = 6e-4
        
        numpy_cpu_log, numpy_cpu_end = self.time_test(epsilon, device='cpu', to_types='numpy')
        torch_cpu_log, torch_cpu_end = self.time_test(epsilon, device='cpu', to_types='torch')
        jax_cpu_log, jax_cpu_end = self.time_test(epsilon, device='cpu', to_types='jax')
        
        torch_gpu_log, torch_gpu_end = self.time_test(epsilon, device='cuda', to_types='torch')
        # jax_gpu_log, jax_gpu_end = self.time_test(epsilon, device='gpu', to_types='jax')

        time_list = [numpy_cpu_end, torch_cpu_end, torch_gpu_end, jax_cpu_end]#, jax_gpu_end]
        log_list = [numpy_cpu_log, torch_cpu_log, torch_gpu_log, jax_cpu_log]#, jax_gpu_log]

        df = self._make_df(time_list, log_list)

        return df


# %%
if __name__ == '__main__':
    data_select = 'color'
    tgw = SpeedTest(data_select)
    tgw.comparison()

# %%
