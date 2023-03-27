# %%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# %%
import time
import numpy as np
import pandas as pd
import torch
import ot
import matplotlib.pyplot as plt
import os, sys
sys.path.append('../')

import warnings
warnings.simplefilter("ignore")

from src.utils.backend import Backend

# %%
class SpeedTest():
    def __init__(self, path_model1, path_model2) -> None:
        self.path_model1 = path_model1
        self.path_model2 = path_model2

    def load_sample_data(self):
        """
        ただ、sample dataをloadするだけの関数。

        Returns:
            _type_: _description_
        """
        model1 = torch.load(self.path_model1)
        model2 = torch.load(self.path_model2)
        p = ot.unif(len(model1))
        q = ot.unif(len(model2))
        return model1, model2, p, q

    def entropic_GW(self, epsilon, T = None, log = True, verbose = False):
        max_iter = 10
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
                # we can speed up the process by checking for the error only all the 10th iterations
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
        epsilon = 6e-4

        torch_gpu_log, torch_gpu_end = self.time_test(epsilon, device='cuda', to_types='torch')
        jax_gpu_log, jax_gpu_end = self.time_test(epsilon, device='gpu', to_types='jax')
         
        numpy_cpu_log, numpy_cpu_end = self.time_test(epsilon, device='cpu', to_types='numpy')
        torch_cpu_log, torch_cpu_end = self.time_test(epsilon, device='cpu', to_types='torch')
        jax_cpu_log, jax_cpu_end = self.time_test(epsilon, device='cpu', to_types='jax')

        time_list = [numpy_cpu_end, torch_cpu_end, torch_gpu_end, jax_cpu_end, jax_gpu_end]
        log_list = [numpy_cpu_log, torch_cpu_log, torch_gpu_log, jax_cpu_log, jax_gpu_log]

        df = self._make_df(time_list, log_list)

        return df


# %%
if __name__ == '__main__':
    path1 = '../data/model1.pt'
    path2 = '../data/model2.pt'

    tgw = SpeedTest(path1, path2)
    tgw.comparison()

# %%
