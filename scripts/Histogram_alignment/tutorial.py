# %%
import time
import numpy as np
import pandas as pd
import torch
import ot
import matplotlib.pyplot as plt
import os

import warnings
warnings.simplefilter("ignore")

from src.gw_alignment import GW_Alignment

# %%


# # %%
# gpus = jax.devices('gpu')

# # %%
# a = jnp.array(1)

# # %%
# b = jax.device_put(a, gpus[0])
# # %%
# print(a.item())
# # %%
# print(b)
# type(b)
# # %%
# c = [a.item(),b]
# # %%

class Tutorial():
    def __init__(self, path_model1, path_model2) -> None:
        self.path_model1 = path_model1
        self.path_model2 = path_model2

        self.model1, self.model2, self.p, self.q = self.load_sample_data()

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

    def time_test(self, epsilon, device, to_types):
        test_gw = GW_Alignment(self.model1, self.model2, self.p,
                               self.q, device=device, to_types=to_types, speed_test=True)

        print('Computation Test : dtype = {}, device = {}.'.format(
            type(test_gw.p), device))

        start = time.time()  # ここから時間計測スタート
        T, log = test_gw.entropic_GW(epsilon)
        end = time.time() - start  # 時間計測終了

        print('Computation Time:', end)
        print('---------------------------------------------------------------------------')

        return log['gw_dist'], end

    def _make_df(self, time_list, arg2):
        numpy_log = arg2[0]
        log_list = [numpy_log] + [a.item() for a in arg2[1:]]

        name_list = ['numpy_cpu', 'torch_cpu',
                     'torch_gpu', 'jax_cpu_end', 'jax_gpu_end']

        df = pd.DataFrame(data={'time': time_list, 'log': log_list}, index=name_list[: len(time_list)])

        df['time'].plot(kind='bar', title='comparison', ylabel='time (sec.)', rot=0)
        plt.show()

        df['log'].plot(kind='bar', title='comparison', ylabel='GW Loss', rot=0, color='C1')
        plt.show()

        return df

    def comparison(self):
        epsilon = 6e-4

        numpy_cpu_log, numpy_cpu_end = self.time_test(epsilon, device='cpu', to_types='numpy')
        torch_cpu_log, torch_cpu_end = self.time_test(epsilon, device='cpu', to_types='torch')
        torch_gpu_log, torch_gpu_end = self.time_test(epsilon, device='cuda', to_types='torch')
        # jax_cpu_log, jax_cpu_end = time_test(model1, model2, p, q, epsilon = 6e-4, device = 'cpu', to_types = 'jax')
        # jax_gpu_log, jax_gpu_end = time_test(model1, model2, p, q, epsilon = 6e-4, device = 'cuda', to_types = 'jax')

        time_list = [numpy_cpu_end, torch_cpu_end, torch_gpu_end]
        log_list = [numpy_cpu_log, torch_cpu_log, torch_gpu_log]

        df = self._make_df(time_list, log_list)

        return df


# %%
if __name__ == '__main__':
    path1 = '../../data/model1.pt'
    path2 = '../../data/model2.pt'

    tgw = Tutorial(path1, path2)
    tgw.comparison()

# %%
