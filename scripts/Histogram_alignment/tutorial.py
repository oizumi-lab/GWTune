# %%
import time
import numpy as np
import pandas as pd
import torch
import ot
import matplotlib.pyplot as plt

# import jax
# import jax.numpy as jnp

import warnings
warnings.simplefilter("ignore")

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
        test_gw = GW_Alignment(self.model1, self.model2, self.p, self.q, device = device, to_types = to_types, speed_test = True)

        print('Computation Test : dtype = {}, device = {}.'.format(type(test_gw.p), device))
        
        start = time.time() # ここから時間計測スタート
        T, log = test_gw.entropic_GW(epsilon)
        end = time.time() - start # 時間計測終了
            
        print('Computation Time:', end)
        print('---------------------------------------------------------------------------')
        
        return log['gw_dist'], end 

    def _make_df(self, *, arg1, arg2):
        
        numpy_end = arg1[0]
        numpy_log = arg2[0]
        
        time_list = [numpy_end] + [a.item() for a in arg1[1:]]
        log_list = [numpy_log] + [a.item() for a in arg2[1:]]
        
        name_list = ['numpy_cpu', 'torch_cpu', 'torch_gpu', 'jax_cpu_end', 'jax_gpu_end']
        
        df = pd.DataFrame(data = {'time':time_list, 'log':log_list}, index = name_list[: len(time_list)])
        
        df['time'].plot(kind='bar', title = 'comparison', ylabel = 'time (sec.)', rot = 0)
        plt.show()
        
        df['log'].plot(kind='bar', title = 'comparison', ylabel = 'GW Loss', rot = 0, color = 'C1')
        plt.show()
        
        return df

    def comparison(self):
        epsilon = 6e-4
        
        numpy_cpu_log, numpy_cpu_end = self.time_test(epsilon, device = 'cpu', to_types = 'numpy')
        torch_cpu_log, torch_cpu_end = self.time_test(epsilon, device = 'cpu', to_types = 'torch')
        torch_gpu_log, torch_gpu_end = self.time_test(epsilon, device = 'cuda', to_types = 'torch')
        # jax_cpu_log, jax_cpu_end = time_test(model1, model2, p, q, epsilon = 6e-4, device = 'cpu', to_types = 'jax')
        # jax_gpu_log, jax_gpu_end = time_test(model1, model2, p, q, epsilon = 6e-4, device = 'cuda', to_types = 'jax')
        
        time_list = [numpy_cpu_end, torch_cpu_end, torch_gpu_end]
        log_list = [numpy_cpu_log, torch_cpu_log, torch_gpu_log]
        
        df = self._make_df(time_list, log_list)
        
        return df

class GW_Alignment():
    def __init__(self, pred_dist, target_dist, p, q, device = 'cpu', to_types = 'torch', speed_test = False):
        self.device = device
        self.to_types = to_types
        self.speed_test = speed_test
        self.pred_dist, self.target_dist, self.p, self.q = self._change_data(pred_dist, target_dist, p, q)
    
    def _change_data(self, *args):
        output = []
        for a in args:
            a = self._change_types(a)
            a = self._cpu_or_gpu(a)
            output.append(a)
        return output
        
    def _cpu_or_gpu(self, args):
        """
        typeを変換した後に、cpuかgpuかの設定をする。
        
        Args:
            args : データ

        Raises:
            ValueError: 三種類以外の方が入ってきたら、エラーを出す。

        Returns:
           args : 変換後のデータ
        """
        if isinstance(args, np.ndarray):
            return args
        
        elif isinstance(args, torch.Tensor):
            if 'gpu' in self.device:
                device = 'cuda'
                return args.to(device)
            else:
                return args.to(self.device)
        
        elif isinstance(args, jax.numpy.ndarray):
            if 'cuda' in self.device or 'gpu' in self.device:
                gpus = jax.devices("gpu")
                return jax.device_put(args, gpus[0])
            else:
                cpus = jax.devices("cpu")
                return jax.device_put(args, cpus[0])
        else:
            raise ValueError("Unknown type of non implemented here.")
    
    def _change_types(self, args):
        """
        ここで、任意の3type(numpy, torch, jax)を変換したいtypeに変換する。

        Args:
            to_types (_type_): 

        Returns:
            args :  
        """
        if self.to_types == 'jax':
            if isinstance(args, torch.Tensor):
                if args.is_cuda == True: 
                    args = args.to('cpu')
            return jnp.array(args)

        elif self.to_types == 'torch':
            if isinstance(args, np.ndarray):
                return torch.from_numpy(args).to(self.device)

            elif isinstance(args, jax.numpy.ndarray):
                args = np.array(args)
                return torch.from_numpy(args).to(self.device)

        elif self.to_types == 'numpy':
            if 'cuda' in self.device:
                raise ValueError("numpy doesn't work on CUDA")
            
            if isinstance(args, torch.Tensor):
                return args.to('cpu').numpy()
        
            elif isinstance(args, jax.numpy.ndarray):
                return np.array(args)
            
        return args
        
    def entropic_GW(self, epsilon, log = True, verbose = True):
        
        T = None 
        max_iter = 10 if self.speed_test else 1000
        tol = 1e-9
        
        C1, C2, p, q = ot.utils.list_to_array(self.pred_dist,
                                              self.target_dist,
                                              self.p,
                                              self.q)
        
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
            return T, log
        
        else:
            return T



# %%
if __name__ == '__main__':
    path1 = '../data/model1.pt'
    path2 = '../data/model2.pt'
    
    tgw = Tutorial(path1, path2)
    tgw.comparison()
 
# %%  

