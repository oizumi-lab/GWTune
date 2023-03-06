# %%
import jax.numpy as jnp
import jax
import time
import numpy as np
import pandas as pd
import torch
import ot
import matplotlib.pyplot as plt
import os

import warnings
# warnings.simplefilter("ignore")

class GW_Alignment():
    """
    2023/3/6 大泉先生
    
    1. epsilonに関して
    epsilon: １つ
    epsilonの範囲を決める：サーチ方法 optuna, 単純なgrid (samplerの種類, optuna)
    
    2. 初期値に関して
    初期値1つ固定: diagonal, uniform outer(p,q), 乱数
    初期値ランダムで複数: 乱数
    """
    def __init__(self, pred_dist, target_dist, p, q, device='cpu', to_types='torch', speed_test=False):
        """

        """
        self.device = device
        self.to_types = to_types
        self.speed_test = speed_test
        self.pred_dist, self.target_dist, self.p, self.q = self._change_data(
            pred_dist, target_dist, p, q)

    def _change_data(self, *args):
        """

        """
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
                return torch.from_numpy(args).float().to(self.device)

            elif isinstance(args, jax.numpy.ndarray):
                args = np.array(args)
                return torch.from_numpy(args).float().to(self.device)

        elif self.to_types == 'numpy':
            if 'cuda' in self.device:
                raise ValueError("numpy doesn't work on CUDA")

            if isinstance(args, torch.Tensor):
                return args.to('cpu').numpy()

            elif isinstance(args, jax.numpy.ndarray):
                return np.array(args)

        return args

    def entropic_GW(self, epsilon, T = None, log = True, verbose = True):
        max_iter = 10 if self.speed_test else 1000
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
            return T, log

        else:
            return T



# %%
