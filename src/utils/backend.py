# %%
import os
import numpy as np
import jax
import jax.numpy as jnp
import torch
import random
import ot


# %%
# JAXの環境変数。
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
jax.config.update("jax_enable_x64", True)

#%%
str_type_error = "All array should be from the same type/backend. Current types are : {}"

def fix_seed(seed=42):
    # Python
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

#%%

def make_backend(to_types = None):
    if to_types == 'numpy':
        return NumpyBackend()
    elif to_types == 'torch':
        return TorchBackend()
    elif to_types == 'jax':
        return JaxBackend()
    # elif isinstance(args[0], cp_type):  # pragma: no cover
    #     return CupyBackend()
    # elif isinstance(args[0], tf_type):
    #     return TensorflowBackend()
    else:
        raise ValueError("Unknown type of non implemented backend.")


def get_backend(*args):
    """Returns the proper backend for a list of input arrays

        Also raises TypeError if all arrays are not from the same backend
    """
    # check that some arrays given
    if not len(args) > 0:
        raise ValueError(" The function takes at least one parameter")
    # check all same type
    if not len(set(type(a) for a in args)) == 1:
        raise ValueError(str_type_error.format([type(a) for a in args]))

    if isinstance(args[0], np.ndarray):
        return NumpyBackend()
    elif isinstance(args[0], torch.Tensor):
        return TorchBackend()
    elif isinstance(args[0], jax.numpy.ndarray):
        return JaxBackend()
    # elif isinstance(args[0], cp_type):  # pragma: no cover
    #     return CupyBackend()
    # elif isinstance(args[0], tf_type):
    #     return TensorflowBackend()
    else:
        raise ValueError("Unknown type of non implemented backend.")


class Backend(ot.backend.Backend):
    def change_data(self, *args, device = 'cpu'):
        """
        3つのデータ型(numpy, torch, jax)を変換したい型にして、変換後にCPUかGPUかの選択ができる(当然numpyはcpuのみ。jitとかで高速化できるけど)。
        Input :
            *args : 変換したい変数を全て受け取る。詳しくは可変型変数を調べてください。
        Returns:
            output : 変換したい変数のリスト
        """
        output = []
        for a in args:
            a = self._change_types(a, device)
            output.append(a)

        if len(args) == 1:
            output = output[0]

        return output

    def save(self, file, a):
        raise NotImplementedError()

    def load(self, file):
        raise NotImplementedError()

    def to(self, a, device, dtype = None):
        raise NotImplementedError()


class NumpyBackend(ot.backend.NumpyBackend, Backend):

    def _change_types(self, args, device):
        if 'cuda' in device or 'gpu' in device:
            raise ValueError("numpy doesn't work on CUDA")

        if isinstance(args, torch.Tensor):
            return args.to('cpu').numpy()

        elif isinstance(args, jax.numpy.ndarray):
            return np.array(args)

        else:
            return args

    def save(self, file, a):
        np.save(file, a)

    def load(self, file):
        return np.load(file)

    def item(self,a):
        if isinstance(a, np.ndarray):
            raise ValueError('Input is not scalar. Use .to_numpy.')
        else:
            return a

    def to(self, a, device, dtype = None):
        if dtype is None:
            return a
        else:
            return a.astype(dtype)


class TorchBackend(ot.backend.TorchBackend, Backend):

    def _change_types(self, args, device):
        if 'gpu' in device:
            raise ValueError('torch uses "cuda" instead of "gpu".')

        if isinstance(args, np.ndarray):
            return torch.from_numpy(args).float().to(device)

        elif isinstance(args, jax.numpy.ndarray):
            args = np.array(args)
            return torch.from_numpy(args).float().to(device)

        else:
            return args.to(device)

    def save(self, file, a):
        torch.save(a, file + '.pt')

    def load(self, file):
        return torch.load(file)

    def item(self,a):
        if a.ndim > 0:
            raise ValueError('Input is not scalar. Use .to_numpy.')
        else:
            return a.item()

    def to(self, a, device, dtype = None):
        if dtype is None:
            return a.to(device)
        elif dtype == 'float':
            return a.float().to(device)
        else:
            ValueError('Not implemented type.')

class JaxBackend(ot.backend.JaxBackend, Backend):

    def _change_types(self, args, device):
        if isinstance(args, torch.Tensor):
            if args.is_cuda == True: args = args.to('cpu')
            args = jnp.asarray(args)

        if isinstance(args, np.ndarray):
            args = jnp.asarray(args)

        if 'cuda' in device or 'gpu' in device: # ここ、のちに拡張工事が必要。GPUのボード番号の選択ができるようにしないといけない。
            gpus = jax.devices("gpu")
            return jax.device_put(args, gpus[0])
        else:
            cpus = jax.devices("cpu")
            return jax.device_put(args, cpus[0])

    def save(self, file, a):
        jax.numpy.save(file, a)

    def load(self, file):
        return jax.numpy.load(file)

    def to(self, a, device, dtype = None):
        if dtype is None:
            return jax.device_put(a, device)
        else:
            ValueError('Not implemented type.')

# %%
if __name__ == '__main__':
    test_numpy1 = np.arange(10)
    test_numpy2 = np.arange(10, 20)

    # %%
    nx = make_backend(to_types = 'jax')
    test_jax1, test_jax2 = nx.change_data(test_numpy1, test_numpy2, device='cpu')
    # backend = Backend(device = 'cpu', to_types = 'jax')
    # test_jax1, test_jax2 = backend.change_data(test_numpy1, test_numpy2)
    print(test_jax1, test_jax2)
    print(type(test_jax1), type(test_jax2))

    # # %%
    nx = make_backend(to_types = 'torch')
    test_torch1 = nx.change_data(test_numpy1, device='cuda')
    # backend = Backend(device = 'cuda', to_types = 'torch')
    # test_torch1 = backend.change_data(test_numpy1)
    print(test_torch1)
    print(type(test_torch1))

    # # %%
    # tt = np.arange(3)
    # tt_jax = jnp.array(tt)
    # gpus = jax.devices('gpu')
    # tt_jax = jax.device_put(tt_jax, gpus[0])


# %%
