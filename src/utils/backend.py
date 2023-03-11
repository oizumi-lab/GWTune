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


# %%
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


class Backend():
    def __init__(self, device = 'cpu', to_types = 'torch'):
        """
        入力された変数の型を揃えてあげる。バラバラな場合でも、単一の型に変換できるはず。
        基本的には3つ(numpy, torch, jax)の型のみ受付られる。
        """
        self.device = device
        self.to_types = to_types
        pass
    
    def change_data(self, *args):
        """
        3つのデータ型(numpy, torch, jax)を変換したい型にして、変換後にCPUかGPUかの選択ができる(当然numpyはcpuのみ。jitとかで高速化できるけど)。
        Input :
            *args : 変換したい変数を全て受け取る。詳しくは可変型変数を調べてください。
        Returns:
            output : 変換したい変数のリスト
        """
        
        output = []
        for a in args:
            a = self._change_types(a)
            output.append(a)
        
        if len(args) == 1:
            output = output[0]
        
        return output

    def _change_types(self, args):
        """
        ここで、任意の3type(numpy, torch, jax)を変換したいtypeに変換する。
        変換後、CPU or GPUの選択ができる。
        
        おそらく、今後、拡張が必要な気がする。GPUの番号指定を行えるようにする必要がある。

        Args:
            to_types (_type_):

        Returns:
            args :
        """
        if self.to_types == 'jax':
            if isinstance(args, torch.Tensor):
                if args.is_cuda == True: args = args.to('cpu')    
                args = jnp.asarray(args)
            
            if isinstance(args, np.ndarray):
                args = jnp.asarray(args)
        
            if 'cuda' in self.device or 'gpu' in self.device: # ここ、のちに拡張工事が必要。GPUのボード番号の選択ができるようにしないといけない。
                gpus = jax.devices("gpu")
                return jax.device_put(args, gpus[0])
            else:
                cpus = jax.devices("cpu")
                return jax.device_put(args, cpus[0])
        
        elif self.to_types == 'torch':
            if 'gpu' in self.device:
                raise ValueError('torch uses "cuda" instead of "gpu".')
            
            if isinstance(args, np.ndarray):
                return torch.from_numpy(args).float().to(self.device)

            elif isinstance(args, jax.numpy.ndarray):
                args = np.array(args)
                return torch.from_numpy(args).float().to(self.device)

            else:
                return args.to(self.device)

        elif self.to_types == 'numpy':
            if 'cuda' in self.device or 'gpu' in self.device:
                raise ValueError("numpy doesn't work on CUDA")

            if isinstance(args, torch.Tensor):
                return args.to('cpu').numpy()

            elif isinstance(args, jax.numpy.ndarray):
                return np.array(args)

            else:
                return args
        
        else:
            raise ValueError("Unknown type of non implemented here.")
        

# %%
if __name__ == '__main__':
    test_numpy1 = np.arange(10)
    test_numpy2 = np.arange(10, 20)
    
    # %%
    backend = Backend(device = 'cpu', to_types = 'jax')
    test_jax1, test_jax2 = backend.change_data(test_numpy1, test_numpy2)
    print(test_jax1, test_jax2)
    print(type(test_jax1), type(test_jax2))
    
    # %%
    backend = Backend(device = 'cuda', to_types = 'torch')
    test_torch1 = backend.change_data(test_numpy1)
    print(test_torch1)
    print(type(test_torch1))

    # %%
    tt = np.arange(3)
    tt_jax = jnp.array(tt)
    gpus = jax.devices('gpu')
    tt_jax = jax.device_put(tt_jax, gpus[0])
   

# %%
