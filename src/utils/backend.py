# %%
import os, re
import numpy as np
import jax
import jax.numpy as jnp
import torch
import random
import ot
import warnings
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
    
    def __call__(self, *args):
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
            a = self._change_device(a, self.device)
            output.append(a)
            
        self.nx = ot.backend.get_backend(*output) #ここでtorchをcpuにしていても、GPUにデータが載ってしまうバグがある。
        
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
                return jnp.asarray(args.to('cpu'))
            
            if isinstance(args, np.ndarray):
                return jnp.asarray(args)
    
        elif self.to_types == 'torch':
            if 'gpu' in self.device:
                raise ValueError('torch uses "cuda" instead of "gpu".')
            
            if isinstance(args, np.ndarray):
                return torch.from_numpy(args).float()

            elif isinstance(args, jax.numpy.ndarray):
                args = np.array(args)
                return torch.from_numpy(args).float()

            else:
                return args

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
        
    def change_device(self, device, *args):
        device_list = ('cpu', 'cuda', 'gpu')

        if not device.startswith(device_list):
            raise ValueError('no device is assigned to change')
        
        output = []
        
        for a in args:
            a = self._change_device(a, device)
            output.append(a)
            
        if len(args) == 1:
            output = output[0]
        
        return output
    
    def _change_device(self, args, device = 'cpu'):
        """変数のGPUやCPUを指定したdeviceに変更する。"""
            
        if isinstance(args, np.ndarray):
            if device == 'cpu':
                return args
            else:
                raise ValueError('Numpy only accepts CPU!!')
        
        elif isinstance(args, jax.numpy.ndarray):
            if 'cuda' in device or 'gpu' in device:
                digits = re.sub('[^0-9]', '', device)
            
                if digits == '':
                    digits = 0
                else:
                    digits = int(digits)
                
                gpus = jax.devices("gpu")
                return jax.device_put(args, gpus[digits])
            
            else:
                cpus = jax.devices("cpu")
                return jax.device_put(args, cpus[0])

        elif isinstance(args, torch.Tensor):
            return args.to(device)
        
        else:
            raise ValueError("Unknown type of non implemented here.")
    
    def get_item_from_torch_or_jax(self, *args):
        l = []
        for v in args:
            if isinstance(v, torch.Tensor) or isinstance(args, jax.numpy.ndarray):
                v = v.item()
            l.append(v)
        
        if len(l) == 1:
            l = l[0]
            
        return  l

    def save_computed_results(self, gw, init_mat, file_path, number):
        # save data
        if self.to_types == 'torch':
            torch.save(gw, file_path + f'/gw_{number}.pt')
            torch.save(init_mat, file_path + f'/init_mat_{number}.pt')

        elif self.to_types == 'numpy':
            np.save(file_path + f'/gw_{number}', gw)
            np.save(file_path + f'/init_mat_{number}', init_mat)

        elif self.to_types == 'jax':
            # jaxの保存方法を作成してください 
            pass
        
    def check_zeros(self, args):
        if isinstance(args, torch.Tensor):
            if torch.count_nonzero(args).item() == 0:
                flag = True
            else:
                flag = False
        
        elif isinstance(args, np.ndarray):
            flag = self.nx.array_equal(args, self.nx.zeros(args.shape))
        
        elif isinstance(args, jax.numpy.ndarray):
            pass
            # まだよくわからない・・・(2023/3/17 佐々木)
            # device_id = jax.devices(args.sharding)#[0].device_id
            # check_data = self.nx.zeros(args.shape)
            # check_data = jax.device_get(check_data, device_id)
            # flag = self.nx.array_equal(args, check_data)
            
        
        return flag
   

# %%
if __name__ == '__main__':
    test_numpy1 = np.arange(10)
    test_numpy2 = np.arange(10, 20)
    
    # %%
    backend = Backend(device = 'cuda:3', to_types = 'jax')
    test_jax1, test_jax2 = backend(test_numpy1, test_numpy2)
    print(test_jax1, test_jax2)
    print(type(test_jax1), type(test_jax2))
    # %%
    backend.check_zeros(test_jax1)
    # %%
    test_jax1 = backend.change_device('cpu', test_jax1)
    print(type(test_jax1))
    
    # %%
    backend.check_zeros(test_jax1)
    # %%
    backend = Backend(device = 'cuda', to_types = 'torch')
    test_torch1 = backend.change_data(test_numpy1)
    print(test_torch1)
    print(type(test_torch1))
   

# %%

# %%
