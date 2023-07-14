# %%
import numpy as np
import torch
import ot

#%%
class Backend():
    def __init__(self, device = 'cpu', to_types = 'torch', data_type = 'double'):
        """
        all the variable for GW alignment will be placed on the device, data type and numpy or torch defined by user.
        
        This instance will assist some functions implemented in python file under "src" directory.

        Args:
            device (str, optional): _description_. Defaults to 'cpu'.
            to_types (str, optional): _description_. Defaults to 'torch'.
            data_type (str, optional): _description_. Defaults to 'double'.
        """
        self.device = device
        self.to_types = to_types
        self.data_type = data_type
        pass

    def __call__(self, *args):
        output = []
        for a in args:
            a = self._change_types(a)
            a = self._change_device(a, self.device)
            output.append(a)

        self.nx = ot.backend.get_backend(*output)

        if len(args) == 1:
            output = output[0]

        return output

    def save(self, file, a):
        raise NotImplementedError()

    def load(self, file):
        raise NotImplementedError()

    def _change_types(self, args):
        if self.to_types == 'torch':
            if 'gpu' in self.device:
                raise ValueError('torch uses "cuda" instead of "gpu".')

            if isinstance(args, np.ndarray):
                return torch.from_numpy(args)
            
            else:
                return args

        elif self.to_types == 'numpy':
            if 'cuda' in self.device or 'gpu' in self.device:
                raise ValueError("numpy doesn't work on CUDA")

            if isinstance(args, torch.Tensor):
                return args.to('cpu').numpy()

            else:
                return args

        else:
            raise ValueError("Unknown type of non implemented here.")

    def change_device(self, device, *args):
        device_list = ('cpu', 'cuda')

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
        if isinstance(args, np.ndarray):
            if device == 'cpu':
                if self.data_type == 'double' or self.data_type == 'float64':
                    return args.astype(np.float64)
                elif self.data_type == 'float' or self.data_type == 'float32':
                    return args.astype(np.float32)
            else:
                raise ValueError('Numpy only accepts CPU!!')
    
        elif isinstance(args, torch.Tensor):
            if self.data_type == 'double' or self.data_type == 'float64':
                return args.to(device).double()
            elif self.data_type == 'float' or self.data_type == 'float32':
                return args.to(device).float()
            
        else:
            raise ValueError("Unknown type of non implemented here.")

    def get_item_from_torch_or_jax(self, *args):
        l = []
        for v in args:
            if isinstance(v, torch.Tensor):
                v = v.item()
            l.append(v)

        if len(l) == 1:
            l = l[0]

        return  l

    def save_computed_results(self, gw, file_path, number):
        # save data
        if self.to_types == 'torch':
            torch.save(gw, file_path + f'/gw_{number}.pt')
        elif self.to_types == 'numpy':
            np.save(file_path + f'/gw_{number}', gw)


    def check_zeros(self, args):
        if isinstance(args, torch.Tensor):
            if torch.count_nonzero(args).item() == 0:
                flag = True
            else:
                flag = False

        elif isinstance(args, np.ndarray):
            flag = self.nx.array_equal(args, self.nx.zeros(args.shape))

        return flag


# %%
if __name__ == '__main__':
    test_numpy1 = np.arange(10)
    test_numpy2 = np.arange(10, 20)

    # %%
    backend = Backend(device = 'cuda:3', to_types = 'torch')
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

