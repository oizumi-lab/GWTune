# %%
from typing import Any, List, Union

import numpy as np
import ot
import torch


#%%
class Backend():
    """A class that manages data types and device-specific operations for Gromov-Wasserstein alignment.

    This instance will assist some functions implemented in python file under "src" directory, adjusting
    variables for GW alignment based on user-defined device, data type, and either numpy or torch settings.

    Attributes:
        device (str): The device to be used for computation, either "cpu" or "cuda".
        to_types (str): Specifies the data structure to be used, either 'torch' or 'numpy'.
        data_type (str): Specifies the type of data to be used in computation.
        nx (backend module): The backend module from POT (Python Optimal Transport) corresponding to the data type.
    """

    def __init__(self, device: str = 'cpu', to_types: str = 'torch', data_type: str = 'double') -> None:
        """Initializes the Backend class.

        Args:
            device (str, optional): The device to be used for computation, either "cpu" or "cuda". Defaults to 'cpu'.
            to_types (str, optional): Specifies the data structure to be used, either 'torch' or 'numpy'. Defaults to 'torch'.
            data_type (str, optional): Specifies the type of data to be used in computation. Defaults to 'double'.
        """

        self.device = device
        self.to_types = to_types
        self.data_type = data_type
        pass

    def __call__(self, *args) -> Union[List[Any], Any]:
        """Convert the provided data to the specified data type and device.

        Args:
            *args: The data items to be converted.

        Returns:
            Union[List[Any], Any]: Converted data items. If only one item is provided, it's returned directly, else a list.
        """
        output = []
        for a in args:
            a = self._change_types(a)
            a = self._change_device(a, self.device)
            output.append(a)

        self.nx = ot.backend.get_backend(*output)

        if len(args) == 1:
            output = output[0]

        return output

    def save(self, file: str, a: Any) -> None:
        """Save the provided data to a file.

        Args:
            file (str): Path to the file.
            a (Any): Data to be saved.

        Raises:
            NotImplementedError: If the method is not implemented.
        """

        raise NotImplementedError()

    def load(self, file: str) -> None:
        """Load data from a file.

        Args:
            file (str): Path to the file.

        Raises:
            NotImplementedError: If the method is not implemented.
        """

        raise NotImplementedError()

    def _change_types(self, args) -> Any:
        """Convert the data type of the provided data.

        Args:
            args (Any): Data to be converted.

        Returns:
            Any: Converted data.

        Raises:
            ValueError: If the provided data type and device do not match.
        """

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

    def change_device(self, device: str, *args) -> Union[List[Any], Any]:
        """Change the device of the provided data.

        Args:
            device (str): The target device.
            *args: Data items to be moved to the specified device.

        Returns:
            Union[List[Any], Any]: Data on the new device. If only one item is provided, it's returned directly, else a list.
        """

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

    def _change_device(self, args: Any, device: str = 'cpu') -> Any:
        """Helper method to change the device of the provided data.

        Args:
            args (Any): Data to be moved.
            device (str, optional): Target device. Defaults to 'cpu'.

        Returns:
            Any: Data on the new device.
        """

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

    def get_item_from_torch_or_jax(self, *args) -> Union[List[float], float]:
        """Retrieve scalar values from tensors.

        Args:
            *args: Data items, either tensors or other.

        Returns:
            Union[List[float], float]: Scalar values. If only one item is provided, it's returned directly, else a list.
        """

        l = []
        for v in args:
            if isinstance(v, torch.Tensor):
                v = v.item()
            l.append(v)

        if len(l) == 1:
            l = l[0]

        return  l

    def save_computed_results(self, gw: Any, file_path: str, number: int) -> None:
        """Save computed Gromov-Wasserstein results to a file.

        Args:
            gw (Any): Array-like, shape (n_source, n_target). Computed Gromov-Wasserstein matrix.
            file_path (str): Base path for saving results.
            number (int): A number to distinguish saved results.
        """

        # save data
        if self.to_types == 'torch':
            torch.save(gw, file_path + f'/gw_{number}.pt')
        elif self.to_types == 'numpy':
            np.save(file_path + f'/gw_{number}', gw)


    def check_zeros(self, args: Any) -> bool:
        """Check if the provided data contains only zeros.

        Args:
            args (Any): Data to be checked.

        Returns:
            bool: True if the data contains only zeros, False otherwise.
        """

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
