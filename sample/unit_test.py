import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import rasterio 
import numpy as np
import h5py
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
def get_xarray_memory_usage_gb(dataset):
    total_bytes = sum(data_array.nbytes for data_array in dataset.data_vars.values())
    total_gigabytes = total_bytes / (1024**3) # Convert bytes to gigabytes
    return total_gigabytes
def print_file_structure(file_path):
    """
    Opens an HDF5 file, prints its structure and the size of its datasets.

    Args:
        file_path (str): The path to the HDF5 file.
    """
    # Ensure the file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    with h5py.File(file_path, 'r') as f:
        print(f"Structure of {file_path}:")
        for key in f.keys():
            print(f" - Dataset '{key}', Shape: {f[key].shape}, Dtype: {f[key].dtype}")

def generate_example(dimensions, type="numpy"):
    """
    Generates data with random values and corresponding labels in specified format.

    Args:
        dimensions (tuple): Dimensions for the data generation.
            For 0D: (q, c)
            For 1D: (q, t, c)
            For 2D: (q, m, n, c)
            For 3D: (q, m, n, t, c)
        type (str): Output format ("numpy", "tensor", "dataset", "dataloader"). Defaults to "numpy".

    Returns:
        Varies by `type`: NumPy array, PyTorch tensor, PyTorch Dataset, or PyTorch DataLoader.
    """
    # Unpack dimensions based on length
    q = dimensions[0]  # Number of samples is always the first element
    if len(dimensions) == 2:
        data_shape = (q, ) + dimensions[1:]  # For 0D case
    else:
        data_shape = (q, ) + dimensions[1:]  # For 1D, 2D, and 3D cases
    
    # Generate data
    data = np.random.uniform(-1, 1, data_shape)
    labels = np.random.uniform(0, 3000, (q, 1))  # Assuming label generation similar to previous cases

    if type == "numpy":
        return data, labels
    elif type == "tensor":
        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
    elif type == "dataset" or type == "dataloader":
        dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))
        if type == "dataset":
            return dataset
        elif type == "dataloader":
            return DataLoader(dataset, batch_size=16, shuffle=True)


import h5py
import numpy as np
from tqdm import tqdm
import os

def generate_large_dataset(dimensions, save_dir, name="dataset"):
    """
    Generates a dataset of specified dimensions and saves it as an HDF5 file with a given name.

    Args:
        dimensions (tuple): The dimensions for the dataset. Supports 0D to 3D datasets:
                            For 0D: (q, c)
                            For 1D: (q, t, c)
                            For 2D: (q, m, n, c)
                            For 3D: (q, m, n, t, c)
        save_dir (str): The directory where the HDF5 file will be saved.
        name (str): Base name of the saved HDF5 file.

    Returns:
        str: The absolute path of the saved HDF5 file.

    Note:
        The HDF5 file is organized into two main datasets:
        - 'data': Contains the generated data with random values uniformly distributed between -1 and 1.
                  The shape of 'data' depends on the specified dimensions, supporting structures from 0D to 3D.
                  For example, a 2D dataset would have the shape (q, m, n, c), where
                  q is the number of samples, m and n are spatial dimensions, and c is the channel dimension.
        - 'labels': A 1D dataset that contains labels for each sample, with values uniformly distributed between 0 and 3000.
                    The shape of 'labels' is (q,), where q is the number of samples.
        The dataset is saved in the specified directory with the filename constructed from the `name` parameter, resulting in '{name}.h5'.
    """
    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{name}.h5")

    # Adjusting dimensions for data and label creation
    q = dimensions[0]  # Number of samples
    data_shape = (q, ) + dimensions[1:]  # Data shape based on provided dimensions

    with h5py.File(file_path, "w") as f:
        dset = f.create_dataset("data", data_shape, dtype='f')
        labels = f.create_dataset("labels", (q,), dtype='f')
        for i in tqdm(range(q)):
            if len(dimensions) == 2:  # For 0D case
                dset[i, ...] = np.random.uniform(-1, 1, dimensions[1:])
            else:
                dset[i, ...] = np.random.uniform(-1, 1, dimensions[1:])
            labels[i] = np.random.uniform(0, 3000)  # Assuming label values between 0 and 3000

    return os.path.abspath(file_path)
import os

def find_files_with_extension(directory, extension):
    """
    Finds all files with the given extension in the specified directory.
    
    Parameters:
    - directory (str): The path to the directory where to look for files.
    - extension (str): The file extension to look for. Include the dot, e.g., '.txt'.
    
    Returns:
    - list: A list of paths to files matching the given extension in the directory.
    """
    # Ensure the extension starts with a dot
    if not extension.startswith('.'):
        extension = '.' + extension

    # List all files in the given directory and filter by extension
    files_with_extension = [os.path.join(directory, file) for file in os.listdir(directory)
                            if os.path.isfile(os.path.join(directory, file)) and file.endswith(extension)]

    return files_with_extension