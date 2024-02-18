from datetime import timedelta
from typing import List, Union

from datetime import timedelta
from scipy.stats import norm

import numpy as np
import xarray as xr
import pandas as pd
from datetime import timedelta
from typing import List, Dict, Any
from tqdm.auto import tqdm  # For Jupyter notebooks, 'tqdm.auto' automatically selects a suitable interface.

from typing import Any
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

import xarray as xr
import cfgrib 

def find_nearest_values(ds, time, latitude, longitude):
    """
    在数据集中找到给定时间、纬度和经度最近的值。

    Parameters:
    ds (xarray.Dataset): 数据集。
    time (str or datetime-like): 指定的时间。
    latitude (float): 指定的纬度。
    longitude (float): 指定的经度。

    Returns:
    xarray.Dataset: 包含最近点的数据集。
    """
    try:
        nearest_data = ds.sel(time=time, latitude=latitude, longitude=longitude, method='nearest')
        return nearest_data
    except Exception as e:
        print(f"查找最近值时出错: {e}")
        return None


# 应用函数
#dataset = xr.open_dataset("saved_on_disk.nc")
def generate_equal_kernel(size: int, resolution: float, offset: float, datehour: bool = False) -> List[Union[int, timedelta]]:
    """
    Generates a kernel with equal intervals.
    
    For an even size, it generates a list of evenly spaced integers around 0 (excluding 0).
    For an odd size, 0 is included in the list. The kernel can optionally be scaled by a resolution, 
    have an offset added, and be converted into a list of `timedelta` objects representing hours.
    
    Args:
        size (int): Number of elements in the kernel.
        resolution (float): Scale factor to apply to each kernel element.
        offset (float): Value to add to each scaled kernel element.
        datehour (bool, optional): If True, converts kernel elements to `timedelta` objects representing hours. Defaults to False.
    
    Returns:
        List[Union[int, timedelta]]: A list of kernel elements, either integers or `timedelta` objects.
    """
    if size % 2 == 0:  # Even size
        kernel = [i for i in range(-size + 1, size, 2)]
    else:  # Odd size
        kernel = [i for i in range(-size + 1, size + 1, 2)]
    
    kernel = [x * resolution + offset for x in kernel]
    if datehour:
        kernel = [timedelta(hours=h) for h in kernel]
    return kernel


def generate_gaussian_kernel(size: int, mean: float, std_dev: float, datehour: bool = False) -> List[Union[float, timedelta]]:
    """
    Generates a set of points that divide the area under the Gaussian distribution into equal parts.

    This is achieved by dividing the cumulative distribution function (CDF) of the Gaussian distribution
    into (size + 1) equal parts and finding the corresponding x values.

    Args:
        size (int): Number of divisions; (size + 1) equal parts will be created.
        mean (float): Mean of the Gaussian distribution.
        std_dev (float): Standard deviation of the Gaussian distribution.
        datehour (bool, optional): If True, converts the Gaussian points to `timedelta` objects representing hours. Defaults to False.

    Returns:
        List[Union[float, timedelta]]: A list of points dividing the Gaussian distribution, either as floats or `timedelta` objects.
    """
    division_points = [(i / (size + 1)) for i in range(1, size + 1)]
    gaussian_points = [norm.ppf(dp, mean, std_dev) for dp in division_points]
    if datehour:
        gaussian_points = [timedelta(hours=h) for h in gaussian_points]
    return gaussian_points


import xarray as xr
import numpy as np
from tqdm import tqdm 
from sample.retrieval import generate_equal_kernel, generate_gaussian_kernel

from typing import Dict, List, Any
import xarray as xr
def get_patch_by_coordinate(dataset: xr.Dataset, coor_dict: Dict[str, Any], kernel_dict: Dict[str, List[float]], method: str = "nearest") -> xr.Dataset:
    """
    Extracts a patch from the dataset around the specified coordinates using the given kernels.

    This function adjusts the specified coordinates by the provided kernels and then extracts
    a patch from the dataset based on these adjusted coordinates. The patch extraction
    interpolates data points using a specified method.

    Args:
        dataset (xr.Dataset): The xarray dataset from which to extract the patch.
        coor_dict (Dict[str, Any]): A dictionary with the coordinates for extraction.
            The keys should match the dimension names in the dataset, and the values
            should be the coordinates' values (e.g., {"time": "2021-01-01", "latitude": 45.0, "longitude": -120.0}).
        kernel_dict (Dict[str, List[float]]): A dictionary where each key corresponds to a dimension
            in `coor_dict` and each value is a list of offsets (kernel) to apply to the coordinate
            for that dimension (e.g., {"time": [-1, 0, 1], "latitude": [-0.1, 0, 0.1], "longitude": [-0.1, 0, 0.1]}).
        method (str, optional): The interpolation method to use when extracting the patch.
            Defaults to "nearest". Other options include "linear", "cubic", etc., depending on
            the xarray interpolation capabilities.

    Returns:
        xr.Dataset: The extracted patch as an xarray dataset.

    Example:
        >>> dataset = xr.open_dataset("example.nc")
        >>> coor_dict = {"time": np.datetime64("2021-01-01"), "latitude": 45.0, "longitude": -120.0}
        >>> kernel_dict = {"time": [-1, 0, 1], "latitude": [-0.1, 0, 0.1], "longitude": [-0.1, 0, 0.1]}
        >>> patch = get_patch_by_coordinate(dataset, coor_dict, kernel_dict)
        >>> print(patch)
    """
    # Step 1: Generate adjusted dimension lists
    adjusted_dims = {}
    for dim, coord in coor_dict.items():
        kernel = kernel_dict.get(dim)
        if kernel:
            # Create a list of coordinates around the specified coordinate using the kernel
            adjusted_dims[dim] = [coord + k for k in kernel]
         
    # Step 3: Extract the patch using the adjusted dimensions
    patch = dataset.interp(adjusted_dims, method=method)
    
    return patch

# patch = get_patch_by_coordinate(dataset, coor_dict={"time": time, "latitude": latitude, "longitude": longitude},
#                                 kernel_dict={"time": t_kernel, "latitude": lat_kernel, "longitude": long_kernel},
#                                 method="linear")




def Xarrayto0D(dataset: xr.Dataset, coor_dict: Dict[str, Any], method: str = "linear") -> List[np.ndarray]:
    """
    Extracts 0D data patches from the given dataset at specified coordinates.

    Args:
        dataset: The xarray dataset to extract data from.
        coor_list: A list of dictionaries, each specifying the coordinates for a data patch.
        method: The interpolation method to use. Defaults to "linear".

    Returns:
        np.ndarray: A NumPy array containing all extracted patches.
    """
    kernel_dict = {"time": [timedelta(hours=0)],"latitude": [0], "longitude": [0] }
    patch = get_patch_by_coordinate(dataset, coor_dict, kernel_dict, method=method)
    return patch.to_array().values.flatten()
    
def Xarrayto1D(dataset: xr.Dataset, coor_dict: Dict[str, Any], t_kernel: List[timedelta], method: str = "linear") -> List[np.ndarray]:
    """
    Extracts 1D data patches from the given dataset at specified coordinates, varying over time.

    Args:
        dataset: The xarray dataset to extract data from.
        coor_list: A list of dictionaries, each specifying the coordinates for a data patch.
        t_kernel: A list of time offsets to define the kernel for time dimension extraction.
        method: The interpolation method to use. Defaults to "linear".

    Returns:
        np.ndarray: A NumPy array containing all extracted patches.
    """
 
    kernel_dict = {"time": t_kernel,"latitude": [0], "longitude": [0] }
    patch = get_patch_by_coordinate(dataset, coor_dict, kernel_dict, method=method)
    return patch.to_array().values.squeeze()

def Xarrayto2D(dataset: xr.Dataset, coor_dict: Dict[str, Any], lat_kernel: List[float], long_kernel: List[float], method: str = "linear") -> List[np.ndarray]:
    """
    Extracts 2D spatial data patches from the given dataset at specified coordinates.

    Args:
        dataset: The xarray dataset to extract data from.
        coor_list: A list of dictionaries, each specifying the coordinates for a data patch.
        lat_kernel: A list of latitude offsets to define the kernel for latitude dimension extraction.
        long_kernel: A list of longitude offsets to define the kernel for longitude dimension extraction.
        method: The interpolation method to use. Defaults to "linear".

    Returns:
        np.ndarray: A NumPy array containing all extracted patches.
    """
    kernel_dict = {"time": [timedelta(hours=0)],"latitude": lat_kernel, "longitude": long_kernel}
    patch = get_patch_by_coordinate(dataset, coor_dict, kernel_dict, method=method)
    return np.squeeze(patch.to_array().values, axis=1)  # Remove time dimension if present

def Xarrayto3D(dataset: xr.Dataset, coor_dict: Dict[str, Any], t_kernel: List[timedelta], lat_kernel: List[float], long_kernel: List[float], method: str = "linear") -> List[np.ndarray]:
    """
    Extracts 3D spatiotemporal data patches from the given dataset at specified coordinates.

    Args:
        dataset: The xarray dataset to extract data from.
        coor_list: A list of dictionaries, each specifying the coordinates for a data patch.
        t_kernel: A list of time offsets to define the kernel for time dimension extraction.
        lat_kernel: A list of latitude offsets to define the kernel for latitude dimension extraction.
        long_kernel: A list of longitude offsets to define the kernel for longitude dimension extraction.
        method: The interpolation method to use. Defaults to "linear".

    Returns:
        np.ndarray: A NumPy array containing all extracted patches.
    """
    kernel_dict = {"time": t_kernel, "latitude": lat_kernel, "longitude": long_kernel}
    patch = get_patch_by_coordinate(dataset, coor_dict, kernel_dict, method=method)
    return patch.to_array().values



import pandas as pd
import xarray as xr
import numpy as np
from tqdm.auto import tqdm

def batch_process_xarray_to_tabular(df: pd.DataFrame, ds: xr.Dataset, batch_size: int = 100) -> pd.DataFrame:
    # Prepare the DataFrame by setting the correct MultiIndex
    df_ind = df.set_index(['Site_number', 'time'], inplace=False)
    
    # Initialize an empty DataFrame for the results
    result_df = pd.DataFrame()
    
    if 'Negative_oxygen_ions' in ds.variables:
        # Drop the variable 'Negative_oxygen_ions' from the dataset
        ds = ds.drop_vars('Negative_oxygen_ions')
        
    variable_names = list(ds.data_vars)
    
    # Process in batches
    num_batches = np.ceil(len(df) / batch_size)
    
    for i in tqdm(range(int(num_batches)), desc="Processing batches"):
        # Calculate start and end indices for the current batch
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        # Select the current batch from the indexed DataFrame    
        df_batch = df_ind.iloc[start_idx:end_idx]
        
        # Select data from the xarray dataset for the current batch
        ds_sel = ds.sel(time=df_batch.index.get_level_values('time'),
                        latitude=df_batch['latitude'].unique(), 
                        longitude=df_batch['longitude'].unique(),
                        method='nearest')
        
        # Convert the selected xarray Dataset to a DataFrame
        ds_df = ds_sel.to_dataframe().reset_index()
        ds_df=ds_df[variable_names+['time', 'latitude', 'longitude']]
        # Merge the xarray-derived DataFrame with the current batch DataFrame
        merged_batch = pd.merge(df_batch.reset_index(), ds_df, 
                                left_on=['time', 'latitude', 'longitude'], 
                                right_on=['time', 'latitude', 'longitude'],
                                how='left').set_index('Site_number')
        
        # Append the processed batch to the result DataFrame
        result_df = pd.concat([result_df, merged_batch])
    
    
    return result_df.reset_index()
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

def Xarray2Tabular(ds: xr.Dataset, df: pd.DataFrame) -> pd.DataFrame:
    # Drop 'Negative_oxygen_ions' from ds if it exists
    if 'Negative_oxygen_ions' in ds:
        ds = ds.drop_vars('Negative_oxygen_ions')
    
    # Prepare the final DataFrame
    final_df = pd.DataFrame()
    
    # Group df by 'Site_number'
    grouped = df.groupby('Site_number')

    # Process other Site_numbers in batches
    for site_number, group in tqdm(grouped, desc="Processing sites"):
        if site_number == 0:
            continue  # Skip Site_number = 0 as it's already processed
        longitude = group['longitude'].iloc[0]
        latitude = group['latitude'].iloc[0]
        group = group.sort_values("time")
        group['time'] = pd.to_datetime(group['time'])
        
        # Select and interpolate subdataset
        ds_sub = ds.sel(longitude=longitude, latitude=latitude, method='nearest')
        interpolated_ds = ds_sub.interp(time=group['time'], method='nearest')
        
        # Convert to DataFrame and merge
        interpolated_df = interpolated_ds.to_dataframe().reset_index()
        interpolated_df.index = group.index
        merged_group = pd.concat([group, interpolated_df.drop(['time', 'latitude', 'longitude'], axis=1)], axis=1)
        
        # Append to final DataFrame
        final_df = pd.concat([final_df, merged_group], ignore_index=True)

    
    # Special handling for Site_number = 0
    if 0 in grouped.groups:
        site_0_group = grouped.get_group(0)
        for index, row in tqdm(site_0_group.iterrows(), desc="Processing Site_number=0", total=len(site_0_group)):
            longitude = row['longitude']
            latitude = row['latitude']
            time = pd.to_datetime(row['time'])  # Ensure datetime format
            ds_sub = ds.sel(longitude=[longitude], latitude=[latitude], time=[time], method='nearest')

            if ds_sub.time.size != 0:
                
                interpolated_df = ds_sub.to_dataframe().reset_index()
                interpolated_df.index = [index]  # Match index for merging
                # Merge and append to final_df
                final_df = pd.concat([final_df, pd.concat([row.to_frame().T, interpolated_df.drop(['time', 'latitude', 'longitude'], axis=1)], axis=1)], ignore_index=True)
    


        
    # Drop specified columns and return
    return final_df.drop(columns=['spatial_ref', 'number', 'step', 'entireAtmosphere', 'surface', 'meanSea'], errors='ignore')
