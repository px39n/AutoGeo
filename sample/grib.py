import eccodes
from tqdm import tqdm
import os
import joblib
import rioxarray
def get_shortname_edition(path: str) -> dict:
    """
    Collects all unique shortName keys for each edition in a GRIB file.

    Args:
        path (str): The path to the GRIB file.

    Returns:
        dict: A dictionary where each key is an edition and the value is a list of unique shortNames for that edition.
    """
    # Dictionary to collect all unique shortName keys by their edition
    edition_shortnames = {}

    # Open the GRIB file in binary read mode
    with open(path, 'rb') as f:
        # Determine the total size for progress estimation
        total_size = os.path.getsize(path)

        # Initialize the progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Analyzing GRIB file") as pbar:
            while True:
                # Decode the next GRIB message
                gid = eccodes.codes_grib_new_from_file(f)
                if gid is None:
                    break  # No more messages
                
                # Get the shortName and edition for the current message
                short_name = eccodes.codes_get(gid, 'shortName')
                edition = eccodes.codes_get(gid, 'edition')
                
                # Initialize the list for this edition if not already present
                if edition not in edition_shortnames:
                    edition_shortnames[edition] = set()

                # Add the shortName to the set for this edition
                edition_shortnames[edition].add(short_name)
                
                # Release the current GRIB message to free memory
                eccodes.codes_release(gid)
                
                # Update the progress bar based on the file's current position
                current_position = f.tell()
                pbar.update(current_position - pbar.n)
    
    # Convert sets to lists for the output
    for edition in edition_shortnames:
        edition_shortnames[edition] = list(edition_shortnames[edition])

    return edition_shortnames
import sys  
import eccodes
from tqdm import tqdm
import os

import eccodes
import numpy as np
import xarray as xr

import eccodes
import eccodes
from tqdm import tqdm
import os 
import eccodes
import os
from tqdm import tqdm


    
import matplotlib.pyplot as plt

def plot_grib_null(count_stat: dict) -> None:
    """
    Plots the distribution of null counts in given categories.

    Args:
        count_stat (dict): Dictionary of counts of missing values per variable.
    """
    categories = {'<10': [],'10-25': [], '25-50': [], '50-100': [], '100-1000': [], '>1000': []}

    for var, stats in count_stat.items():
        missing = stats['missing']
        
        if 0 <  missing < 10:
            categories['<10'].append(var)     
        elif 10 <= missing <= 25:    
            categories['10-25'].append(var)
        elif 25 < missing <= 50:
            categories['25-50'].append(var)
        elif 50 < missing <= 100:
            categories['50-100'].append(var)
        elif 100 < missing <= 1000:
            categories['100-1000'].append(var)
        elif missing > 1000:
            categories['>1000'].append(var)
 

    # For simplicity, here we print the categories, but you can adjust to plot as needed
    for cat, vars in categories.items():
        print(f'{cat}: {", ".join(vars)}')
    return categories
  

def get_unique_variables_from_grib(file_path):
    unique_variables = set()

    # Open the GRIB file
    with open(file_path, 'rb') as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break  # End of the file

            # Get the short name of the current variable
            short_name = eccodes.codes_get(gid, 'shortName')
            unique_variables.add(short_name)

            # Release the current GRIB message to free memory
            eccodes.codes_release(gid)

    return unique_variables

# Example usage
#file_path = filterd_grib_path_step
#unique_vars = get_unique_variables_from_grib(file_path)
#print("Unique variables in the GRIB file:", unique_vars)
import eccodes
import numpy as np
import xarray as xr

import eccodes
import eccodes
from tqdm import tqdm
import os 
import eccodes
import os
from tqdm import tqdm

def check_null_and_step_edition(file_path: str) -> dict:
    """
    Counts the missing values for each shortName in a GRIB file, checks if they have a coordinate step,
    and collects all unique shortName keys for each edition.

    Args:
        file_path (str): The path to the GRIB file.

    Returns:
        dict: Three dictionaries inside a dict:
              - 'summary': With shortName as keys and a sub-dictionary as values, where the sub-dictionary contains
                'missing' and 'total' counts.
              - 'summary2': With keys 'has_step' and 'hasnt_step' listing unique shortNames according to whether
                they have a coordinate step or not.
              - 'summary3': Similar to edition_shortnames, where each key is an edition and the value is a list of
                unique shortNames for that edition.
    """
    summary = {}
    has_step = set()
    hasnt_step = set()
    edition_shortnames = {}  # New dictionary for collecting unique shortNames by edition

    # Determine the total size of the input file for progress estimation
    total_size = os.path.getsize(file_path)

    with open(file_path, 'rb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc="Analyzing Edition") as pbar:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break  # End of file
            
            short_name = eccodes.codes_get(gid, 'shortName')
            edition = eccodes.codes_get(gid, 'edition')  # Retrieve the edition for the current message
            values = eccodes.codes_get_values(gid)
            missing_value = eccodes.codes_get(gid, 'missingValue')
            
            missing_count = (values == missing_value).sum()
            total_values = len(values)
            
            if short_name not in summary:
                summary[short_name] = {'missing': 0, 'total': 0}
            summary[short_name]['missing'] += missing_count
            summary[short_name]['total'] += total_values

            # Initialize the list for this edition if not already present
            if edition not in edition_shortnames:
                edition_shortnames[edition] = set()
            edition_shortnames[edition].add(short_name)  # Add the shortName to the set for this edition

            # Check for coordinate step
            try:
                eccodes.codes_get(gid, 'step')
                has_step.add(short_name)
            except Exception:
                if short_name not in has_step:
                    hasnt_step.add(short_name)
            
            eccodes.codes_release(gid)
            
            # Update the progress bar based on the current file read position
            current_position = f.tell()
            pbar.update(current_position - pbar.n)

    # Convert sets to lists for the output
    for edition in edition_shortnames:
        edition_shortnames[edition] = list(edition_shortnames[edition])
    summary2 = {"has_step": list(has_step), "hasnt_step": list(hasnt_step)}
    summary3 = edition_shortnames  # Convert the edition_shortnames dict to the required format

    return {'null': summary, 'step': summary2, 'edition': summary3}



    
import matplotlib.pyplot as plt

def plot_grib_null(count_stat: dict) -> None:
    """
    Plots the distribution of null counts in given categories.

    Args:
        count_stat (dict): Dictionary of counts of missing values per variable.
    """
    categories = {'<10': [],'10-25': [], '25-50': [], '50-100': [], '100-1000': [], '>1000': []}

    for var, stats in count_stat.items():
        missing = stats['missing']
        
        if 0 <  missing < 10:
            categories['<10'].append(var)     
        elif 10 <= missing <= 25:    
            categories['10-25'].append(var)
        elif 25 < missing <= 50:
            categories['25-50'].append(var)
        elif 50 < missing <= 100:
            categories['50-100'].append(var)
        elif 100 < missing <= 1000:
            categories['100-1000'].append(var)
        elif missing > 1000:
            categories['>1000'].append(var)
 

    # For simplicity, here we print the categories, but you can adjust to plot as needed
    for cat, vars in categories.items():
        print(f'{cat}: {", ".join(vars)}')
    return categories
    
from typing import List, Tuple, Set
import eccodes
from tqdm import tqdm
import os

def filter_grib_file(input_grib_file_path: str, variables_to_delete: List[str],name: str="_filtered" ) -> Tuple[str, Set[str]]:
    """
    Filters out specified variables from a GRIB file and saves the result to a new file. Also returns a set of remaining variables.

    Args:
        input_grib_file_path (str): The path to the input GRIB file.
        variables_to_delete (List[str]): A list of variable names (shortName) to be excluded from the output file.
        name: name: str="_filtered.grib"  as default
    Returns:
        Tuple[str, Set[str]]: The path to the filtered output GRIB file and a set of variables that were not excluded.
    """
    # Define the output file path based on the input file path
    new_dir = os.path.join(os.path.dirname(input_grib_file_path), name)
    os.makedirs(new_dir, exist_ok=True)
    output_grib_file_path = os.path.join(new_dir, os.path.basename(input_grib_file_path))

    # Initialize a set to keep track of remaining variables
    remaining_variables = set()

    # Open the input GRIB file
    with open(input_grib_file_path, 'rb') as input_file, open(output_grib_file_path, 'wb') as output_file:
        # Determine the total size of the input file for progress estimation
        total_size = os.path.getsize(input_grib_file_path)

        # Initialize the progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Filtering") as pbar:
            while True:
                # Use eccodes to read each message from the input GRIB file
                gid = eccodes.codes_grib_new_from_file(input_file)
                if gid is None:
                    break  # Break the loop if we have reached the end of the file
                
                # Retrieve shortName of the current parameter
                short_name = eccodes.codes_get(gid, 'shortName')
                
                # If the variable is not in the delete list, write it to the output file and add to remaining variables
                if short_name not in variables_to_delete:
                    coded_message = eccodes.codes_get_message(gid)
                    output_file.write(coded_message)
                    remaining_variables.add(short_name)
                    
                # Release the current message to avoid memory leaks
                eccodes.codes_release(gid)
                
                # Update the progress bar
                current_position = input_file.tell()
                pbar.update(current_position - pbar.n)  # Update progress based on the amount of data processed

    print(f"Filtered GRIB file has been saved to: {output_grib_file_path}")
    return output_grib_file_path, remaining_variables

import eccodes

def get_unique_variables_from_grib(file_path):
    unique_variables = set()

    # Open the GRIB file
    with open(file_path, 'rb') as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break  # End of the file

            # Get the short name of the current variable
            short_name = eccodes.codes_get(gid, 'shortName')
            unique_variables.add(short_name)

            # Release the current GRIB message to free memory
            eccodes.codes_release(gid)

    return unique_variables

# Example usage
#file_path = filterd_grib_path_step
#unique_vars = get_unique_variables_from_grib(file_path)
#print("Unique variables in the GRIB file:", unique_vars)
import eccodes
from tqdm import tqdm
import xarray as xr
import cfgrib
import pandas as pd

def merge_time_and_step(ds):
    """
    将 'step' 维度合并到 'time' 维度中。

    Parameters:
    ds (xarray.Dataset): 原始数据集。

    Returns:
    xarray.Dataset: 转换后的数据集。
    """
    # 创建一个新的空的列表来存储新的时间和相应的数据

    if 'step' in ds:
        new_times = []
        data_arrays = []
    
        # 遍历每个 time 和 step 的组合
        for t in tqdm(ds['time'].values, desc='Merging Step'):
            for s in ds['step'].values:
                new_time = pd.to_datetime(t) + pd.to_timedelta(s)
                new_times.append(new_time)
    
                # 选择对应的数据并更新时间
                da = ds.sel(time=t, step=s)
                da = da.expand_dims(time=[new_time])
                data_arrays.append(da)
    
        # 合并所有的数据数组
        combined = xr.concat(data_arrays, dim='time')
    
        # 删除 'step' 维度
        del combined['step']
    # 如果 'edition' 维度存在，则删除
    if 'edition' in combined.dims or 'edition' in combined:
        combined = combined.drop_vars('edition', errors='ignore')
        
    return combined
def detect_step_ds(filtered,shortname_list):
    variable_names_no_step=set()
    variable_names_step=set()
    ds_step_flag=0
    ds_no_step_flag=0
    for name in tqdm(shortname_list, desc='Detection Step') :
         
        ds =xr.open_dataset(filtered, engine='cfgrib',
                             backend_kwargs={'filter_by_keys':{'shortName': name}})
        if len(ds.sizes)==3:
            variable_names_no_step.add(name)     
            if ds_no_step_flag:
                ds_no_step= xr.merge([ds_no_step, ds],compat='override') #
            else:
                ds_no_step=ds
                ds_no_step_flag=1
                
        elif len(ds.sizes)==4:
            variable_names_step.add(name) 
            if ds_step_flag:
                ds_step= xr.merge([ds_step, ds],compat='override')
            else:
                ds_step=ds
                ds_step_flag=1
        else:
            pass
            #print(name)
            #print(len(ds.sizes))
    return ds_step, ds_no_step



def load_era(grib_file_path):
    new_dir = os.path.join(os.path.dirname(grib_file_path), "_filtered")
    filterd_grib_path= os.path.join(new_dir, os.path.basename(grib_file_path))
    filterd_sl_path=filterd_grib_path.replace(".grib",".pkl")
    filterd_nc_path=filterd_grib_path.replace(".grib",".nc")
    if os.path.exists(filterd_grib_path):
        print("========filter file detected========")
        shortname_list=joblib.load(filterd_sl_path)
    else:
        count_stat = check_null_and_step_edition(grib_file_path)
        
        # Example usage
        #interval_stat = plot_grib_null(count_stat["null"])
        
        # remove the variable which has more than 100 null value and edition.
        variables_to_delete=set(count_stat["edition"][2]) #interval_stat[">1000"]+
        filterd_grib_path, shortname_list=filter_grib_file(grib_file_path,variables_to_delete)
        joblib.dump(shortname_list,filterd_sl_path)
    
    #ds=xr.open_dataset(filterd_grib_path, engine='cfgrib') # Build index
    #print(shortname_list)
    #print(interval_stat) 
    
    if os.path.exists(filterd_nc_path):
        print("========merged nc file detected========")
        dataset = xr.open_dataset(filterd_nc_path)
    else:
        ds_step, ds_no_step=detect_step_ds(filterd_grib_path,shortname_list)     
        ds_step_merged=merge_time_and_step(ds_step)
        dataset = xr.merge([ds_step_merged, ds_no_step],compat='override')
        dataset.to_netcdf(filterd_nc_path)
        
    return dataset.rio.write_crs("EPSG:4326", inplace=True)

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

def concatenate_ds_by_time(ds1,ds2):


        
    # Assuming ds1 and ds2 are your datasets and they are sorted by time
    ds1 = ds1.sortby('time')
    ds2 = ds2.sortby('time')
    
    # Convert time coordinates to pandas datetime for easy comparison
    # ds1_times = pd.to_datetime(ds1.time.values)
    # ds2_times = pd.to_datetime(ds2.time.values)
    
    # Find common times between ds1 and ds2
    common_times = np.intersect1d(ds1.time.values, ds2.time.values)
    
    if len(common_times) > 0:
        # There is an overlap
        # Select overlapping periods in both datasets using the common times
        ds1_overlap = ds1.sel(time=common_times)
        ds2_overlap = ds2.sel(time=common_times)
        
        # Merge variables from ds1_overlap and ds2_overlap, prioritizing ds1 and filling in missing variables from ds2
        def merge_variables(ds1_var, ds2_var):
            if ds1_var is not None and ds2_var is not None:
                # Both ds1 and ds2 have the variable, prioritize non-NaN values from ds1
                return xr.where(np.isnan(ds1_var), ds2_var, ds1_var)
            elif ds1_var is not None:
                # Only ds1 has the variable
                return ds1_var
            elif ds2_var is not None:
                # Only ds2 has the variable
                return ds2_var
        
        # Create a combined dataset for the overlapping period with prioritization logic
        combined_overlap_vars = {}
        for var in set(ds1_overlap.data_vars).union(ds2_overlap.data_vars):
            ds1_var = ds1_overlap.data_vars.get(var)
            ds2_var = ds2_overlap.data_vars.get(var)
            combined_overlap_vars[var] = merge_variables(ds1_var, ds2_var)
        
        combined_overlap = xr.Dataset(combined_overlap_vars)
        
        # Concatenate non-overlapping parts with the combined overlap
        ds1_non_overlap = ds1.sel(time=~ds1.time.isin(combined_overlap.time.values))
        ds2_non_overlap = ds2.sel(time=~ds2.time.isin(combined_overlap.time.values))
        
        shared_coords = set(ds1_non_overlap.coords) & set(combined_overlap.coords) & set(ds2_non_overlap.coords)
        unshared_coords = (set(ds1_non_overlap.coords) | set(combined_overlap.coords) | set(ds2_non_overlap.coords)) - shared_coords
            
        # Remove unshared coordinates from each dataset
        ds1_non_overlap = ds1_non_overlap.drop_vars(list(unshared_coords), errors='ignore')
        combined_overlap = combined_overlap.drop_vars(list(unshared_coords), errors='ignore')
        ds2_non_overlap = ds2_non_overlap.drop_vars(list(unshared_coords), errors='ignore')
    
        # Concatenate the datasets
        ds_combined = xr.concat([ds1_non_overlap, combined_overlap, ds2_non_overlap], dim='time')
    else:
        # No overlap, just concatenate ds1 and ds2 (assuming ds2 starts after ds1 ends)
        ds_combined = xr.concat([ds1, ds2], dim='time')
    
    # Optionally, sort by time in case of any misalignment
    ds_combined = ds_combined.sortby('time')
     
    return ds_combined
    
def load_era5_batch(directory):
    extension = '.grib'
    matching_files = find_files_with_extension(directory, extension)
    
    for i, grib_file_path in enumerate(matching_files):
        print(f"[autoGEO][Info] Process {i+1}th file in {len(matching_files)}")
        if i==0:
            merged_ds=load_era(grib_file_path)
        else:
            ds=load_era(grib_file_path)
            merged_ds=concatenate_ds_by_time(merged_ds,ds)
    
    
    return merged_ds.rio.write_crs("EPSG:4326", inplace=True)