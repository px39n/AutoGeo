import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import rasterio.features
import h5py
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from typing import List, Tuple
import xarray as xr
import geopandas as gpd
from affine import Affine
import numpy as np
import xarray as xr
from typing import Union
import joblib
import os
import rioxarray

def make_mask(ds: xr.Dataset, shp: gpd.GeoDataFrame) -> xr.DataArray:
    """
    Creates a mask for an xarray dataset based on the geometry of a given shapefile (GeoDataFrame).

    Args:
        ds (xr.Dataset): The input xarray dataset containing 'ELongtitude' and 'NLatitude' coordinates.
        shp (gpd.GeoDataFrame): A GeoDataFrame containing the geometries to mask the dataset.

    Returns:
        xr.DataArray: A boolean mask DataArray where True represents the presence of the geometry and False represents its absence.
    
    Note:
        This implementation assumes the dataset and the shapefile are properly aligned spatially. The function currently
        uses only the first geometry in the GeoDataFrame for demonstration. Adjust the function to use all geometries as needed.
    """
    
    # Assuming x_predicted and gdf are already defined as shown previously
    
    # Manual approach to create the affine transform
    # Define the spatial resolution (dx, dy) if not already defined
    try:
        dx = abs(ds.ELongtitude[1].values - ds.ELongtitude[0].values)
        dy = abs(ds.NLatitude[1].values - ds.NLatitude[0].values)
        # Create the affine transform for rasterio
        transform = Affine.translation(min(ds.ELongtitude.values) - dx / 2, min(ds.NLatitude.values) - dy / 2) * Affine.scale(dx, dy)
        # Create polygon mask for the first geometry in the GeoDataFrame for demonstration
        # Adjust to use all geometries as needed
        mask = rasterio.features.geometry_mask([shp.geometry.iloc[0].__geo_interface__], 
                                               out_shape=(len(ds.NLatitude), len(ds.ELongtitude)), 
                                               transform=transform, 
                                               invert=True)
        
        # Convert the mask to an xarray DataArray
        mask_da = xr.DataArray(mask, dims=("NLatitude", "ELongtitude"), 
                            coords={"NLatitude": ds.NLatitude, "ELongtitude": ds.ELongtitude})
    
    except:
        dx = abs(ds.longitude[1].values - ds.longitude[0].values)
        dy = abs(ds.latitude[1].values - ds.latitude[0].values)
        # Create the affine transform for rasterio
        transform = Affine.translation(min(ds.longitude.values) - dx / 2, min(ds.latitude.values) - dy / 2) * Affine.scale(dx, dy)
        # Create polygon mask for the first geometry in the GeoDataFrame for demonstration
        # Adjust to use all geometries as needed
        mask = rasterio.features.geometry_mask([shp.geometry.iloc[0].__geo_interface__], 
                                               out_shape=(len(ds.latitude), len(ds.longitude)), 
                                               transform=transform, 
                                               invert=True)
        
        # Convert the mask to an xarray DataArray
        mask_da = xr.DataArray(mask, dims=("latitude", "longitude"), 
                            coords={"latitude": ds.latitude, "longitude": ds.longitude})
    

    return mask_da




def create_xarray_within_boundary(datetime_list: List[pd.Timestamp], boundary: Tuple[float, float, float, float], resolution: float) -> xr.Dataset:
    """
    Creates an xarray dataset with dimensions of datetime, longitude, and latitude within a specified boundary.

    Args:
        datetime_list (List[pd.Timestamp]): List of datetime objects for the datetime dimension.
        boundary (Tuple[float, float, float, float]): The geographical boundary specified as (min_latitude, min_longitude, max_latitude, max_longitude).
        resolution (float): The resolution in degrees for the longitude and latitude dimensions.

    Returns:
        xr.Dataset: An xarray dataset with dimensions of datetime, longitude, and latitude.
    """
    # Unpack the boundary
    min_lat, min_lon, max_lat, max_lon = boundary
    
    # Generate longitude and latitude arrays within the boundary at the specified resolution
    longitudes = np.arange(min_lon, max_lon, resolution)
    latitudes = np.arange(min_lat, max_lat, resolution)
    
    # Create a 3D meshgrid for longitude, latitude, and datetime dimensions
    lon, lat = np.meshgrid(longitudes, latitudes)
    time = np.array(datetime_list)
    
    # Initialize an empty array for the data dimension
    data = np.zeros((len(time), len(latitudes), len(longitudes)))
    
    # Create the xarray Dataset
    ds = xr.Dataset(
        {
            "Negative_oxygen_ions": (["time", "latitude", "longitude"], data)
        },
        coords={
            "time": time,
            "latitude": ("latitude", latitudes),
            "longitude": ("longitude", longitudes)
        }
    )
    
    return ds.rio.write_crs("EPSG:4326", inplace=True)




import xarray as xr
import numpy as np
from tqdm import tqdm
from typing import List

def merge_features_with_progress(
    ds_a: xr.Dataset, 
    ds_b: xr.Dataset, 
    features_list: List[str] = [], 
    method: str="linear", 
    memory_optim: bool=False
) -> xr.Dataset:
    """
    Merge features from one xarray Dataset into another based on nearest matching coordinates.
    
    Args:
        ds_a (xr.Dataset): The primary xarray Dataset to which features will be added.
        ds_b (xr.Dataset): The secondary xarray Dataset from which features are extracted.
        features_list (List[str]): List of feature variable names to be extracted and added.
        
    Returns:
        xr.Dataset: The updated xarray Dataset containing merged features.
    """
    if not features_list:
        features_list=ds_b.data_vars
    for feature in tqdm(features_list, desc="Merging Features"):
        if feature in ds_b:
            # Assuming the matching coordinates in ds_a to be named 'Datetime', 'ELongtitude', and 'NLatitude'
            if memory_optim:
                temp = ds_b[feature].interp(
                    #time=ds_a.time, 
                    longitude=ds_a.longitude, 
                    latitude=ds_a.latitude, 
                    method=method
                )
                ds_a[feature] = temp.interp(
                    time=ds_a.time, 
                    #longitude=ds_a.longitude, 
                    #latitude=ds_a.latitude, 
                    method="nearest"
                )
            else:
                ds_a[feature] = ds_b[feature].interp(
                    time=ds_a.time, 
                    longitude=ds_a.longitude, 
                    latitude=ds_a.latitude, 
                    method=method
                )
    
    return ds_a



def filter_dataframe_by_date(csv_path: str, start_date: Union[str, pd.Timestamp], end_date: Union[str, pd.Timestamp]) -> pd.DataFrame:
    """
    Loads a CSV file into a DataFrame, filters it based on a specified date range, and
    retains only specified columns.

    Args:
        csv_path (str): The file path to the CSV file to be loaded.
        start_date (Union[str, pd.Timestamp]): The start date of the filter range. Can be a string or a Timestamp.
        end_date (Union[str, pd.Timestamp]): The end date of the filter range. Can be a string or a Timestamp.

    Returns:
        pd.DataFrame: A DataFrame filtered by the specified date range and reduced to specified columns.

    Example:
        >>> filtered_df = filter_dataframe_by_date("path/to/data.csv", "2021-01-01", "2021-01-31")
        >>> print(filtered_df.head())
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)  # Replace with your actual file path
    
    # Specify columns to retain
    columns_to_keep = ['Site_number','Site_name',  'time', 'Negative_oxygen_ions', 'longitude', 'latitude', 'Air_level',"PM25","O3"]
    df = df[columns_to_keep]
    
    # Convert 'Datetime' column to datetime type and filter by date range
    df['time'] = pd.to_datetime(df['time'])
    df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
    # Drop duplicates based on 'Site_number' and 'time'
    df_unique = df.drop_duplicates(subset=['Site_number', 'time'], keep='first')

    return df_unique

 
class H5Dataset(Dataset):
    def __init__(self, file_path):
        """
        Custom dataset that loads data from an HDF5 file on-the-fly.

        Args:
            file_path (str): Path to the HDF5 file.
        """
        self.file_path = file_path
        # Open the HDF5 file and access the size of the datasets
        with h5py.File(file_path, 'r') as f:
            self.data_len = len(f['data'])

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.data_len

    def __getitem__(self, idx):
        """
        Returns a single data point and its label.
        """
        # Use context manager to ensure the file is opened and closed each time,
        # which allows for efficient reading of data without loading it all into memory.
        with h5py.File(self.file_path, 'r') as f:
            # Note: The data is read into memory here, but only one item at a time.
            # Depending on your use case, you might need to apply transformations here.
            data = torch.tensor(f['data'][idx], dtype=torch.float32)
            label = torch.tensor(f['labels'][idx], dtype=torch.float32)
        return data, label


import xarray as xr
from typing import List
from tqdm.auto import tqdm
import numpy as np

def merge_features_with_broadcast(
    ds_a: xr.Dataset, 
    ds_b: xr.Dataset, 
    features_list: List[str] = [], 
    method: str = "linear", 
) -> xr.Dataset:
    """
    Merge features from one xarray Dataset into another based on nearest matching coordinates,
    copying the data for the nearest time frame from ds_b to ds_a.
    """
    if not features_list:
        features_list = list(ds_b.data_vars)
    
    # Calculate the nearest times in ds_b for each time in ds_a
    nearest_times = []
    for time_a in ds_a.time.values:
        # Calculate the absolute differences between this time and all times in ds_b
        time_diffs = abs(ds_b.time - time_a)
        # Find the index of the minimum difference
        nearest_index = time_diffs.argmin().item()
        nearest_time = ds_b.time.isel(time=nearest_index).values
        nearest_times.append(nearest_time)
    
    for feature in tqdm(features_list, desc="Merging Features"):
        if feature in ds_b:
            # Interpolate spatially for latitude and longitude first
            temp = ds_b[feature].interp(
                longitude=ds_a.longitude, 
                latitude=ds_a.latitude, 
                method=method
            )
            
            # Initialize an empty DataArray for the interpolated feature with the same shape as ds_a
            interpolated_feature = xr.full_like(ds_a[list(ds_a.data_vars)[0]], fill_value=np.nan)
            
            # Assign the interpolated values based on nearest times
            for i, time_a in enumerate(ds_a.time.values):
                nearest_time = nearest_times[i]
                # Select the data from temp using the nearest time
                interpolated_value = temp.sel(time=nearest_time, method="nearest")
                interpolated_feature.loc[dict(time=time_a)] = interpolated_value
                
            # Finally, update ds_a with the new interpolated feature data
            ds_a[feature] = interpolated_feature
            
    return ds_a
import numpy as np
import pandas as pd

def split_test_train(df, seed=42, split=0.1, flag="Site", pseudo=True):
    np.random.seed(seed)  # Set the seed for reproducibility
    
    df_copy = df.copy()
    df_copy['Set'] = 'Train'  # Initialize 'Set' column
    
    if flag == "Site":
        unique_sites = df_copy[df_copy['Site_number'] != 0]['Site_number'].unique()
        num_sites_for_test = round(len(unique_sites) * split)
        test_sites = np.random.choice(unique_sites, size=num_sites_for_test, replace=False)
        
        df_copy.loc[df_copy['Site_number'].isin(test_sites), 'Set'] = 'Test'
        
    elif flag == "Sample":
        non_pseudo_indices = df_copy[df_copy['Site_number'] != 0].index
        test_indices = np.random.choice(non_pseudo_indices, size=int(len(non_pseudo_indices) * split), replace=False)
        
        df_copy.loc[test_indices, 'Set'] = 'Test'


    df_copy.loc[df_copy['Site_number'] == 0, 'Set'] = 'Pseudo'

    # Extract indices and site numbers for train and test sets
    if pseudo:
        train_indices = df_copy[df_copy['Set'].isin(['Train', 'Pseudo'])].index.tolist()
    else:
        train_indices = df_copy[df_copy['Set'] == 'Train'].index.tolist()
    test_indices = df_copy[df_copy['Set'] == 'Test'].index.tolist()
    train_site_numbers = df_copy.loc[train_indices, 'Site_number'].unique().tolist()
    test_site_numbers = df_copy.loc[test_indices, 'Site_number'].unique().tolist()
      
    # Calculate statistics
    selected_site_count = len(test_site_numbers)
    unique_sites = len(df_copy[df_copy['Site_number'] != 0]['Site_number'].unique())
    selected_site_percentage = (selected_site_count / unique_sites) * 100
    
    selected_row_count = len(test_indices)
    total_rows = len(df_copy[df_copy['Site_number'] != 0])
    selected_row_percentage = (selected_row_count / total_rows) * 100
    
    print(f"Selected Site Count: {selected_site_count}, ({selected_site_percentage:.2f}%)")
    print(f"Selected DataRow Count: {selected_row_count}, ({selected_row_percentage:.2f}%)")

    # Training statistics excluding pseudo
    training_site_count = len(train_site_numbers)
    training_row_count = len(train_indices)
    training_site_percentage = (training_site_count / unique_sites) * 100
    training_row_percentage = (training_row_count / total_rows) * 100
    
    print(f"Training Site Count: {training_site_count}, ({training_site_percentage:.2f}%)")
    print(f"Training DataRow Count: {training_row_count}, ({training_row_percentage:.2f}%)")

 

    return train_site_numbers, test_site_numbers, train_indices, test_indices, df_copy


import pandas as pd
import numpy as np
import geopandas as gpd
from pykrige.ok import OrdinaryKriging
from shapely.geometry import Point
import random
from tqdm import tqdm

def pseudo_df_generator(df):
        
    zhejiang_bbox = gpd.read_file(r'C:\Datasets\Zhejiang20-23RS\Earth/province.shp').to_crs('EPSG:4326')
    zhejiang_bbox= zhejiang_bbox[zhejiang_bbox["NAME"] =="浙江"]
    
    
    # Step 0: Keep only numeric and date type columns in df
    numeric_cols = ['Negative_oxygen_ions']
    df_filtered = df[list(df.select_dtypes(include=[np.number]).columns)+["time"]]
    
    # Step 1: Extract unique time frames where Site_number count > 30
    time_frames = df_filtered.groupby('time').filter(lambda x: x['Site_number'].nunique() > 30)['time'].unique()
    
    def generate_random_points_within_polygon(polygon, num_points):
        points = []
        minx, miny, maxx, maxy = polygon.bounds
        while len(points) < num_points:
            pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if polygon.contains(pnt):
                points.append(pnt)
        return points
    
    # Assuming the Zhejiang bounding box is a single polygon
    polygon = zhejiang_bbox.unary_union
    
    pseudo_data_list = []  # Use a list to collect all pseudo data frames
    
    
    for time_frame in tqdm(time_frames):
        df_time_frame = df[df['time'] == time_frame]
        random_points = generate_random_points_within_polygon(polygon, 30)
        
        for point in random_points:
            pseudo_data = {
                'Site_number': 0000,
                'time': time_frame,
                'longitude': point.x,
                'latitude': point.y,
                'Site_name': "PseudoPoints",
            }
                # Kriging interpolation for each numeric column
            for col in numeric_cols:
                lons = df_time_frame['longitude'].values
                lats = df_time_frame['latitude'].values
                vals = df_time_frame[col].values
        
                OK = OrdinaryKriging(lons, lats, vals, variogram_model='linear', verbose=False, enable_plotting=False)
                z, ss = OK.execute('points', np.array([point.x]), np.array([point.y]))
                pseudo_data[col] = z[0]
            
            pseudo_data_list.append(pseudo_data)
    
    # Convert the list of dictionaries to a DataFrame
    pseudo_points_df = pd.DataFrame(pseudo_data_list)
    
    # Concatenate the original and pseudo DataFrames (ensure df_filtered is defined appropriately)
    df_filtered = df[['Site_number','Site_name', 'time', 'longitude', 'latitude'] + numeric_cols]  # Adjust as needed
    final_df = pd.concat([df_filtered, pseudo_points_df], ignore_index=True)
    return final_df