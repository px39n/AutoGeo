import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from typing import List, Tuple
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
import rasterio.features
from affine import Affine
import numpy as np
import xarray as xr
from typing import Union
import joblib
import os


def make_mask(ds: xr.Dataset, shp: gpd.GeoDataFrame) -> xr.DataArray:
    """
    Creates a mask for an xarray dataset based on the geometry of a given shapefile (GeoDataFrame).

    Args:
        ds (xr.Dataset): The input xarray dataset containing 'longitude' and 'latitude' coordinates.
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


def load_model(save_dir):
    model = joblib.load(os.path.join(save_dir, 'model.pkl'))
    t1 = joblib.load(os.path.join(save_dir, 'transformer_t1.pkl'))
    t2 = joblib.load(os.path.join(save_dir, 'transformer_t2.pkl'))
    t3 = joblib.load(os.path.join(save_dir, 'transformer_t3.pkl'))
    tt1 = joblib.load(os.path.join(save_dir, 'transformer_tt1.pkl'))
    tt2 = joblib.load(os.path.join(save_dir, 'transformer_tt2.pkl'))
    tt3 = joblib.load(os.path.join(save_dir, 'transformer_tt3.pkl'))

    features = joblib.load(os.path.join(save_dir, 'features.pkl'))
    print("Items have been loaded successfully.")
    return model,t1,t2,t3,tt1,tt2,tt3,features

def plot_single_ds(ax,ds,color_range=None):
    # Step 1: Convert selected data into a numpy array
    data_array = ds.values
     
    # Assuming the presence of longitude and latitude values in your dataset
    try:
        longitude = ds.longitude.values
        latitude = ds.latitude.values
    except:
        longitude = ds.longitude.values
        latitude = ds.latitude.values
    
    
    lon, lat = np.meshgrid(longitude, latitude)

    if color_range:
        # Use pcolormesh for plotting, specify vmin and vmax for the color scale range
        c = ax.pcolormesh(lon, lat, data_array, cmap=plt.cm.RdYlGn, vmin=color_range[0], vmax=color_range[1])
    else:
        c = ax.pcolormesh(lon, lat, data_array, cmap=plt.cm.RdYlGn)
    return ax





import pandas as pd
import numpy as np
import xarray as xr
from tqdm.auto import tqdm
from typing import List, Any

def predict_with_xarray(
    xarray_data: xr.Dataset,
    features: List[str],
    transformers: List[Any],
    model: Any,
    prediction_var: str = 'Negative_oxygen_ions'
) -> xr.Dataset:
    """
    Adds a prediction variable to an xarray dataset, fills it with predicted values
    for each timestamp using the specified model and features.

    Args:
        xarray_data (xr.Dataset): The input xarray dataset.
        features (List[str]): The list of variable names to be used as features.
        transformers (List[Any]): A list of transformers to apply to the features.
        model (LGBMRegressor): The prediction model.
        prediction_var (str): The name of the variable to be added for predictions.

    Returns:
        xr.Dataset: The updated xarray dataset with predictions.
    """
 
    # Assuming 'xarray_data', 'transformers', 'model', 'features', and 'prediction_var' are defined
    #xarray_data[prediction_var] = xr.full_like(ds['lai_hv'], fill_value=np.nan)
    # Loop through each timestamp and update predictions
    for timestamp in tqdm(xarray_data['time'].values, desc="Predicting"):
        # Extract data for the current timestamp
        df = xarray_data.sel(time=timestamp).to_dataframe().reset_index()
        
        # Keep only the required features
        df_features = df[features]
        
        # Apply transformations
        for transformer in transformers:
            df_features = transformer.transform(df_features)
        
        # Generate predictions
        predictions = model.predict(df_features)
        
        # Reshape predictions to match the 'latitude' and 'longitude' dimensions
        predictions_reshaped = predictions.reshape((len(xarray_data['latitude']), len(xarray_data['longitude'])))
        
        # Create a temporary DataArray for the reshaped predictions
        temp_pred_da = xr.DataArray(predictions_reshaped, dims=['latitude', 'longitude'], 
                                    coords={'latitude': xarray_data['latitude'], 'longitude': xarray_data['longitude']})
        
        # Update the xarray dataset for the current timestamp
        xarray_data.loc[{'time': timestamp}] = xarray_data.loc[{'time': timestamp}].assign({prediction_var: temp_pred_da})

    return xarray_data
