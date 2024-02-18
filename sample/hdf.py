import rasterio 
import rioxarray
import numpy as np
import pandas as pd
import sys 
import xarray as xr
import os 
sys.path.append("C:/Users/isxzl/OneDrive/Code/AutoGeo")
def load_modis(hdf4_file_path, band='LC_Type4'):
    
    xds = rioxarray.open_rasterio(hdf4_file_path)
    xds_wgs84 = xds.rio.reproject("EPSG:4326")
    xds_wgs84 = xds_wgs84.where(xds_wgs84 != 255, np.nan)
    range_ending_date = pd.to_datetime(xds_wgs84.attrs['RANGEENDINGDATE'])
    xds_wgs84.coords['time'] = range_ending_date
    xds_wgs84_expanded = xds_wgs84.expand_dims({"time": [range_ending_date]})
    xds_wgs84_expanded = xds_wgs84_expanded.rename({'y': 'latitude', 'x': 'longitude'})
    
    return xds_wgs84_expanded[[band]].squeeze('band', drop=True).sortby("latitude").sortby("longitude")

 

from sample.unit_test import find_files_with_extension
def spatio_merge(ds1,ds2):
        
    # Determine the minimum and maximum bounds for both datasets
    lat_min = min(ds1.latitude.min(), ds2.latitude.min()).item()
    lat_max = max(ds1.latitude.max(), ds2.latitude.max()).item()
    lon_min = min(ds1.longitude.min(), ds2.longitude.min()).item()
    lon_max = max(ds1.longitude.max(), ds2.longitude.max()).item()
    
    # Check if ds1's coordinate bounds fully contain ds2's coordinates
    contained_lat = ds1.latitude.min().item() <= lat_min and ds1.latitude.max().item() >= lat_max
    contained_lon = ds1.longitude.min().item() <= lon_min and ds1.longitude.max().item() >= lon_max
    
    # If ds2 is not fully contained within ds1, adjust ds1's coordinates
    if not (contained_lat and contained_lon):
        # Calculate the resolution of ds1
        lat_res = np.diff(ds1.latitude.values).mean()
        lon_res = np.diff(ds1.longitude.values).mean()
        
        # Generate new coordinate arrays for ds1 to cover the combined extent
        new_lat = np.arange(start=lat_min, stop=lat_max + lat_res, step=lat_res)
        new_lon = np.arange(start=lon_min, stop=lon_max + lon_res, step=lon_res)
        
        # Reindex ds1 to the new coordinates, interpolating as necessary
        ds1 = ds1.interp(latitude=new_lat, longitude=new_lon, method='nearest')
        #print(111)
    # Interpolate ds2 to the (possibly updated) coordinate system of ds1
    ds2_interp = ds2.interp(latitude=ds1.latitude, longitude=ds1.longitude, method='nearest')
    
    # Merge the datasets
    ds_merged = xr.merge([ds1,ds2_interp])
    return ds_merged



def load_modis_batch(directory):
    #def load_era5_batch(directory):
    extension = '.hdf'
    matching_files = find_files_with_extension(directory, extension)
    merge_path = os.path.join(directory,"Merged.nc")
    
    if os.path.exists(merge_path):
        print("Load merged file")
        merged_ds = xr.open_dataset(merge_path)
    else:    
        for i, file_path in enumerate(matching_files):
            print(f"[autoGEO][Info] Process {i+1}th file in {len(matching_files)}")
            if i==0:
                merged_ds=load_modis(file_path,band='LC_Type4')
            else:
                ds=load_modis(file_path,band='LC_Type4')
                merged_ds=spatio_merge(merged_ds,ds)
        merged_ds.to_netcdf(merge_path, format='NETCDF4')  
    
    return merged_ds