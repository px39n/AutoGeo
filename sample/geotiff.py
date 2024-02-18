from datetime import datetime, date
from typing import Optional, List
import rioxarray

def extract_date_from_string(str1: str, flag: Optional[List[str]] = None, date_format: str='%yyyy%ddd') -> Optional[date]:
    """
    Extracts a date from a string based on the provided flag and date format.

    Args:
        str1 (str): The string from which to extract the date.
        flag (Optional[List[str]]): A list containing two elements: the start and end flags to isolate the date part in the string. 
                                     If None, the entire string is considered.
        date_format (str): The format of the date in the string. Use '%yyyy%ddd' for year and day of year, 
                           '%yyyy%mm%dd' for year, month, and day.

    Returns:
        Optional[date]: The extracted date if found and correctly formatted, None otherwise.

    Example usage:
        >>> str1 = "MYD04_L2.A2022001.mosaic.061.2024007025724.pssggs_000502078783.Deep_Blue_Aerosol_Optical_Depth_550_Land-Deep_Blue_Aerosol_Optical_Depth_550_Land"
        >>> flag = ["L2.A", ".mosaic"]
        >>> date_format = "%yyyy%ddd"
        >>> extract_date_from_string(str1, flag, date_format)
    """
    # Extract the date part from the string using the flag
    if flag is None:
        date_str = str1
    else:
        start_flag, end_flag = flag
        start_index = str1.find(start_flag) + len(start_flag)
        end_index = str1.find(end_flag, start_index)
        date_str = str1[start_index:end_index]

    # Parse the date based on the given format
    try:
        if date_format == '%yyyy%ddd':
            # Adjusting the format to Python's datetime module requirements
            date_str = date_str.replace('%yyyy', '%Y').replace('%ddd', '%j')
            return datetime.strptime(date_str, '%Y%j').date()
        elif date_format == '%yyyy%mm%dd':
            # Adjusting the format to Python's datetime module requirements
            date_str = date_str.replace('%yyyy', '%Y').replace('%mm', '%m').replace('%dd', '%d')
            return datetime.strptime(date_str, '%Y%m%d').date()
    except ValueError:
        return None

    return None
import sys 
import numpy as np
import os
import rioxarray
import pandas as pd
def load_srtm(srtm_path):
    date=extract_date_from_string(os.path.basename(srtm_path).strip(".img"))
    time_coord = pd.to_datetime([date])
    
    data = rioxarray.open_rasterio(srtm_path)
    data = data.expand_dims({"time": time_coord}).squeeze('band', drop=True)
    data = data.rio.reproject("EPSG:4326")
    data = data.rename({'y': 'latitude', 'x': 'longitude'})
    dataset = data.to_dataset(name='srtm')
    return dataset.where(dataset['srtm'] != 32767, np.nan)