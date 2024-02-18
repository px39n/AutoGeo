
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def get_world(ax: plt.Axes, edgecolor: str, color: str = 'none') -> plt.Axes:
    """
    Plots the world map excluding China with specified colors and edgecolors.

    Args:
        ax (plt.Axes): The matplotlib axes object where the map will be plotted.
        edgecolor (str): The color of the edges of the countries.
        color (str, optional): The fill color of the countries. Defaults to 'none'.

    Returns:
        plt.Axes: The matplotlib axes object with the world map plotted.
    """
    world = gpd.read_file(r'C:\Datasets\Zhejiang20-23RS\Earth/country.shp').to_crs('EPSG:4326')
    world[world["NAME"] != "China"].plot(ax=ax, color=color, edgecolor=edgecolor)
    ax.axis('off')
    return ax
    
def get_china(ax: plt.Axes, color: str = 'none') -> plt.Axes:
    """
    Plots the map of China including provinces and the South China Sea parts.

    Args:
        ax (plt.Axes): The matplotlib axes object where the map will be plotted.
        color (str, optional): The fill color for the provinces. Defaults to 'none'.

    Returns:
        plt.Axes: The matplotlib axes object with the China map plotted.
    """
    # Load China provinces and South China Sea parts
    country1 = gpd.read_file(r'C:\Datasets\Zhejiang20-23RS\Earth/province.shp').to_crs('EPSG:4326')
    country2 = gpd.read_file(r'C:\Datasets\Zhejiang20-23RS\Earth\bou1.shp').to_crs('EPSG:4326')
    country3 = gpd.read_file(r'C:\Datasets\Zhejiang20-23RS\Earth\bou2.shp').to_crs('EPSG:4326')
    country4 = gpd.read_file(r'C:\Datasets\Zhejiang20-23RS\Earth\bou3.shp').to_crs('EPSG:4326')

    for country in [country1, country2, country3, country4]:
        country.plot(ax=ax, edgecolor='black', facecolor=color)
        
    ax.set_xlim([71, 137])
    ax.set_ylim([16, 57])

    # Creating the South China Sea inset map
    ax_child = inset_axes(ax, width="25%", height="25%", loc='lower right',
                          bbox_to_anchor=(0.06, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    for country in [country1, country2, country3]:
        country.plot(ax=ax_child, edgecolor='black', facecolor=color)
    ax_child.set_xlim([106.5, 123])
    ax_child.set_ylim([2.8, 24.5])
    ax_child.set_xticks([])
    ax_child.set_yticks([])
    ax.axis('off')
    return ax
    
def get_zhejiang(ax: plt.Axes, edgecolor: str, color: str = 'none') -> plt.Axes:
    """
    Plots the map of Zhejiang province with specified colors and edgecolors.

    Args:
        ax (plt.Axes): The matplotlib axes object where the map will be plotted.
        edgecolor (str): The color of the edges of Zhejiang province.
        color (str, optional): The fill color of Zhejiang province. Defaults to 'none'.

    Returns:
        plt.Axes: The matplotlib axes object with the Zhejiang map plotted.
    """
    zhejiang = gpd.read_file(r'C:\Datasets\Zhejiang20-23RS\Earth\Xian.shp').to_crs('EPSG:4326')
    zhejiang.plot(ax=ax, color=color, edgecolor=edgecolor)
    ax.axis('off')
    return ax 

import matplotlib.pyplot as plt
import numpy as np

def plot_null_percentage(datasets, dim):
    """
    Plots the percentage of null values in the given datasets along a specified dimension.
    Converts 1D data to 2D for consistent plotting. Plots color bar only once after all plots.

    Parameters:
    - datasets: List of xarray DataArray objects.
    - dim: String or list of strings. Dimensions along which to aggregate the data.
    """
    figs = []  # To store figures for color bar addition at the end

    for ds in datasets:
        # Calculate the null percentage
        null_percentage = ds.isnull().mean(dim=dim).values * 100  # Convert to numpy array

        # Ensure the data is 2D for consistent plotting
        if null_percentage.ndim == 1:
            null_percentage = null_percentage[np.newaxis, :]  # Make it 2D without using reshape
            fig, ax = plt.subplots(figsize=(6, 0.3))  # Adjust the size as needed
        else:
            fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size as needed

        cax = ax.imshow(null_percentage, aspect='auto', cmap='viridis')
        ax.set_title(f'Null Percentage for {ds.name}')
        ax.set_xticks([])
        ax.set_yticks([])  # Hide axis intervals
        figs.append((fig, cax))

    # Plot color bar only once, using the last figure and color axis
    fig.colorbar(cax, ax=ax, label='Null Percentage')
    plt.show()

