import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import sys 
from scipy.stats import gaussian_kde

sys.path.append("C:/Users/isxzl/OneDrive/Code/FeatureInsight")



def regression_evaluation_report(y_true, y_pred, plot=True):
    """
    Generates a report for evaluating regression models, including a plot with enhanced visualization.
    """
    # Ensure y_true and y_pred are numpy arrays to avoid indexing issues
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    # Calculate metrics
    print("[AutoGEO] [Info] Start Analysis the Prediction Accuracy")
    mae = mean_absolute_error(y_true_np, y_pred_np)
    mse = mean_squared_error(y_true_np, y_pred_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_np, y_pred_np)
    n_samples = len(y_true_np)
    
    # Linear regression fit for Y=AX+B
    A, B = np.polyfit(y_true_np, y_pred_np, 1)

    # Metrics report
    metrics_report = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'RMSE', 'R^2', 'Samples'],
        'Value': [mae, mse, rmse, r2, n_samples]
    })

    # Print the metrics
    print(metrics_report.to_string(index=False))

    # Plotting
    if plot:
        plt.figure(figsize=(10, 6))

        # Calculate density
        xy = np.vstack([y_true_np, y_pred_np])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = y_true_np[idx], y_pred_np[idx], z[idx]

        scatter = plt.scatter(x, y, c=z, cmap='coolwarm')
        plt.colorbar(scatter, label='Density')

        # Fit line
        plt.plot(np.unique(y_true_np), np.poly1d([A, B])(np.unique(y_true_np)), 'r-', lw=2)

        # Identity line
        plt.plot([y_true_np.min(), y_true_np.max()], [y_true_np.min(), y_true_np.max()], 'k--', lw=2)

        # Annotations
        plt.text(0.05, 0.95, f'Y={A:.2f}X + {B:.2f}\nR²={r2:.2f}\nN={n_samples}\nMAE={mae:.2f}\nRMSE={rmse:.2f}',
                 transform=plt.gca().transAxes, verticalalignment='top')

        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted Values')

        plt.show()

    return metrics_report




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
                'latitude': point.y
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
    df_filtered = df[['Site_number', 'time', 'longitude', 'latitude'] + numeric_cols]  # Adjust as needed
    final_df = pd.concat([df_filtered, pseudo_points_df], ignore_index=True)
    return final_df