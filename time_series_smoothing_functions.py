# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:51:11 2024

@author: QuadroRTX
"""

############Approach with variable threshold function 
# Import the libraries
import glob
import numpy as np
import rasterio
from sklearn.linear_model import TheilSenRegressor
import geowombat as gw 
import xarray as xr


# Define the function
def transform_rasters_error_func_new(files, slope, intercept):
    """
    Transforms a series of raster files by applying a threshold-based change detection algorithm.
    The function generates two sets of outputs: adjusted rasters, where detected changes are adjusted
    according to a provided slope and intercept, and change magnitude rasters, highlighting the magnitude
    of change at each pixel.
    
    Parameters:
    - files (list of str): Paths to the raster files to be processed.
    - slope (float): The slope coefficient used to calculate the change detection threshold.
    - intercept (float): The intercept used along with the slope to calculate the change detection threshold.
    
    Returns:
    - tuple of two lists:
        - The first list contains the adjusted rasters as numpy arrays.
        - The second list contains rasters of change magnitudes, where each pixel's value represents the 
          magnitude of change detected at that location.
    """
    
    # Initialize an empty list to store the transformed rasters and the rasters containing change magnitude
    transformed_rasters = []
    change_magnitude_rasters = []
    # Create a flag variable to indicate whether it is the first iteration or not
    first_iteration = True
    # Loop through the files
    for file in files:
        
        # Open the current file as a numpy array
        with rasterio.open(file) as src:
            current_raster = src.read(1)
            nodata_value = src.nodata
            current_raster_nan = np.where(current_raster == nodata_value, np.nan, current_raster)
            
        # If it is the first iteration, set the previous raster as the current raster to prepare for first run 
        if first_iteration:
            transformed_rasters.append(current_raster_nan.copy())
            previous_raster = current_raster_nan
            # Set the flag to False
            first_iteration = False
            
        # Else, update the previous raster with the current raster values where the mask is True
        else:
            # Get the threshold array from the threshold function
            threshold = slope * previous_raster + intercept

            # Create a mask where the difference between the current and the previous raster exceeds the threshold
            mask = np.abs(current_raster_nan - previous_raster) > threshold
            
            # Calculate the change magnitude only for pixels exceeding the threshold, others are set to 0
            change_magnitude = np.where(mask, current_raster_nan - previous_raster, 0)
            # Append the change magnitude raster to its list
            change_magnitude_rasters.append(change_magnitude)
            
            #create the new raster which contains values from the new raster only where the change exceeds the threshold mask 
            new_raster = np.where(mask, current_raster_nan, previous_raster)  
            previous_raster = new_raster
            
            # Append a copy of the previous raster to the list
            transformed_rasters.append(new_raster.copy())
    # Return the list of transformed rasters
    return transformed_rasters, change_magnitude_rasters


def open_raster_files_as_xarray(files):
    """
    Opens a list of raster files and stacks them into a single xarray.DataArray using rioxarray.

    Parameters:
    - files (list of str): A list of file paths to the raster files.

    Returns:
    - xarray.DataArray: A data array containing the stacked raster data, with dimensions ('time', 'y', 'x').
    """
    data_arrays = []
    
    for file in files:
        with gw.open(file) as raster:
            #get attributes of input raster for export later
            export_attrs_lst = []
            export_attrs_lst.append(raster.attrs)
            # get the nodata value from the mask raster
            nodata_lst = [] 
            nodata_lst.append(raster.attrs['_FillValue'])
            #add current raster to array list
            data_arrays.append(raster)

    
    # Combine into a single DataArray with a 'time' dimension
    stacked_da = xr.concat(data_arrays, dim='time')
    # Assign time coordinates (e.g., years or dates if available)
    stacked_da = stacked_da.assign_coords(time=np.arange(len(files)))
    
    return stacked_da, export_attrs_lst, nodata_lst

def calculate_ols_trend_xarray(da):
    """
    Calculates the linear trend (slope) of each pixel across a time series using OLS regression with xarray.

    Parameters:
    - da (xarray.DataArray): The data array containing the time series data for each pixel.
    - dim (str): The dimension along which to calculate the trend (typically 'time').

    Returns:
    - xarray.DataArray: An array containing the slope of the linear trend for each pixel.
    """
    # Fit the linear model
    polyfit_results = da.polyfit(dim='time', deg=1)
    print('polyfit_results: ', polyfit_results)
    
    # Extract the slope
    slope = polyfit_results.polyfit_coefficients.sel(degree=1)
    print('slope: ', slope)
    
    # Alternaticely - Extract the intercept
    #intercept = polyfit_results.polyfit_coefficients.sel(degree=0)

    return slope

def write_xarray_to_raster_rio(da, output_file, attrs_lst, nodata_lst):
    """
    Writes an xarray.DataArray to a raster file using rioxarray.

    Parameters:
    - da (xarray.DataArray): The data array to write. Assumes the array includes spatial metadata.
    - output_file (str): The path to the output raster file.
    """
    chunksize = 1024
    export_raster = da.assign_attrs(attrs_lst[0]).chunk(chunks=chunksize)
    print(export_raster)

    export_raster.gw.save(output_file,
                          num_workers = 16,
                          nodata = nodata_lst[0])




def calculate_theil_trend_raster(files):
    """
    Calculates the trend of each pixel across a series of raster files using Theil-Sen regression,
    a robust method to minimize the impact of outliers. This function is useful for identifying trends
    in noisy time-series raster data.

    Parameters:
    - files (list of str): Paths to the raster files to be processed.

    Returns:
    - np.ndarray: A numpy array representing the trend raster, where each pixel's value indicates
                  the trend's slope over time at that location.
    """
    # Initialize an empty list to store all raster arrays
    raster_stack = []
    
    # Load all rasters into a 3D numpy array (stack)
    for file in files:
        with rasterio.open(file) as src:
            raster_stack.append(src.read(1))
    raster_stack = np.array(raster_stack)
    
    # Handle no data values and convert them to NaN for calculations
    nodata_value = src.nodata
    raster_stack[raster_stack == nodata_value] = np.nan
    print('raster_stack', raster_stack)
    
    # Initialize the output trend raster with the same shape as individual rasters, filled with NaN
    trend_raster = np.full(raster_stack[0].shape, np.nan)
    print('trend_raster', trend_raster)
    
    # Get the number of time points (assuming the first dimension of raster_stack is time)
    time_points = np.arange(raster_stack.shape[0])
    
    # Loop through each pixel
    for i in range(raster_stack.shape[1]): # For each row
        for j in range(raster_stack.shape[2]): # For each column
            pixel_series = raster_stack[:, i, j]
            if not np.all(np.isnan(pixel_series)): # Skip processing if all values are NaN
                # Use Theil-Sen regression to estimate trend
                model = TheilSenRegressor().fit(time_points.reshape(-1, 1), pixel_series)
                # Store the slope (trend) in the trend raster
                trend_raster[i, j] = model.coef_[0]
    
    return trend_raster


# Glob all the tif files in the directory
files = glob.glob (r'T:\2022_UrbanGreenEye\results_thermale_entlastung\veg_indicators_leipzig\TCD\transformer\3years\*tif')
result_path = r'T:\2022_UrbanGreenEye\results_thermale_entlastung\veg_indicators_leipzig\TCD\transformer\3years\trend_test'
slope = 0.15
intercept = 0.1

# Sort the files by name (assuming they are named by year)
files.sort()

# Call the tranform raster function and get the result
transformed_rasters, change_magnitude_rasters = transform_rasters_error_func_new(files, slope, intercept)

# Loop through the result list and print descriptive statistics
for i, raster in enumerate(transformed_rasters):
    # Print the year and the shape of the raster
    print(f"Year: {2018+i}, Shape: {raster.shape}") #year currently hard coded 
    # Calculate and print the mean, standard deviation, minimum, and maximum of the raster
    mean = np.nanmean(raster)
    std = np.nanstd(raster)
    minim = np.nanmin(raster)
    maxim = np.nanmax(raster)
    print(f"Mean: {mean}, Std: {std}, Min: {minim}, Max: {maxim}")
    # Add a blank line for readability
    print()
    


#call the functions opening rasters as xarray and performing trend analysis and exporting the rasters
xarray_stack, attrs, crs = open_raster_files_as_xarray(files)
print('xarray_stack: ', xarray_stack)

xarray_trend = calculate_ols_trend_xarray(xarray_stack)
print('xarray trend: ', xarray_trend)

output_file = f"{result_path}/OLS_trend_raster.tif"
write_xarray_to_raster_rio(xarray_trend, output_file, attrs, crs)


#call the function getting the Theil-Sen regressor from the data -> not tested, first approaches really slow and cancelled 
#theil_trend = calculate_theil_trend_raster(files)

    
################ Saving files 

#write the trend raster 
# theil_profile = rasterio.open(files[0]).profile
# theil_output = f"{result_path}/theil_trend_raster.tif"
# with rasterio.open (theil_output, "w", **theil_profile) as dst:
#     dst.write(theil_trend, 1)


# Write the smoothed rasters time series as a new tif file for each year
for i in range(len(transformed_rasters)):
    #get profile 
    profile = rasterio.open(files[i]).profile
    # Create the output file name
    output_file = f"{result_path}/transformed_raster_{2018+i}_slope{slope}_interc{intercept}.tif"
    # Write the raster as a new tif file
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(transformed_rasters[i], 1) # Write the first band

# Write the change magnitude raster as a new tif file for each year
for i in range(len(change_magnitude_rasters)):
    #get profile 
    profile = rasterio.open(files[i+1]).profile
    # Create the output file name
    output_file = f"{result_path}/change_magnitude_raster_{2019+i}_slope{slope}_interc{intercept}.tif"
    # Write the raster as a new tif file
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(change_magnitude_rasters[i], 1) # Write the first band
        
