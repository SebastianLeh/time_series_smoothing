import glob
import numpy as np
import rasterio
from sklearn.linear_model import TheilSenRegressor
import geowombat as gw
import xarray as xr

def transform_rasters_error_func(files, slope, intercept):
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

        # If it is the first iteration, set the previous raster as the current raster to prepare for second run
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

            # create the new raster which contains values from the new raster only where the change exceeds the threshold mask
            new_raster = np.where(mask, current_raster_nan, previous_raster)
            previous_raster = new_raster

            # Append a copy of the previous raster to the list
            transformed_rasters.append(new_raster.copy())
    # Return the list of transformed rasters
    return transformed_rasters, change_magnitude_rasters


def open_raster_files_as_xarray(files):
    """
    Opens a list of raster files as a single xarray.DataArray.

    Parameters:
        files (list of str): File paths to the raster files.

    Returns:
        tuple: An xarray.DataArray of the stacked raster data, a list of attributes for exporting, and the last raster opened.
    """
    data_arrays = []
    export_attrs_lst = []
    with gw.open(files[0]) as first_file:
        ref_bounds = first_file.gw.bounds
    for file in files:
        with gw.config.update(ref_bounds=ref_bounds):
            with gw.open(file) as raster:
                export_attrs_lst.append(raster.attrs)
                data_arrays.append(raster)
    stacked_da = xr.concat(data_arrays, dim='time')
    stacked_da = stacked_da.assign_coords(time=np.arange(len(files)))
    return stacked_da, export_attrs_lst, raster


def calculate_ols_trend_xarray(da):
    """
    Calculates the linear trend (slope) of each pixel using OLS regression within an xarray.DataArray.

    Parameters:
        da (xarray.DataArray): Data array containing the time series data for each pixel.

    Returns:
        xarray.DataArray: An array containing the slope of the linear trend for each pixel.
    """
    polyfit_results = da.polyfit(dim='time', deg=1)
    slope = polyfit_results.polyfit_coefficients.sel(degree=1)
    return slope


def write_xarray_to_raster(da, output_file, attrs_lst):
    """
    Writes an xarray.DataArray to a raster file.

    Parameters:
        da (xarray.DataArray): The data array to write.
        output_file (str): Path to the output raster file.
        attrs_lst (list): List of attributes to assign to the output raster.
    """
    chunksize = 1024
    export_raster = da.assign_attrs(attrs_lst[0]).chunk(chunks=chunksize)
    export_raster.gw.save(output_file, num_workers=16)


def print_raster_statistics(rasters, labels):
    """
    Prints the statistics of each raster in a list of rasters.

    Parameters:
        rasters (list of np.ndarray): List of raster arrays.
        labels (list of str): Labels corresponding to each raster.
    """
    for i, raster in enumerate(rasters):
        label = labels[i]
        mean = np.nanmean(raster)
        std = np.nanstd(raster)
        minim = np.nanmin(raster)
        maxim = np.nanmax(raster)
        print(f"{label} - Shape: {raster.shape}, Mean: {mean}, Std: {std}, Min: {minim}, Max: {maxim}\n")

def calculate_theil_trend_raster(files):
    """
    Calculates the trend of each pixel across a series of raster files using Theil-Sen regression.

    Parameters:
        files (list of str): Paths to the raster files to be processed.

    Returns:
        np.ndarray: A numpy array representing the trend raster, where each pixel's value indicates the trend's slope over time.
    """
    raster_stack = []
    for file in files:
        with rasterio.open(file) as src:
            raster_stack.append(src.read(1))
    raster_stack = np.array(raster_stack)
    nodata_value = src.nodata
    raster_stack[raster_stack == nodata_value] = np.nan
    trend_raster = np.full(raster_stack[0].shape, np.nan)
    time_points = np.arange(raster_stack.shape[0])
    for i in range(raster_stack.shape[1]):
        for j in range(raster_stack.shape[2]):
            pixel_series = raster_stack[:, i, j]
            if not np.all(np.isnan(pixel_series)):
                model = TheilSenRegressor().fit(time_points.reshape(-1, 1), pixel_series)
                trend_raster[i, j] = model.coef_[0]
    return trend_raster
