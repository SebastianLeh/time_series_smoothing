import glob
import rasterio
from core_functions import (transform_rasters_error_func, open_raster_files_as_xarray,
                              calculate_ols_trend_xarray, write_xarray_to_raster,
                              calculate_theil_trend_raster, print_raster_statistics)



# Path configurations
input_path = r'T:\2022_UrbanGreenEye\results_thermale_entlastung\veg_indicators_leipzig\TCD\transformer\3years' # folder containing yearly raster files
result_path = r'T:\2022_UrbanGreenEye\results_thermale_entlastung\veg_indicators_leipzig\TCD\transformer\3years\test_time_series_smoothing' # folder to save results in

# Parameters for the linear function used as threshold
slope = 0.15
intercept = 0.1

def main():
    # Load the files
    files = glob.glob(input_path+'\*tif')
    files.sort()  # Ensure files are sorted, necessary for correct time-series analysis

    # Process rasters
    transformed_rasters, change_magnitude_rasters = transform_rasters_error_func(files, slope, intercept)

    # Write the smoothed rasters time series as a new tif file for each year
    for i in range(len(transformed_rasters)):
        # get profile
        profile = rasterio.open(files[i]).profile
        # Create the output file name
        output_file = f"{result_path}/transformed_raster_{i+1}_slope{slope}_interc{intercept}.tif"
        # Write the raster as a new tif file
        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(transformed_rasters[i], 1)  # Write the first band

    # Write the change magnitude raster as a new tif file for each year
    for i in range(len(change_magnitude_rasters)):
        # get profile
        profile = rasterio.open(files[i + 1]).profile
        # Create the output file name
        output_file = f"{result_path}/change_magnitude_raster_{i+1}_slope{slope}_interc{intercept}.tif"
        # Write the raster as a new tif file
        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(change_magnitude_rasters[i], 1)  # Write the first band

    # Prepare labels for printing statistics
    labels = [f"Raster_{i+1}" for i in range(len(transformed_rasters))]

    # Print raster statistics
    print_raster_statistics(transformed_rasters, labels)

    # Open rasters as xarray, calculate trend, and write to raster
    xarray_stack, attrs, _ = open_raster_files_as_xarray(files)
    xarray_trend = calculate_ols_trend_xarray(xarray_stack)
    output_file = f"{result_path}/OLS_trend_raster.tif"
    print('writing trend raster...')
    write_xarray_to_raster(xarray_trend, output_file, attrs)

if __name__ == '__main__':
    main()
