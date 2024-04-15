# time_series_smoothing

This repository contains a set of function to be used for smoothing raster time series data according to a dynamic threshold and calculating trends. We use it so smooth out noise in satellite based model predictions. 

## Description

The core functionality of this toolkit is encapsulated in the `transform_rasters_error_func()` function, which applies a threshold-based change detection algorithm to a series of geospatial raster files. 
This function calculates the difference between consecutive raster images, identifies changes that exceed a user-defined threshold, and adjusts the values based on the threshold criteria. The primary outputs are two sets of raster files: one representing the adjusted values where changes were detected, and another highlighting the magnitude of these changes.
Additionally, the toolkit includes functionality for analyzing trends in the raster data using both ordinary least squares and Theil-Sen regression methods, enabling a robust assessment of temporal changes across the dataset. 

## Getting Started

### Dependencies

* Requires Python 3.6 or later.
* Libraries: `numpy`, `rasterio`, `scikit-learn`, `geowombat`, `xarray`, and their dependencies.
* Operating System: Compatible with Windows, Mac OS, and Linux.

### Installing

* Clone this repository to your local machine using: TBA
* No modifications are needed to the files/folders as long as all dependencies are installed.

### Executing program

* Ensure that the input raster files are stored in a known directory and the paths in the scripts are set correctly.
* Execute the main pipeline by running the following command from the root directory of the project: python execute_pipeline.py

## Help

Should you encounter any issues with running the scripts, first ensure that all dependencies are correctly installed and that the input data is formatted correctly.

## Authors

* [Sebastian Lehmler](https://github.com/SebastianLeh)

## Known Issues

* The program assumes all raster input files are of the same dimension and georeferencing.
* Memory usage may be high with very large raster datasets.

## License

TBA

## References

* The raster processing techniques implemented in this toolkit were inspired by various academic sources on remote sensing and environmental monitoring.
* [Geowombat library](https://geowombat.readthedocs.io/en/latest/) 



