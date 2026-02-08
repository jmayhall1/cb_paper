# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall

Purpose:
    Run tropical cyclone data through a pre-trained CNN model using the Setup class.
    Outputs include:
        - Probability images (saved to disk)
        - Spreadsheet with tropical cyclone metadata
"""
from catalog_setup import Setup


if __name__ == "__main__":
    # Configuration settings for model and dataset paths
    config: dict[str, str | float] = {
        "model_path": "/rstor/jmayhall/Model_Training_Code/cnn_creation",
        "latlon_path": "/rstor/jmayhall/cataloging/nc_process/geojson_transform/completed_arrays/latlon_arrs/*.npz",
        "cutoff": 0.02,  # Minimum probability value to be plotted
        "c8_path": "/rstor/jmayhall/cataloging/nc_process/geojson_transform/completed_arrays/C08_scaled/*",
        "c13_scaled_path": "/rstor/jmayhall/cataloging/nc_process/geojson_transform/completed_arrays/C13_scaled/*",
        "c13_unscaled_path": "/rstor/jmayhall/cataloging/nc_process/geojson_transform/completed_arrays/C13_unscaled/*",
        "dataframe_path": "/rstor/jmayhall/cataloging/hurdat_update_interp.txt",
        "plot": 0  # 0 for compressed predict arrays, 1 for plotted arrays, 2 for only C13 plot, 3 for both 0 and 1
    }

    # Log start of process
    print("Initializing Setup with configuration.")
    interface = Setup(**config)
    print("Running code.")
    interface.main()
    print("Process completed successfully.")
