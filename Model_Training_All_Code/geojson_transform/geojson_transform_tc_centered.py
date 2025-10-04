# coding=utf-8
"""
@author: John Mark Mayhall
Optimized: 10/02/2025

This script transforms geojson files into arrays by checking if points from
lat/lon arrays fall within polygons defined by the geojsons.

Paths and variables may need to be updated depending on the environment.
"""

import glob
import os

import pandas as pd
from geo_transform_setup_tc_centered import Transform

if __name__ == '__main__':
    # ---------------- Configuration ---------------- #
    channel_lst = ["13"]  # List of ABI channels to process
    path_add = {"08": "ch8/", "04": "ch4/"}  # Subfolder mapping by channel
    path = "/rstor/jmayhall/Model_Training_Full/geojsons/"  # Base geojson path
    folder = "verification/"  # Subfolder with storm cases
    path_arr = "/rstor/jmayhall/Model_Training_Full/latlon_arrays/"  # Lat/lon array location
    plot, rotate = True, True  # Debug/test switches (unused here but may be needed downstream)
    # ------------------------------------------------ #

    for channel in channel_lst:
        # Paths to uncut arrays for the current channel
        uncut_path = glob.glob(
            f"/rstor/jmayhall/Model_Training_Full/{path_add.get(channel, '')}uncut_arrays/{folder}*"
        )

        # Paths to arrays already processed into scaled versions
        cut_path = glob.glob(
            "/rstor/jmayhall/Model_Training_Code/geojson_transform/13/*_scaledrot0.npz"
        )
        # Convert cut paths into equivalent uncut paths for comparison
        cut_path = [
            w.replace("_scaledrot0", "_scaled")
            .replace("Model_Training_Code/geojson_transform/13",
                     "Model_Training_Full/uncut_arrays/training")
            for w in cut_path
        ]

        # Remaining uncut arrays that still need processing
        uncut_arr = list(set(uncut_path) - set(cut_path))
        uncut_df = pd.DataFrame(uncut_arr, columns=["File"])

        # Change to correct working directory (per-channel)
        os.chdir(f"/rstor/jmayhall/Model_Training_Code/geojson_transform/{channel}/")

        # Run transformation
        interface = Transform(path, folder, path_arr, uncut_df, rotate)
        interface.main()
