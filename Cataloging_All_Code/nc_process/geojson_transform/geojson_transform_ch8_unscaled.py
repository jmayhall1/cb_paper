# coding=utf-8
"""
Optimized: 10/01/2025
Author: John Mark Mayhall

Purpose:
--------
This script transforms .geojson-defined polygons into arrays by checking whether
lat/lon grid points fall within the polygons. It prepares data arrays, filters
for files that still need processing, and runs the `Transform` class on them.

Notes:
------
- Paths and input/output directories may need to be changed depending on environment.
- The Transform class is imported from `geo_transform_setup`.
"""

import glob

import numpy as np
import pandas as pd
from geo_transform_setup import Transform


# === Load Lat/Lon Grids ===
def load_latlon(prefix: str):
    """Helper function to load lat/lon arrays from npz file."""
    arr = np.load(f"{base_path}cataloging/nc_process/geojson_transform/latlon_arrays/{prefix}.npz")
    return np.array(arr["lat"], dtype=np.float32), np.array(arr["lon"], dtype=np.float32)


if __name__ == '__main__':
    # === Path Setup ===
    online = True
    path_dict = {
        True: "/rstor/jmayhall/",
        False: "//uahdata/rstor/"
    }
    base_path = path_dict[online]

    lat_arr_al, lon_arr_al = load_latlon("AL_latlon")
    lat_arr_ep17, lon_arr_ep17 = load_latlon("EP_latlon_17")
    lat_arr_ep18, lon_arr_ep18 = load_latlon("EP_latlon_18")

    # === Load and Process HURDAT Interpolated Track Data ===
    df_AL = pd.read_csv(f"{base_path}cataloging/hurdat_update_interp_AL.txt", sep="\t")
    df_EP = pd.read_csv(f"{base_path}cataloging/hurdat_update_interp_EP.txt", sep="\t")

    df = pd.concat([df_AL, df_EP], ignore_index=True)[["ID", "Date", "Time", "Lat", "Lon"]]
    df["Date"] = pd.to_numeric(df["Date"], downcast="integer")
    df["Time"] = pd.to_numeric(df["Time"], downcast="integer")

    # === Define Channels to Process ===
    # Each entry contains:
    # [0] → channel subfolder name
    # [1] → string with processing instructions (passed into Transform)
    channel_arr = np.array([
        np.array([
            "C08_unscaled/",
            "uncut_all = bright_arr[slicey, slicex]; "
            "global all_arr; "
            "all_arr = np.full((1024, 1024), 10, dtype=np.float32); "
            "all_arr[slice_final_y, slice_final_x] = uncut_all; "
            "save_path = (f'{path_dict[online]}cataloging/nc_process/"
            "geojson_transform/completed_arrays/' + channel[0] + "
            "file[path_len:].replace('.npz', '')); "
            "np.savez_compressed(f'{save_path}_cut', brightness=all_arr); "
        ])
    ])

    # === Latitude/Longitude Sign Convention ===
    sign = {"W": -1, "S": -1}  # Western and Southern Hemisphere → negative values

    # === Main Processing Loop ===
    for channel in channel_arr:
        # Define path to uncut and already-processed arrays
        plot_path = f"{base_path}cataloging/nc_process/plotting/{channel[0]}"
        path_len = len(plot_path)

        uncut_arr = glob.glob(f"{plot_path}/*.npz")
        done_arr = glob.glob(f"{base_path}cataloging/nc_process/geojson_transform/completed_arrays/{channel[0]}/*.npz")

        # Convert completed arrays' paths to match uncut_arr format
        done_arr = [f.replace("geojson_transform/completed_arrays", "plotting").replace("_cut", "") for f in done_arr]

        # Filter to only unprocessed arrays
        uncut_arr = np.array(list(set(uncut_arr) - set(done_arr)))
        uncut_number = len(uncut_arr)

        if uncut_number == 0:
            print(f"Channel {channel[0]}: No unprocessed files found.")
            continue

        # Run the transformation interface
        interface = Transform(
            uncut_arr, uncut_number, channel, path_len, df,
            lon_arr_al, lon_arr_ep17, lon_arr_ep18,
            lat_arr_al, lat_arr_ep17, lat_arr_ep18,
            sign, online, path_dict
        )
        interface.main()
        print(f"Channel {channel[0]}: Completed processing {uncut_number} arrays.")
