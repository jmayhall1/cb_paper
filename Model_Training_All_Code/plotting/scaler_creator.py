# coding=utf-8
"""
Author: John Mark Mayhall, Andrew White
Optimized: 10/02/2025

Purpose:
--------
This script creates a StandardScaler from a set of brightness temperature (BT) npz/netCDF files
for different channels. The scaler can then be used to scale future arrays consistently.

Steps:
1. Load all netCDF files for a given channel.
2. Convert radiances to brightness temperatures in Celsius.
3. Fit a StandardScaler incrementally (partial_fit) to handle large datasets.
4. Save the fitted scaler to disk for later use.
"""

import os

import joblib
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # ---------------- CONFIGURATION ---------------- #
    # Channels and their respective data paths
    channels = {
        'ch13': '/rstor/jmayhall/Model_Training_Full/ncfiles/sport_data_images_training/',
        'ch8': '/rstor/jmayhall/Model_Training_Full/ch8/ncfiles/sport_data_images_training/'
    }

    # Output scaler filenames
    scaler_files = {
        'ch13': 'data_scaler_ch13.save',
        'ch8': 'data_scaler_ch8.save'
    }

    # ---------------- MAIN EXECUTION ---------------- #

    for ch, path in channels.items():
        print(f"Processing channel: {ch}")

        # Create StandardScaler instance
        scaler = StandardScaler()

        # List all netCDF files in the folder
        lst_images = sorted(os.listdir(path))

        for i, file in enumerate(lst_images, start=1):
            print(f'  Processing file {i} of {len(lst_images)}: {file}')

            # Load ABI data using xarray
            abidata = xr.open_dataset(os.path.join(path, file), engine='h5netcdf').CMI.values

            # Convert radiance to brightness temperature in Celsius
            bt_data = np.array(abidata - 273.15, dtype=np.float16)

            # Incrementally fit the scaler
            scaler.partial_fit(bt_data)

            # Free memory for large datasets
            del abidata, bt_data

        # Save the fitted scaler
        joblib.dump(scaler, scaler_files[ch])
        print(f"  Saved scaler to {scaler_files[ch]}\n")

        # Free memory before processing the next channel
        del scaler, lst_images
