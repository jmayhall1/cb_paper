# coding=utf-8
"""
@author: John Mark Mayhall
@author: Andrew White
Optimized: 10/02/2025

Purpose:
--------
This class scales brightness temperature arrays using a pre-fitted StandardScaler.
It saves both unscaled and scaled arrays, and replaces NaNs with a value outside the training range.
"""

import joblib
import numpy as np


class Scale:
    """Class for scaling arrays and saving scaled/unscaled versions."""

    def __init__(self, images, geojsons, xf, xe, yf, ye, unscaled, lst_images, trim, channel):
        """
        Initialize Scale class with needed variables.
        Parameters:
            images (str): Path to images
            geojsons (str): Path to geojsons
            xf, xe, yf, ye (int): Subsetting indices
            unscaled (list): List of unscaled numpy arrays
            lst_images (list): Corresponding file names
            trim (bool): Whether to trim images
            channel (str): Channel identifier
        """
        self.images = images
        self.geojsons = geojsons
        self.xf = xf
        self.xe = xe
        self.yf = yf
        self.ye = ye
        self.unscaled = unscaled
        self.names = lst_images
        self.trim = trim
        self.channel = channel

    def scaler_func(self):
        """
        Scales arrays according to a pre-fitted StandardScaler.
        Saves both unscaled and scaled versions of each array.
        NaN values in scaled arrays are replaced with 10 (outside training range).
        """
        print('Scaling arrays...')

        # Load pre-fitted scaler
        scaler = joblib.load('data_scaler.save')

        for i, arr in enumerate(self.unscaled):
            print(f'Processing File {i + 1} of {len(self.unscaled)}')

            # Determine file slicing length for saving
            str_slice = 34 if '.nc' in self.names[i][:43] else 43

            # Save unscaled array
            np.savez(self.names[i][:str_slice] + '_unscaled', brightness=arr)

            # Flatten the array to 2D (needed for StandardScaler)
            arr_flat = arr.reshape(-1, arr.shape[-1])

            # Scale the array
            scaled_flat = scaler.transform(arr_flat)

            # Reshape back to original shape
            scaled_arr = scaled_flat.reshape(arr.shape)

            # Replace NaNs with 10 (outside training set range)
            scaled_arr = np.nan_to_num(scaled_arr, nan=10)

            # Save scaled array
            np.savez(self.names[i][:str_slice] + '_scaled', brightness=scaled_arr)

            # Free memory
            del arr_flat, scaled_flat, scaled_arr

        print("Scaling complete.")
