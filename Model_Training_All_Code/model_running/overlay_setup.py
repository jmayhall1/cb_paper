# coding=utf-8
"""
@author: John Mark Mayhall
Optimized: 10/02/2025

Purpose:
Sets up and runs the pre-trained CNN model on test images, generating probability maps contoured over
Channel 13 brightness temperatures. Supports plotting for multiple images and model epochs.
"""

import gc
import glob
import os

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from tensorflow.python import keras as k  # for custom loss casting


# ------------------------------------------
# Data loading function
# ------------------------------------------
def load(test_path_bright, arr, ch8, latlon_path):
    """
    Loads Channel 13, Channel 8, unscaled, and lat/lon arrays for a single test file.

    Returns:
        load_arr: scaled Channel 13 array with batch dimension
        x_test8: scaled Channel 8 array with batch dimension
        x_unscaled: unscaled Channel 13 array
        lat, lon: latitude and longitude arrays
    """
    load_arr = np.load(os.path.join(test_path_bright, arr))["brightness"]
    x_test8 = np.load(os.path.join(ch8, arr))["brightness"]

    unscale_arr = arr.replace('scaledrot0', 'cutunscaled')
    x_unscaled = np.load(f'/rstor/jmayhall/Model_Training_Full/cut_unscaled_arrays/test/{unscale_arr}')['brightness']

    latlon_arr = arr.replace('scaledrot0', 'latlon')
    lat = np.nan_to_num(np.load(os.path.join(latlon_path, latlon_arr))["lat"])
    lon = np.nan_to_num(np.load(os.path.join(latlon_path, latlon_arr))["lon"])

    return load_arr[None, :], x_test8[None, :], x_unscaled, lat, lon


# ------------------------------------------
# Overlay class
# ------------------------------------------
class Overlay:
    """Generates probability images contoured over brightness temperature images using a pre-trained CNN."""

    def __init__(self, path, test_path_bright, month_dict, latlon_path, ch8):
        self.path = path
        self.test_path_bright = test_path_bright
        self.month_dict = month_dict
        self.latlon_path = latlon_path
        self.ch8 = ch8

    # --------------------------------------
    # Main execution function
    # --------------------------------------
    def main(self):
        """Runs the model on test images and optionally across multiple epochs."""

        # List all test files
        lst_xtest = sorted(os.listdir(self.test_path_bright))

        # Suppress TensorFlow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # Load the primary model
        model = load_model(os.path.join(self.path, 'model.keras'), compile=False)

        # Pre-define colormaps for plotting
        colormap = cm.Greys
        normalize = mcolors.Normalize(vmin=-80, vmax=30)
        s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
        cmappable = ScalarMappable(Normalize(0, 1), cmap='rainbow')

        # -------------------------------
        # Loop through each test image
        # -------------------------------
        for arr in lst_xtest:
            # Determine day string from filename
            day = arr[13:14] if int(arr[12:14]) < 10 else arr[12:14]

            # Load arrays
            load_arr, x_test8, x_unscaled, lat, lon = load(
                self.test_path_bright, arr, self.ch8, self.latlon_path
            )

            # Predict probabilities
            predict = model.predict([load_arr, x_test8])[0, :, :, 0]

            # Plot original and probability images side by side
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 14), constrained_layout=True)

            for axis in ax:
                axis.imshow(
                    x_unscaled,
                    vmin=-80, vmax=30,
                    cmap='Greys',
                    extent=[np.min(lon), np.max(lon), np.min(lat), np.max(lat)]
                )

            ax[1].contour(
                predict,
                vmin=0, vmax=1,
                cmap='rainbow',
                extent=[np.min(lon), np.max(lon), np.max(lat), np.min(lat)]
            )

            # Titles
            ax[0].set_title('Channel 13 Brightness Temperatures (\N{degree sign}C)')
            ax[1].set_title('TCB Probabilities')

            # Figure title
            fig.suptitle(
                f"Transverse Cirrus Bands Probabilities for "
                f"{self.month_dict.get(arr[10:12])} {day}, {arr[6:10]} at {arr[15:17]}:{arr[17:19]} UTC",
                y=0.8
            )

            # Colorbars
            cbar_ax = fig.add_axes((0.565, 0.15, 0.4, 0.05))
            cbar2_ax = fig.add_axes((0.065, 0.15, 0.4, 0.05))
            fig.colorbar(s_map, cax=cbar2_ax, orientation="horizontal",
                         ticks=[-80, -50, -25, 0, 30], label='Brightness Temperatures (\N{degree sign}C)')
            fig.colorbar(cmappable, cax=cbar_ax, orientation="horizontal",
                         ticks=[0, 0.25, 0.5, 0.75, 1], label='Probability of TCB')

            # Save figure
            plt.savefig(f"{arr[:36]}_prediction.png", dpi=300)
            plt.close('all')
            gc.collect()

        # -------------------------------
        # Optional: Loop through all model epochs
        # -------------------------------
        epochs = sorted(glob.glob(os.path.join(self.path, 'logs', '*.keras')))
        if epochs:
            # Use a single representative file for epoch predictions
            arr = lst_xtest[14]
            day = arr[13:14] if int(arr[12:14]) < 10 else arr[12:14]
            load_arr, x_test8, x_unscaled, lat, lon = load(
                self.test_path_bright, arr, self.ch8, self.latlon_path
            )

            for i, epoch_path in enumerate(epochs):
                model = load_model(epoch_path, compile=False)
                predict = model.predict([load_arr, x_test8])[0, :, :, 0]

                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 14), constrained_layout=True)
                for axis in ax:
                    axis.imshow(
                        x_unscaled,
                        vmin=-80, vmax=30,
                        cmap='Greys',
                        extent=[np.min(lon), np.max(lon), np.min(lat), np.max(lat)]
                    )
                ax[1].contour(
                    predict,
                    vmin=0, vmax=1,
                    cmap='rainbow',
                    extent=[np.min(lon), np.max(lon), np.max(lat), np.min(lat)]
                )

                ax[0].set_title('Channel 13 Brightness Temperatures (\N{degree sign}C)')
                ax[1].set_title('TCB Probabilities')

                s_map = cm.ScalarMappable(norm=mcolors.Normalize(-80, 30), cmap=cm.Greys)
                cmappable = ScalarMappable(Normalize(0, 1), cmap='rainbow')

                fig.suptitle(
                    f"Transverse Cirrus Bands Probabilities for "
                    f"{self.month_dict.get(arr[10:12])} {day}, {arr[6:10]} at {arr[15:17]}:{arr[17:19]} UTC",
                    y=0.8
                )

                cbar_ax = fig.add_axes((0.565, 0.15, 0.4, 0.05))
                cbar2_ax = fig.add_axes((0.065, 0.15, 0.4, 0.05))
                fig.colorbar(s_map, cax=cbar2_ax, orientation="horizontal",
                             ticks=[-80, -50, -25, 0, 30], label='Brightness Temperatures (\N{degree sign}C)')
                fig.colorbar(cmappable, cax=cbar_ax, orientation="horizontal",
                             ticks=[0, 0.25, 0.5, 0.75, 1], label='Probability of TCB')

                plt.savefig(f"{arr[:36]}_prediction_epoch{i + 1}.png", dpi=300)
                plt.close('all')
                gc.collect()
