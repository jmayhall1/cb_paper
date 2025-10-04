# coding=utf-8
"""
@author: John Mark Mayhall
Optimized: 10/02/2025

This module defines the PointMethod class, which generates a list of point objects,
filters geojson files based on completed processing, and loads latitude/longitude arrays.
"""

import glob
import os

import numpy as np


class PointMethod:
    """
    Handles creation of file lists and loading of lat/lon arrays
    for processing TC-centered geojsons.
    """

    def __init__(self, path: str, folder: str, path_arr: str):
        """
        Initialize with required paths.

        Parameters
        ----------
        path : str
            Base path where geojson folders are located.
        folder : str
            Subfolder containing geojson files.
        path_arr : str
            Path to lat/lon array storage.
        """
        self.path = path
        self.folder = folder
        self.path_arr = path_arr

    def pointcreate(self):
        """
        Creates the necessary lists and loads lat/lon arrays.

        Returns
        -------
        geo_lst : list
            Filtered list of geojson files still needing processing.
        nfiles : int
            Number of geojson files remaining.
        lon_arr : np.ndarray
            Longitude array loaded from .npz file.
        lat_arr : np.ndarray
            Latitude array loaded from .npz file.
        """

        # Get the full list of geojsons in the specified folder
        geo_lst = os.listdir(os.path.join(self.path, self.folder))

        # Define processed file directories
        processed_dirs = {
            "08": "/rstor/jmayhall/Model_Training_Code/geojson_transform/tc_centered/08/",
            "13": "/rstor/jmayhall/Model_Training_Code/geojson_transform/tc_centered/13/"
        }

        def get_processed_list(channel: str):
            """Helper to get processed geojsons for a given channel."""
            done_files = glob.glob(os.path.join(processed_dirs[channel], "*_scaledrot0.npz"))
            # Strip path and replace suffix with .geojson
            return [
                os.path.basename(f).replace("_scaledrot0.npz", ".geojson")
                for f in done_files
            ]

        # Get lists of completed files for both channels
        done_list_8 = get_processed_list("08")
        done_list_13 = get_processed_list("13")

        # Exclude files already processed in both channels
        geo_lst = list(set(geo_lst) - (set(done_list_13) & set(done_list_8)))

        # Load first lat/lon array (all should share grid structure)
        arr_files = os.listdir(self.path_arr)
        if not arr_files:
            raise FileNotFoundError(f"No array files found in {self.path_arr}")

        arr_data = np.load(os.path.join(self.path_arr, arr_files[0]))
        lon_arr, lat_arr = arr_data["lon"], arr_data["lat"]

        # Number of files left
        nfiles = len(geo_lst)

        return geo_lst, nfiles, lon_arr, lat_arr
