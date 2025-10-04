# coding=utf-8
"""
Authors: John Mark Mayhall, Patrick Duran
Optimized: 10/02/2025

This module defines the Nearest class, which finds the nearest array index to a given
polygon center using pyproj (geodesic distance). It then extracts a 1024x1024 subset
of lon/lat/brightness arrays centered on that nearest point, and returns shapely Point objects.
"""

import glob

import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import Point


class Nearest:
    """
    Class for finding the nearest lat/lon index to a polygon center, slicing arrays,
    and generating shapely Point objects for each pixel.
    """

    def __init__(self, lon_arr, lat_arr, center, uncut_df, folder, num, p, file, str_pattern):
        """
        Parameters
        ----------
        lon_arr : np.ndarray
            Longitude grid array.
        lat_arr : np.ndarray
            Latitude grid array.
        center : tuple[float, float]
            Polygon center (lon, lat).
        uncut_df : pd.DataFrame
            A DataFrame containing file paths of uncut arrays (with 'File' column).
        folder : str
            Subfolder name for storm data.
        num : int
            File index (for tracking).
        p : int
            Polygon index (for tracking).
        file : str
            Current geojson filename.
        str_pattern : str
            Unique substring pattern used to identify correct array files.
        """
        self.lon_arr = lon_arr
        self.lat_arr = lat_arr
        self.center = center
        self.uncut_df = uncut_df
        self.folder = folder
        self.num = num
        self.p = p
        self.file = file
        self.str_pattern = str_pattern

    def distance(self):
        """
        Finds the nearest point to the polygon center and extracts 1024x1024 arrays.

        Returns
        -------
        point_obj : list[list[Point]]
            List of shapely Point objects for each grid cell.
        lon_arr : np.ndarray
            Cropped longitude array (1024x1024).
        lat_arr : np.ndarray
            Cropped latitude array (1024x1024).
        bright_arr : np.ndarray
            Cropped scaled brightness array (1024x1024).
        """

        clon, clat = self.center

        # Compute geodesic distance from center to every grid point
        geodesic = Geod(ellps="WGS84")
        _, _, dist = geodesic.inv(
            np.full_like(self.lon_arr, clon),
            np.full_like(self.lat_arr, clat),
            self.lon_arr,
            self.lat_arr
        )
        dist[np.isnan(dist)] = 1e6  # Replace NaNs with large number

        # Index of nearest grid point
        nearx, neary = np.where(np.isclose(dist, np.min(dist)))
        x, y = nearx[0], neary[0]

        # Load brightness arrays
        bright_path = self.uncut_df[self.uncut_df["File"].str.contains(self.str_pattern)].values[0][0]
        bright_arr = np.load(str(bright_path))["brightness"]

        unscaled_df = pd.DataFrame(
            glob.glob(f"/rstor/jmayhall/Model_Training_Full/unscaled_arrays/{self.folder}*"),
            columns=["File"]
        )
        unscaled_path = unscaled_df[unscaled_df["File"].str.contains(self.str_pattern)].values[0][0]
        unscaled_arr = np.load(str(unscaled_path))["brightness"]

        # Clamp indices to stay within [512, max_dim-512]
        max_dim = 5424

        def clamp(val, size: int, max_size: int):
            """
            Helper function to keep indices within min and max range
            :param val: Coordinate value
            :param size: Size of dimension
            :param max_size: Max dimension size
            :return: Clamp value
            """
            return min(max(val, size), max_size - size)

        x, y = clamp(x, 512, max_dim), clamp(y, 512, max_dim)

        # Slice arrays (lon, lat, scaled brightness, unscaled brightness)
        all_arr = np.stack(
            (self.lon_arr, self.lat_arr, bright_arr, unscaled_arr)
        )[:, x - 512:x + 512, y - 512:y + 512]

        self.lon_arr, self.lat_arr, bright_arr = all_arr[0], all_arr[1], all_arr[2]

        # Create shapely Point objects
        points_ori = zip(self.lon_arr.flatten(), self.lat_arr.flatten())
        point_obj = [[Point(lon, lat)] for lon, lat in points_ori]

        return point_obj, self.lon_arr, self.lat_arr, bright_arr
