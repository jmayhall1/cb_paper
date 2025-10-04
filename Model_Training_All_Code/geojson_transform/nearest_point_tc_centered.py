# coding=utf-8
"""
@author: John Mark Mayhall
@author: Patrick Duran
Optimized: 10/02/2025

This module defines the Nearest class, which finds the nearest array index to a given
polygon center using pyproj (geodesic distance). It then extracts a 1024x1024 subset
of lon/lat/brightness arrays centered on that nearest point, and returns shapely point objects.
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

        clon, clat = self.center  # Polygon center (lon, lat)

        # Copy lat/lon grids
        dlon, dlat = np.copy(self.lon_arr), np.copy(self.lat_arr)

        # Compute geodesic distance from center to every grid point
        geodesic = Geod(ellps="WGS84")
        _, _, dist = geodesic.inv(
            np.full_like(dlon, clon),  # repeat lon center to match grid
            np.full_like(dlat, clat),  # repeat lat center
            dlon, dlat
        )

        # Replace NaNs with a large number so they're ignored
        dist[np.isnan(dist)] = 1e6

        # Index of nearest grid point
        nearx, neary = np.where(np.isclose(dist, np.min(dist)))

        # Load scaled and unscaled brightness arrays for this pattern
        bright_path = self.uncut_df[self.uncut_df["File"].str.contains(self.str_pattern)].values[0][0]
        bright_arr = np.load(str(bright_path))["brightness"]

        unscaled_df = pd.DataFrame(
            glob.glob(f"/rstor/jmayhall/Model_Training_Full/unscaled_arrays/{self.folder}*"),
            columns=["File"]
        )
        unscaled_path = unscaled_df[unscaled_df["File"].str.contains(self.str_pattern)].values[0][0]
        unscaled_arr = np.load(str(unscaled_path))["brightness"]

        # Ensure the nearest index stays within valid slicing range (center +/- 512 pixels)
        max_dim = 5424  # full array dimension
        x, y = nearx[0], neary[0]

        def clamp_center(val, size, max_size):
            """Ensure the index is within [size, max_size-size] range."""
            return min(max(val, size), max_size - size)

        x = clamp_center(x, 512, max_dim)
        y = clamp_center(y, 512, max_dim)

        # Slice arrays: lon, lat, scaled brightness, unscaled brightness
        all_arr = np.stack(
            (np.copy(self.lon_arr), np.copy(self.lat_arr), np.copy(bright_arr), np.copy(unscaled_arr))
        )[:, x - 512:x + 512, y - 512:y + 512]

        # Update lon/lat arrays
        self.lon_arr = np.copy(all_arr[0])
        self.lat_arr = np.copy(all_arr[1])
        bright_arr = np.copy(all_arr[2])  # final scaled brightness

        # Flatten lon/lat to make shapely Point objects
        list_lon = self.lon_arr.flatten()
        list_lat = self.lat_arr.flatten()
        points_ori = zip(list_lon, list_lat)

        # Create shapely Point objects (nested list structure preserved)
        point_obj = [[Point(lon, lat)] for lon, lat in points_ori]

        return point_obj, self.lon_arr, self.lat_arr, bright_arr
