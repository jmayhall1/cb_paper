# coding=utf-8
"""
@author: John Mark Mayhall
Optimized: 10/02/2025

This module defines the Setup class for:
1. Converting GOES-16 ABI radiances to brightness temperatures.
2. Converting ABI projection coordinates into latitude/longitude.
3. Plotting polygons from a GeoJSON file over brightness temp imagery.

References:
GOES-R Series Product Definition and Users' Guide (PUG):
https://www.goes-r.gov/users/docs/PUG-L1b-vol3.pdf
"""

import geojson
import numpy as np


class Setup:
    """Handles GOES-16 brightness temperature calculations and plotting of polygons."""

    def __init__(self, abidata_func, geojson_filename, ax_func, xf, xe, yf, ye, trim):
        """
        Initialize Setup object.

        Parameters
        ----------
        abidata_func : xarray.Dataset
            GOES-16 ABI dataset containing radiance and projection info.
        geojson_filename : str
            Path to GeoJSON file containing polygons.
        ax_func : matplotlib.axes.Axes
            Matplotlib axis for plotting.
        xf, xe, yf, ye : int
            Subset indices for trimming data (x-min, x-max, y-min, y-max).
        trim : bool
            If True, trim arrays based on provided indices.
        """
        self.abidata_func = abidata_func
        self.geojson_filename = geojson_filename
        self.ax_func = ax_func
        self.xf, self.xe, self.yf, self.ye = xf, xe, yf, ye
        self.trim = trim

    def calc_lat_lon(self):
        """
        Converts ABI x/y scan angles to latitude/longitude coordinates.

        Steps:
        1. Subset x/y if trimming is enabled.
        2. Use GOES-16 projection geometry to calculate (lat, lon).
        3. Return 2D arrays of lat/lon in degrees.

        Returns
        -------
        lat_func, lon_func : np.ndarray
            Arrays of latitude and longitude in degrees.
        """
        # Create 2D grid of ABI scan angles
        x, y = np.meshgrid(self.abidata_func.x.values, self.abidata_func.y.values)
        if self.trim:
            x, y = x[self.xf:self.xe, self.yf:self.ye], y[self.xf:self.xe, self.yf:self.ye]

        # Extract projection parameters
        proj = self.abidata_func.goes_imager_projection
        major, minor = float(proj.semi_major_axis), float(proj.semi_minor_axis)
        h = float(proj.perspective_point_height) + major
        lambda_0 = np.radians(float(proj.longitude_of_projection_origin))

        # GOES-R projection math (vectorized)
        a = (np.sin(x)) ** 2 + (np.cos(x)) ** 2 * ((np.cos(y)) ** 2 + (major ** 2 / minor ** 2) * (np.sin(y)) ** 2)
        b = -2 * h * np.cos(x) * np.cos(y)
        c = h ** 2 - major ** 2
        r_s = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

        s_x = r_s * np.cos(x) * np.cos(y)
        s_y = -r_s * np.sin(x)
        s_z = r_s * np.cos(x) * np.sin(y)

        lat_func = np.degrees(np.arctan((major ** 2 / minor ** 2) * (s_z / np.sqrt((h - s_x) ** 2 + s_y ** 2))))
        lon_func = np.degrees(lambda_0 - np.arctan(s_y / (h - s_x)))

        return lat_func, lon_func

    def brightness_calc(self):
        """
        Converts GOES-16 radiances to brightness temperatures in Celsius.

        Returns
        -------
        brightness_func : np.ndarray
            2D array of brightness temperatures in Celsius.
        """
        # Trim if requested, otherwise use full domain
        lv = self.abidata_func.CMI.values[
            self.xf:self.xe, self.yf:self.ye] if self.trim else self.abidata_func.CMI.values
        return lv - 273.15  # Convert Kelvin → Celsius

    def plot_polygons(self):
        """
        Plots polygons from a GeoJSON file onto the given Matplotlib axis.
        Colors:
            Yellow = TCB present (property 'TCB' == '1')
            Red    = TCB absent (property 'TCB' == '0')
        """
        with open(self.geojson_filename) as geojson_file:
            geojson_data = geojson.load(geojson_file)

        for feature in geojson_data.get("features", []):
            coords = feature["geometry"]["coordinates"][0]  # Outer polygon coordinates
            x_geo, y_geo = zip(*coords)  # Separate x and y
            color = {"1": "yellow", "0": "red"}.get(feature.get("properties", {}).get("TCB"))
            if color:  # Only plot if TCB label exists
                self.ax_func.plot(x_geo, y_geo, c=color)
