# coding=utf-8
"""
Optimized: 10/01/2025
Author: John Mark Mayhall

Setup class for generating binary containment masks from GeoJSON storm polygons.
Optimizations:
- Clearer error handling for file/array size issues.
- Safe polygon handling (supports both Polygon and MultiPolygon).
- Vectorized nearest-point search using pyproj.Geod for accuracy.
- Memory-safe reshaping and trimming.
"""

import json

import numpy as np
import pandas as pd
from pyproj import Geod
from shapely import points as shapely_points
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely.prepared import prep


class Setup:
    """
    Process a storm's GeoJSON record and generate a binary containment mask
    aligned to the nearest 1024x1024 region of lon/lat grid points.

    Attributes
    ----------
    y_true_file : str
        Path to GeoJSON file containing polygon(s).
    lon_arr : np.ndarray
        Full longitude grid array.
    lat_arr : np.ndarray
        Full latitude grid array.
    storm : pd.DataFrame
        Row(s) from HURDAT with storm metadata (contains Lon, Lat).
    """

    def __init__(self, y_true_file: str, lon_arr: np.ndarray, lat_arr: np.ndarray,
                 storm: pd.DataFrame):
        self.y_true_file = y_true_file
        self.lon_arr = lon_arr
        self.lat_arr = lat_arr
        self.storm = storm
        self.geo_poly = None
        self.geod = Geod(ellps='WGS84')

    # ----------------------------
    # Data Loading
    # ----------------------------
    def load_geojson(self):
        """
        Load polygons from a GeoJSON file.

        Returns
        -------
        list[Polygon]
            List of shapely Polygon objects.
        """
        try:
            with open(self.y_true_file) as f:
                data = json.load(f)
            return [shape(feature['geometry']) for feature in data['features']]
        except Exception as e:
            print(f"Error loading GeoJSON {self.y_true_file}: {e}")
            return []

    # ----------------------------
    # Storm Center
    # ----------------------------
    def get_storm_center(self) -> Point:
        """
        Extract storm center from storm DataFrame.

        Returns
        -------
        Point
            Shapely Point of storm center.
        """
        lon, lat = float(self.storm.iloc[0]['Lon']), float(self.storm.iloc[0]['Lat'])
        return Point(lon, lat)

    # ----------------------------
    # Nearest Grid Point
    # ----------------------------
    def find_nearest_point(self, center: Point) -> tuple[int, int]:
        """
        Find the nearest grid point to the storm center.

        Returns
        -------
        tuple[int, int]
            Indices (x, y) of the nearest lon/lat grid point.
        """
        try:
            _, _, dist = self.geod.inv(
                np.full_like(self.lon_arr, center.x),
                np.full_like(self.lat_arr, center.y),
                self.lon_arr, self.lat_arr
            )
            return np.unravel_index(np.nanargmin(dist), dist.shape)
        except Exception as e:
            print(f"Error computing nearest point: {e}")
            return self.lon_arr.shape[0] // 2, self.lon_arr.shape[1] // 2

    # ----------------------------
    # Trimming
    # ----------------------------
    def trim_arrays(self, nearx: int, neary: int, size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        """
        Trim lon/lat arrays to a square region centered on (nearx, neary).

        Parameters
        ----------
        nearx, neary : int
            Indices of storm center on grid.
        size : int
            Output region size (default: 1024).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Trimmed longitude and latitude arrays.
        """
        half = size // 2
        x_min, x_max = nearx - half, nearx + half
        y_min, y_max = neary - half, neary + half

        # Clip bounds to avoid IndexErrors
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(self.lon_arr.shape[0], x_max), min(self.lon_arr.shape[1], y_max)

        return self.lon_arr[x_min:x_max, y_min:y_max], self.lat_arr[x_min:x_max, y_min:y_max]

    # ----------------------------
    # Containment Mask
    # ----------------------------
    def generate_containment_array(self, points: np.ndarray, size: int = 1024) -> np.ndarray:
        """
        Create a binary mask where pixels inside storm polygon = 1.

        Parameters
        ----------
        points : np.ndarray
            Array of (lon, lat) pairs with shape (H, W, 2).
        size : int
            Expected square mask size (default: 1024).

        Returns
        -------
        np.ndarray
            Binary containment mask.
        """
        if self.geo_poly is None:
            print("GeoJSON polygon not loaded. Returning zeros.")
            return np.zeros((size, size), dtype=np.uint8)

        flat_points = points.reshape(-1, 2)
        multi_points = shapely_points(flat_points)

        # Use shapely prepared geometry for fast point-in-polygon checks
        mask = prep(self.geo_poly).contains(multi_points).astype(np.uint8)
        return mask.reshape(points.shape[0], points.shape[1])

    # ----------------------------
    # Main Pipeline
    # ----------------------------
    def main(self) -> np.ndarray:
        """
        Execute full processing pipeline:
        1. Load polygons from GeoJSON
        2. Compute storm center
        3. Find the nearest grid point
        4. Trim lon/lat arrays
        5. Generate containment mask

        Returns
        -------
        np.ndarray
            Binary mask (storm polygon vs background).
        """
        polygons = self.load_geojson()
        if not polygons:
            return np.zeros((1024, 1024), dtype=np.uint8)

        # Handle MultiPolygon case automatically
        self.geo_poly = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)

        center = self.get_storm_center()
        nearx, neary = self.find_nearest_point(center)
        lon_trim, lat_trim = self.trim_arrays(nearx, neary)
        points = np.dstack((lon_trim, lat_trim))

        return self.generate_containment_array(points, size=points.shape[0])
