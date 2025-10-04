# coding=utf-8
"""
@author: John Mark Mayhall
Optimized: 10/02/2025

This module sets up the transformation of GeoJSON polygons into
2D arrays aligned with GOES ABI grid points.

Workflow:
1. Generate lat/lon point arrays (via PointMethod).
2. Load GeoJSON polygons for each file in a folder.
3. For each polygon, determine which points fall inside.
4. Save a binary mask array (1 = inside polygon, 0 = outside).
"""

import json
from multiprocessing import Lock
from multiprocessing.pool import ThreadPool

from geo_transform_processing_tc_centered import Creation
from point_creation_tc_centered import PointMethod


class Transform:
    """Main driver class for turning GeoJSONs into binary arrays."""

    def __init__(self, path, folder, path_arr, uncut_df, rotate):
        """
        Parameters
        ----------
        path : str
            Path to the folder containing GeoJSONs.
        folder : str
            Subfolder name where files are located.
        path_arr : str
            Path to save processed arrays.
        uncut_df : pandas.DataFrame
            Reference grid dataframe (used for aligning polygons with ABI grid).
        rotate : bool
            Whether to rotate the arrays for consistency with ABI data.
        """
        self.path = path
        self.folder = folder
        self.path_arr = path_arr
        self.uncut_df = uncut_df
        self.rotate = rotate

    def main(self):
        """
        Orchestrates the workflow:
        1. Generate grid point objects (lat/lon arrays).
        2. Parallelize GeoJSON file processing.
        """
        # Generate point arrays for lat/lon grid
        point_interface = PointMethod(self.path, self.folder, self.path_arr)
        geo_files, nfiles, lon_arr, lat_arr = point_interface.pointcreate()

        def process_file(lock, idx, filename, total, path, folder, lon, lat, uncut_df, rotate):
            """
            Process a single GeoJSON file:
            - Load polygons
            - Extract coordinates
            - Pass polygons to Creation class for rasterization
            """
            print(f"Processing file {idx + 1} of {total}: {filename}")

            with open(f"{path}{folder}{filename}") as f:
                data = json.load(f)

            # Extract polygon coordinates for each feature
            coords = [feature["geometry"]["coordinates"][0] for feature in data["features"]]

            # Initialize processing interface for polygon → array
            processor = Creation(
                lock, coords, [], filename, len(data["features"]),
                idx, uncut_df, folder, lon, lat, rotate
            )
            processor.processing()

        # Use thread pool for parallel processing
        lock = Lock()
        with ThreadPool(processes=8) as pool:
            pool.starmap(
                process_file,
                [
                    (lock, idx, file, nfiles, self.path, self.folder,
                     lon_arr, lat_arr, self.uncut_df, self.rotate)
                    for idx, file in enumerate(geo_files)
                ]
            )
