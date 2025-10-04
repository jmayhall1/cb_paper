# coding=utf-8
"""
@author: John Mark Mayhall
Optimized: 10/02/2025

This module processes GeoJSON polygons (TCB = 1 regions) into binary arrays
aligned with ABI brightness temperature data. Each polygon is rasterized into
a mask (1 = inside polygon, 0 = outside). Outputs are saved as .npz arrays and
preview images for QC.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nearest_point_tc_centered import Nearest
from shapely import MultiPolygon, Polygon


class Creation:
    """
    Handles the processing of GeoJSON polygons into 2D binary arrays aligned
    with GOES ABI data. Also saves brightness temperature arrays and rotated
    versions if required.
    """

    def __init__(self, lock, coords, contain_final, file, p, num, uncut_df,
                 folder, lon_arr, lat_arr, rotate):
        """
        Parameters
        ----------
        lock : multiprocessing.Lock
            Synchronization lock for safe file saving.
        coords : list
            List of polygon coordinate arrays from GeoJSON.
        contain_final : list
            List that will be populated with binary mask values.
        file : str
            Filename of the GeoJSON being processed.
        p : int
            Polygon index (last polygon in current file).
        num : int
            File index in batch processing.
        uncut_df : pandas.DataFrame
            Reference ABI grid dataframe.
        folder : str
            Folder where GeoJSON resides.
        lon_arr : np.ndarray
            Longitude array of the ABI grid.
        lat_arr : np.ndarray
            Latitude array of the ABI grid.
        rotate : bool
            Whether to generate rotated arrays (90, 180, 270 degrees).
        """
        self.l = lock
        self.coords = coords
        self.contain_final = contain_final
        self.file, self.p, self.num = file, p, num
        self.uncut_df, self.folder = uncut_df, folder
        self.lon_arr, self.lat_arr = lon_arr, lat_arr
        self.rotate = rotate

    def processing(self):
        """Processes GeoJSON polygons and generates binary mask + brightness arrays."""

        # Load HURDAT2 reference file with storm track info
        df = pd.read_csv(
            '/rstor/jmayhall/Model_Training_Code/geojson_transform/hurdat_update_interp_2018.txt',
            sep='\t', usecols=['ID', 'Date', 'Time', 'Lat', 'Lon']
        )
        df['Date'] = pd.to_numeric(df['Date'], downcast="integer")
        df['Time'] = pd.to_numeric(df['Time'], downcast="integer")

        # File naming details (extract storm ID, date, time)
        storm_id = self.file[35:43]
        storm_date = int(self.file[6:14])
        storm_time_str = self.file[15:21]
        storm_time = int(self.file[15:19])

        # Build polygons into shapely MultiPolygon
        poly_list = [Polygon([(x, y) for x, y in poly]) for poly in self.coords]
        geo_poly = MultiPolygon(poly_list)

        # Extract storm center coordinates from HURDAT
        lon_str = df.loc[
            (df['ID'] == storm_id) & (df['Date'] == storm_date) & (df['Time'] == storm_time),
            'Lon'
        ].values[0].replace(' ', '')
        lat_str = df.loc[
            (df['ID'] == storm_id) & (df['Date'] == storm_date) & (df['Time'] == storm_time),
            'Lat'
        ].values[0].replace(' ', '')

        def parse_coord(coord_str):
            """Convert coordinate string like '12.3N' to float value."""
            multiplier = 1 if coord_str[-1] in ['N', 'E'] else -1
            return float(coord_str[:-1]) * multiplier

        lon, lat = parse_coord(lon_str), parse_coord(lat_str)
        center = [lon, lat]

        # File naming pattern for outputs
        str_pattern = f"{storm_date}T{storm_time_str.ljust(6, '0')}_labels_WGS84_{storm_id}"

        # Nearest point mapping (align storm-centered ABI arrays)
        nearest_interface = Nearest(
            self.lon_arr, self.lat_arr, center,
            self.uncut_df, self.folder, self.num, self.p,
            self.file, str_pattern
        )
        final_obj, self.lon_arr, self.lat_arr, bright_arr = nearest_interface.distance()

        print("Array and list slicing complete.")

        # Generate mask (1 if inside polygon, 0 otherwise)
        self.contain_final = np.array([1 if geo_poly.contains(pt) else 0 for pt in final_obj])
        self.contain_final = self.contain_final.reshape((1024, 1024))

        # Save original orientation arrays
        self.l.acquire()
        np.savez(f"SPoRT_{str_pattern}_containedrot0", self.contain_final)
        np.savez(f"SPoRT_{str_pattern}_scaledrot0", brightness=bright_arr)

        # Quick visualization for QC
        fig, ax = plt.subplots()
        ax.contourf(bright_arr, cmap='Greys', levels=110, vmax=2)
        ax.contour(self.contain_final)
        ax.invert_yaxis()
        plt.savefig(f"SPoRT_{str_pattern}_rot0.png")
        plt.close('all')

        # Save rotated versions if required
        if self.rotate:
            rot_map = {'0': 90, '1': 180, '2': 270}
            for i, angle in rot_map.items():
                np.savez(
                    f"SPoRT_{str_pattern}_scaledrot{angle}",
                    brightness=np.rot90(bright_arr.copy(), k=(int(i) + 1))
                )
                np.savez(
                    f"SPoRT_{str_pattern}_containedrot{angle}",
                    np.rot90(self.contain_final.copy(), k=(int(i) + 1))
                )
        self.l.release()
