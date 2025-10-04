# coding=utf-8
"""
Optimized: 10/01/2025
@author: John Mark Mayhall

Purpose:
--------
Class for cutting large GOES arrays to 1024x1024 regions centered on storm positions.
Uses multiprocessing for efficiency.
"""

from multiprocessing import Pool

import pandas as pd
from nearest_point import mp_geo
from shapely.geometry import Point


class Transform:
    """Class for cutting GOES arrays around tropical cyclone centers and saving them."""

    def __init__(self, uncut_arr: list, uncut_number: int, channel: list, path_len: int,
                 df: pd.DataFrame, lon_arr_al: np.ndarray, lon_arr_ep17: np.ndarray, lon_arr_ep18: np.ndarray,
                 lat_arr_al: np.ndarray, lat_arr_ep17: np.ndarray, lat_arr_ep18: np.ndarray,
                 sign: dict, online: bool, path_dict: dict) -> None:
        """
        Initializes the Transform class with necessary arrays, HURDAT2 data, and configuration.

        Parameters
        ----------
        uncut_arr : list
            List of file paths for the arrays to cut.
        uncut_number : int
            Number of files to process.
        channel : list
            Channel info: [folder_name, exec_string].
        path_len : int
            Base path length for string slicing.
        df : pd.DataFrame
            HURDAT2 dataframe containing storm info.
        lon_arr_al, lon_arr_ep17, lon_arr_ep18 : np.ndarray
            Longitude arrays for each satellite.
        lat_arr_al, lat_arr_ep17, lat_arr_ep18 : np.ndarray
            Latitude arrays for each satellite.
        sign : dict
            Dictionary mapping N/S/E/W to ±1 for coordinate conversion.
        online : bool
            Flag for choosing online/offline path prefix.
        path_dict : dict
            Dictionary mapping online/offline to base paths.
        """
        self.uncut_arr = uncut_arr
        self.uncut_number = uncut_number
        self.channel = channel
        self.path_len = path_len
        self.df = df
        self.lon_arr_al = lon_arr_al
        self.lon_arr_ep17 = lon_arr_ep17
        self.lon_arr_ep18 = lon_arr_ep18
        self.lat_arr_al = lat_arr_al
        self.lat_arr_ep17 = lat_arr_ep17
        self.lat_arr_ep18 = lat_arr_ep18
        self.sign = sign
        self.online = online
        self.path_dict = path_dict

    def main(self) -> None:
        """
        Cuts all arrays to 1024x1024 around storm centers using the nearest lat/lon points.
        Prepares arguments for multiprocessing and calls mp_geo.
        """

        needed_args = []

        for idx, file in enumerate(self.uncut_arr):
            print(f'Processing file {idx + 1} of {self.uncut_number}')

            # Extract storm ID
            storm_id = file[self.path_len:self.path_len + 8]

            # Select the appropriate lat/lon arrays
            if 'AL' in storm_id:
                current_lons, current_lats = self.lon_arr_al, self.lat_arr_al
            elif 'EP' in storm_id:
                if '2023' in storm_id:
                    current_lons, current_lats = self.lon_arr_ep18, self.lat_arr_ep18
                else:
                    current_lons, current_lats = self.lon_arr_ep17, self.lat_arr_ep17
            else:
                raise ValueError(f"Unknown basin in storm ID: {storm_id}")

            # Extract storm date/time
            date = int(file[self.path_len + 9:self.path_len + 17])
            time = int(file[self.path_len + 18:self.path_len + 22])

            # Find storm center in HURDAT2 dataframe
            storm = self.df[(self.df.ID == storm_id) & (self.df.Date == date) & (self.df.Time == time)]
            if storm.empty:
                print(f"Warning: Storm {storm_id} at {date}-{time} not found in HURDAT2.")
                continue

            # Convert Lat/Lon string to float with proper sign
            lon_str, lat_str = storm['Lon'].item(), storm['Lat'].item()
            x = float(lon_str[1:-1]) * self.sign.get(lon_str[-1], 1)
            y = float(lat_str[1:-1]) * self.sign.get(lat_str[-1], 1)
            center = Point(x, y)

            # Prepare arguments for multiprocessing
            needed_args.append({
                'lon_arr': current_lons,
                'lat_arr': current_lats,
                'center': center,
                'file': file,
                'path_len': self.path_len,
                'channel': self.channel,
                'online': self.online,
                'path_dict': self.path_dict
            })

        # Multiprocessing execution
        num_workers = 64
        with Pool(num_workers) as pool:
            pool.map(mp_geo, needed_args)
