# coding=utf-8
"""
Optimized: 10/01/2025
@author: John Mark Mayhall

Purpose:
--------
Transform GeoJSON files into brightness arrays for C08 channel.
Crops the arrays to 1024x1024 around the storm center and saves results.
"""

import glob

import numpy as np
import pandas as pd
from geo_transform_setup import Transform


def load_latlon_arrays(online: bool, path_dict: dict) -> dict:
    """Load all necessary latitude and longitude arrays for GOES satellites."""
    arrays = {}
    for basin, file_suffix in [('AL', 'AL_latlon.npz'),
                               ('EP17', 'EP_latlon_17.npz'),
                               ('EP18', 'EP_latlon_18.npz')]:
        full_path = f"{path_dict[online]}cataloging/nc_process/geojson_transform/latlon_arrays/{file_suffix}"
        data = np.load(full_path)
        arrays[f'lon_{basin}'] = np.array(data['lon'], dtype=np.float32)
        arrays[f'lat_{basin}'] = np.array(data['lat'], dtype=np.float32)
    return arrays


if __name__ == "__main__":
    online = True
    path_dict = {True: '/rstor/jmayhall/', False: '//uahdata/rstor/'}

    # Load lat/lon arrays
    latlon_arrays = load_latlon_arrays(online, path_dict)

    # Load HURDAT2 data
    df_AL = pd.read_csv(f"{path_dict[online]}cataloging/hurdat_update_interp_AL.txt", sep='\t')
    df_EP = pd.read_csv(f"{path_dict[online]}cataloging/hurdat_update_interp_EP.txt", sep='\t')
    df = pd.concat([df_AL, df_EP])[['ID', 'Date', 'Time', 'Lat', 'Lon']]
    df['Date'] = pd.to_numeric(df['Date'], downcast="integer")
    df['Time'] = pd.to_numeric(df['Time'], downcast="integer")

    # Define channels to process (C08 channel here)
    channel_arr = np.array([np.array([
        'C08_scaled/',
        # Code string for creating cut arrays
        ("uncut_all = bright_arr[slicey, slicex]; "
         "global all_arr; "
         "all_arr = np.full(shape=(1024, 1024), fill_value=np.array([10]), dtype=np.float32); "
         "all_arr[slice_final_y, slice_final_x] = uncut_all; "
         "save_path = (f'{path_dict.get(online)}cataloging/nc_process/geojson_transform/completed_arrays/' "
         "+ channel[0] + file[path_len:].replace('.npz', '')); "
         "np.savez_compressed(f'{save_path}_cut', brightness=all_arr);")
    ])])

    # Dictionary for coordinate sign conversion
    sign = {'W': -1, 'S': -1}

    # Process each channel
    for channel in channel_arr:
        folder_path = f"{path_dict[online]}cataloging/nc_process/plotting/{channel[0]}"
        path_len = len(folder_path)

        # Find unprocessed arrays
        all_files = set(glob.glob(f"{folder_path}/*.npz"))
        done_files = set(glob.glob(
            f"{path_dict[online]}cataloging/nc_process/geojson_transform/completed_arrays/{channel[0]}/*.npz"))
        # Normalize paths for comparison
        done_files = {f.replace('geojson_transform/completed_arrays', 'plotting').replace('_cut', '') for f in
                      done_files}
        uncut_arr = np.array(list(all_files - done_files))
        uncut_number = len(uncut_arr)

        # Initialize and run Transform
        interface = Transform(
            uncut_arr=uncut_arr,
            uncut_number=uncut_number,
            channel=channel,
            path_len=path_len,
            df=df,
            lon_arr_al=latlon_arrays['lon_AL'],
            lon_arr_ep17=latlon_arrays['lon_EP17'],
            lon_arr_ep18=latlon_arrays['lon_EP18'],
            lat_arr_al=latlon_arrays['lat_AL'],
            lat_arr_ep17=latlon_arrays['lat_EP17'],
            lat_arr_ep18=latlon_arrays['lat_EP18'],
            sign=sign,
            online=online,
            path_dict=path_dict
        )
        interface.main()
