# coding=utf-8
"""
Optimized: 10/01/2025
Author: John Mark Mayhall

Purpose:
--------
Calculate the nearest array index to a polygon center, extract a 1024x1024 region
around it, and update the array using a channel-specific transformation.

Notes:
------
- Uses `np.nanargmin` on squared Euclidean distances for nearest point calculation.
- Supports multiprocessing via a simple wrapper.
- Ensures array shapes are validated after processing.
"""

from multiprocessing import Lock

import numpy as np

# Shared lock for thread-safe operations (writing arrays)
lock = Lock()


def distance(lon_arr: np.ndarray, lat_arr: np.ndarray, center,
             file: str, path_len: int, channel, online: bool, path_dict: dict) -> None:
    """
    Calculates the nearest grid point to the storm center, extracts a 1024x1024
    slice of the lat/lon and brightness arrays, and executes a channel-specific
    processing command.

    Parameters
    ----------
    lon_arr : np.ndarray
        2D longitude array
    lat_arr : np.ndarray
        2D latitude array
    center : shapely.geometry.Point
        Storm center coordinates
    file : str
        Path to brightness array file (.npz)
    path_len : int
        Length of base file path (used in channel exec)
    channel : list
        Channel-specific list [folder_name, exec_string]
    online : bool
        Flag for determining path prefix
    path_dict : dict
        Dictionary mapping `online` to base paths

    Returns
    -------
    None
    """
    # Storm center coordinates
    clon, clat = center.x, center.y

    # Compute squared distance to all grid points (vectorized)
    dist_sq = (lon_arr - clon) ** 2 + (lat_arr - clat) ** 2
    flat_idx = np.nanargmin(dist_sq)

    # Convert flat index to 2D array indices
    neary, nearx = np.unravel_index(flat_idx, dist_sq.shape)

    # Define 1024x1024 slice bounds
    miny, maxy = neary - 512, neary + 512
    minx, maxx = nearx - 512, nearx + 512
    gridy, gridx = lon_arr.shape

    slicey = slice(max(miny, 0), min(maxy, gridy))
    slicex = slice(max(minx, 0), min(maxx, gridx))

    # Determine slices for final 1024x1024 array
    slice_final_y = slice(max(-miny, 0), max(1024 - (maxy - gridy), 0))
    slice_final_x = slice(max(-minx, 0), max(1024 - (maxx - gridx), 0))

    # Load the brightness temperature array
    bright_arr = np.load(file)["brightness"]

    _ = [path_len, online, path_dict, slicey, slicex, slice_final_x,
         slice_final_y, bright_arr]  # Ensure needed variables exists

    # Thread-safe execution of channel-specific code
    with lock:
        exec(channel[1])  # expects `all_arr`, `slice_final_x`, `slice_final_y` variables

    # Validate resulting array shape
    if all_arr.shape not in [(1024, 1024), (3, 1024, 1024)]:
        print("Unexpected array shape:", all_arr.shape)
        print(all_arr)
        raise ValueError("Processed array shape is invalid.")


def mp_geo(kwargs: dict):
    """
    Multiprocessing wrapper for `distance()`.

    Parameters
    ----------
    kwargs : dict
        Dictionary of keyword arguments to pass to `distance()`

    Returns
    -------
    None
    """
    return distance(**kwargs)
