# coding=utf-8
"""
Last Edited: 10/01/2025
Author: John Mark Mayhall

Description:
------------
Optimized setup for downloading GOES-16/17/18 ABI L2 .nc files from AWS S3.

Workflow:
---------
- Initializes a global S3 filesystem and multiprocessing lock.
- Searches AWS GOES bucket for specific .nc files.
- Downloads and saves the files locally with informative naming.

Notes:
------
- Adjust paths, datetime formats, or channels as needed.
- Requires the `s3fs` and `requests` packages.
"""

from multiprocessing import Lock

import pandas as pd
import requests
import s3fs

# Globals for multiprocessing
fs, lock, user_path = None, None, None


def init_worker():
    """
    Initialize global S3 filesystem, lock, and the user path variable for multiprocessing workers.
    This prevents reinitializing S3 connection in every process.
    """
    global fs, lock, user_path
    fs = s3fs.S3FileSystem(anon=True)
    user_path = '/rstor/jmayhall/cataloging/nc_file/'
    lock = Lock()


def search(file_dict: dict, search_str: str) -> str | None:
    """
    Search a nested dict of lists for a string containing `search_str`.

    Parameters
    ----------
    file_dict : dict
        Dictionary of lists of filenames.
    search_str : str
        Substring to search for in filenames.

    Returns
    -------
    str or None
        Matching filename if found, otherwise None.
    """
    for v_list in file_dict.values():
        for v in v_list:
            if search_str in v:
                return v
    return None


def downloading(index: int, row: pd.Series, length: int, channel: str, sat: str):
    """
    Download a GOES ABI .nc file from AWS S3 for a specific storm record.

    Parameters
    ----------
    index : int
        Position of the current row in the dataset.
    row : pd.Series
        Row from HURDAT2 DataFrame containing storm info.
    length : int
        Total number of rows in dataset (for progress reporting).
    channel : str
        ABI channel (e.g., '08', '13').
    sat : str
        GOES satellite number ('16', '17', '18').
    """
    print(f"Downloading file {index + 1} of {length} for channel {channel} on GOES-{sat}")

    # Parse timestamp
    try:
        timestamp = pd.to_datetime(row['Date'] + row['Time'].zfill(4), format="mixed")
    except Exception as e:
        print(f"[ERROR] Invalid date/time format at index {index}: {e}")
        return

    # Build S3 search path (year/day-of-year/hour)
    path = (
        f"s3://noaa-goes{sat}/ABI-L2-CMIPF/"
        f"{timestamp.year}/{str(timestamp.dayofyear).zfill(3)}/{str(timestamp.hour).zfill(2)}/*.nc"
    )

    try:
        # List matching files from S3
        file_list = fs.glob(path)
        final_file = next((f for f in file_list if f"C{channel}" in f), None)

        if final_file:
            # Build public HTTPS URL from S3 key
            url = f"https://noaa-goes{sat}.s3.amazonaws.com/{final_file[12:]}"
            response = requests.get(url, verify=False)

            # Local save path
            filename = (
                f"{user_path}"
                f"{row['ID']}_{row['Date']}_{row['Time'].zfill(4)}_C{channel}.nc"
            )

            # Write safely with lock (prevents collisions in multiprocess)
            with lock:
                with open(filename, "wb") as f:
                    f.write(response.content)
        else:
            print(f"[WARNING] No matching file found for C{channel} at index {index}")

    except Exception as e:
        print(f"[ERROR] Failed to download file at index {index}: {e}")


def mp_downloading(kwargs: dict):
    """
    Wrapper for multiprocessing `map` calls.

    Parameters
    ----------
    kwargs : dict
        Dictionary of arguments for `downloading`.
    """
    downloading(**kwargs)
