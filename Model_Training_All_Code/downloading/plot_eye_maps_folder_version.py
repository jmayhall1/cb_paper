# coding=utf-8
"""
@author: John Mark Mayhall
@author: Patrick Duran
Optimized: 10/02/2025

Purpose:
This module allows users to input a folder name and download GOES-16 files for a specific time frame via AWS.
Files are accessed using HTTPS requests instead of the "s3fs" package for authentication-free downloading.

Notes:
- Assumes files follow the SPoRT_YYYYMMDDTHHMMSS_* naming convention.
- Adjustments may be needed for different naming formats or different channels.
"""

import os

import pandas as pd
import requests
import s3fs
from downloading_setup_folder import Setup
from requests.packages.urllib3.exceptions import InsecureRequestWarning


class Folder:
    """Class to handle downloading GOES-16 files for a given folder, channel, and view."""

    def __init__(self, view: str, folder: str, path: str, sec_path: str, channel: int):
        """
        Initializes the Folder object.

        Args:
            view (str): 'conus' or 'full' determines the domain.
            folder (str): Local folder to store downloaded files.
            path (str): Path used in Setup for file operations.
            sec_path (str): Secondary path where files are saved.
            channel (int): GOES-16 ABI channel number (e.g., 13 for C13).
        """
        self.view = view
        self.folder = folder
        self.path = path
        self.sec_path = sec_path
        self.channel = channel

    def folder_download(self):
        """
        Downloads .nc files for all timestamps calculated by the Setup class.

        Workflow:
        1. Disables SSL warnings for HTTPS downloads.
        2. Uses Setup to calculate timestamps and identifiers.
        3. Iterates through timestamps, generating file URLs based on the view.
        4. Downloads the first matching file for each timestamp and saves it locally.
        """
        # Disable insecure request warnings
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

        # Initial Setup instance to calculate timestamps and IDs
        interface = Setup(self.folder, 0, self.channel, self.path)
        dates, times, s_ids = interface.calc_time()

        # Anonymous S3 filesystem for listing files
        fs = s3fs.S3FileSystem(anon=True)

        for date_str, time_val, s_id in zip(dates, times, s_ids):
            ts = pd.Timestamp(time_val)

            # Determine S3 path based on view
            if self.view.lower() == 'conus':
                s3_path = f's3://noaa-goes16/ABI-L2-CMIPC/{ts.year:04d}/{ts.dayofyear:03d}/{ts.hour:02d}/*.nc'
            elif self.view.lower() == 'full':
                s3_path = f's3://noaa-goes16/ABI-L2-CMIPF/{ts.year:04d}/{ts.dayofyear:03d}/{ts.hour:02d}/*.nc'
            else:
                raise ValueError(f"Invalid view option: {self.view}. Use 'conus' or 'full'.")

            # Find all matching files in S3
            filelist = fs.glob(s3_path)

            if not filelist:
                print(f"No files found for {ts} at view '{self.view}'")
                continue

            # Run Setup to identify target channel files
            interface = Setup(self.folder, filelist, f'C{self.channel}', self.path)
            target_files = interface.search()

            if not target_files:
                print(f"No channel {self.channel} files found for {ts}")
                continue

            # Download the first matching file
            file_key = target_files[0]
            url = f'https://noaa-goes16.s3.amazonaws.com/{file_key[12:]}'  # AWS public URL

            try:
                response = requests.get(url, verify=False)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"Failed to download {url}: {e}")
                continue

            # Save file using context manager
            filename = os.path.join(self.sec_path, f"{date_str}_labels_WGS84_{s_id}.nc")
            with open(filename, 'wb') as f:
                f.write(response.content)

            print(f"Downloaded and saved: {filename}")
