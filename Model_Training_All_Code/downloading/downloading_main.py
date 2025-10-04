# coding=utf-8
"""
@author: John Mark Mayhall
Optimized: 10/02/2025

Purpose:
This script uses the Folder class from plot_eye_maps_folder_version to download .nc files
from a specified folder of geojsons. Users can choose the view, folder, and channels to download.
File paths may need adjustments depending on local or online environment.
"""

from plot_eye_maps_folder_version import Folder

if __name__ == '__main__':
    # ===================== User Configuration =====================
    view = 'full'  # Options could be 'full', 'zoomed', etc., depending on your setup
    folder = 'verification'  # Folder of geojsons to process
    base_geojson_path = "/rstor/jmayhall/Model_Training_Full/geojsons/"  # Path to geojson files
    channel_list = ['13', '08']  # List of channels to download

    # Base path for downloaded files
    download_base_path = "/rstor/jmayhall/Model_Training_Code/downloading/"

    # ===================== Download Loop =====================
    for channel in channel_list:
        # Construct channel-specific download path
        channel_path = f"{download_base_path}{channel}/SPoRT_"

        # Initialize Folder interface
        interface = Folder(view=view,
                           folder=folder,
                           geojson_path=base_geojson_path,
                           save_path=channel_path,
                           channel=channel)

        # Execute download
        interface.folder_download()
        print(f"Download completed for channel {channel} to {channel_path}")
