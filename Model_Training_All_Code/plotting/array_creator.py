# coding=utf-8
"""
Author: John Mark Mayhall
Optimized: 10/02/2025

Purpose:
--------
This script configures and runs the Array processing pipeline to:
    - Scale brightness temperature (BT) data from ABI channels
    - Save brightness arrays
    - Optionally convert x/y grid to lat/lon
    - Optionally plot ABI images with brightness overlays

Notes:
------
- Paths may need to be updated for your local environment
- The channel list (channel_lst) controls which ABI bands are processed
"""

import os

from nc_geojson_plotting import Array

if __name__ == '_main__':
    # ---------------- CONFIGURATION ---------------- #
    # Channels to process (13 = IR longwave, 08 = water vapor, etc.)
    channel_lst = ['13']

    # Map channel IDs to their subdirectories
    path_add = {'08': 'ch8/', '04': 'ch4/', '13': 'ch13/'}

    # File and folder paths
    geojsons = "/rstor/jmayhall/Model_Training_Full/geojsons/"  # Path to .geojson files
    folder = "test/"  # Subfolder of interest

    # ABI data bounds in pixel space
    xf, xe, yf, ye = 660, 3476, 446, 3262

    # Output options
    image = False  # Save figure?
    latlon = False  # Save lat/lon arrays?
    trim = False  # Trim image edges?

    # Plot appearance settings
    xleft, xright, ytop, ybottom = -140, -50, 50, 10  # Lat/lon plot limits
    levels = 25  # Contour levels
    cmap = "gist_gray_r"  # Colormap
    vmin, vmax = -80, 30  # Temperature scale limits
    dpi = 1000  # Figure resolution

    # ---------------- MAIN EXECUTION ---------------- #

    for channel in channel_lst:
        # Construct path to ABI netCDF files for this channel
        images = os.path.join(
            "/rstor/jmayhall/Model_Training_Full/",
            path_add.get(channel, ""),
            "ncfiles/sport_data_images_"
        )

        # Create Array processor instance with configuration
        interface = Array(
            images, geojsons, folder,
            xf, xe, yf, ye,
            image, latlon,
            xleft, xright, ytop, ybottom,
            levels, cmap, vmin, vmax,
            dpi, trim, channel
        )

        # Change working directory to channel-specific plotting folder
        os.chdir(f"/rstor/jmayhall/Model_Training_Code/plotting/{channel}/")

        # Run main processing pipeline
        interface.main()
