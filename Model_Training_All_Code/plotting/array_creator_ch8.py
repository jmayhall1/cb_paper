# coding=utf-8
"""
Author: John Mark Mayhall
Optimized: 10/02/2025

Purpose:
--------
This script configures parameters for scaling ABI brightness temperature data.
It passes the configuration into the `Array` class (from `nc_geojson_plotting`)
which handles:
    - Scaling brightness temperatures
    - Saving arrays
    - Optionally converting x/y to lat/lon
    - Optionally plotting ABI images with lat/lon and brightness overlays

Notes:
------
- Paths should be verified for your environment
- Channel list (`channel_lst`) controls which ABI bands to process
"""

import os

from nc_geojson_plotting import Array

if __name__ == '__main__':
    # ---------------- CONFIGURATION ---------------- #
    # List of ABI channels to process (08 = water vapor, 04 = visible, etc.)
    channel_lst = ['08']

    # Channel-specific subdirectory mapping
    path_add = {'08': 'ch8/', '04': 'ch4/'}

    # File and folder paths
    geojsons = "/rstor/jmayhall/Model_Training_Full/geojsons/"  # Path to geojsons
    folder = "test/"  # Subfolder of images

    # ABI data extraction bounds (x/y in pixel space)
    xf, xe, yf, ye = 660, 3476, 446, 3262

    # Output controls
    image, latlon = False, False  # Save figures? Save lat/lon arrays?
    xleft, xright, ytop, ybottom = -140, -50, 50, 10  # Lat/lon plotting bounds

    # Plot appearance
    levels = 25  # Number of contour levels
    cmap = "gist_gray_r"  # Colormap
    vmin, vmax = -80, 30  # Color scale limits
    dpi = 1000  # Figure resolution

    # Trim option for arrays
    trim = False

    # ---------------- MAIN EXECUTION ---------------- #

    for channel in channel_lst:
        # Build input path to ABI netCDF files for this channel
        images = os.path.join(
            "/rstor/jmayhall/Model_Training_Full/",
            path_add.get(channel, ""),
            "ncfiles/sport_data_images_"
        )

        # Instantiate Array processor with all config options
        Interface = Array(
            images, geojsons, folder, xf, xe, yf, ye,
            image, latlon, xleft, xright, ytop, ybottom,
            levels, cmap, vmin, vmax, dpi, trim, channel
        )

        # Change working directory to channel-specific plotting folder
        os.chdir(f"/rstor/jmayhall/Model_Training_Code/plotting/{channel}/")

        # Run main processing pipeline
        Interface.main()
