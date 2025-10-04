# coding=utf-8
"""
@author: John Mark Mayhall
@author: Patrick Duran
Optimized: 10/02/2025
ABI Coordinates to Lat/Lon and Radiance to Brightness Temperature Conversion
Purpose:
--------
This code allows a user to:
1. Read a list of geojsons and plot polygons defined by their coordinates.
2. Read a list of netCDF files and plot brightness temperatures in lat/lon space.

Notes:
------
- Paths may need to be adjusted for your environment.
- Figure names are auto-generated based on ABI netCDF filenames.
- Additional edits may be needed for colormap, levels, or scaling logic.
"""

import os
import warnings
import glob

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from plotting_setup import Setup
from scaler import Scale


class Array:
    """Main driver class for processing ABI netCDF + geojsons into brightness arrays."""

    def __init__(self, images, geojsons, folder, xf, xe, yf, ye, image, latlon,
                 xleft, xright, ytop, ybottom, levels, cmap, vmin, vmax,
                 dpi, trim, channel):
        # File paths and folders
        self.images, self.geojsons, self.folder = images, geojsons, folder

        # Crop bounds for ABI
        self.xf, self.xe, self.yf, self.ye = xf, xe, yf, ye

        # Plotting / saving options
        self.image, self.latlon = image, latlon
        self.xleft, self.xright, self.ytop, self.ybottom = xleft, xright, ytop, ybottom
        self.levels, self.cmap, self.vmin, self.vmax, self.dpi = levels, cmap, vmin, vmax, dpi

        # Misc options
        self.trim, self.channel = trim, channel

    def main(self):
        """Main pipeline: read ABI files, match geojsons, compute lat/lon + brightness, save + scale arrays."""
        warnings.filterwarnings("ignore", category=RuntimeWarning)  # Ignore NaN warnings

        # Collect ABI files + geojson files
        lst_images = os.listdir(str(os.path.join(self.images, self.folder)))
        lst_geojsons = os.listdir(str(os.path.join(self.geojsons, self.folder)))

        # Exclude already processed files
        done = glob.glob(f"/rstor/jmayhall/Model_Training_Full/uncut_arrays/{self.folder}/*.npz")
        # Normalize "done" list to match naming convention of lst_images/geojsons
        done_nc = [img[74:108].replace("WGS84", "WGS84.nc") for img in done]
        done_geo = [img.replace("nc", "geojson") for img in done_nc]

        lst_images = list(set(lst_images) - set(done_nc))
        lst_geojsons = list(set(lst_geojsons) - set(done_geo))
        geojsons_list = [os.path.join(self.geojsons, self.folder, f) for f in lst_geojsons]

        # Hold all unscaled brightness arrays
        unscaled = []

        # ---------------- Loop over ABI files ---------------- #
        for i, file in enumerate(lst_images):
            fig, ax = plt.subplots()
            ax.set_xlim(left=self.xleft, right=self.xright)
            ax.set_ylim(top=self.ytop, bottom=self.ybottom)

            # Open ABI data
            abidata = xr.open_dataset(os.path.join(self.images, self.folder, file), engine="h5netcdf")

            if self.folder != "null/":  # Null folder = no polygons to process
                interface = Setup(
                    abidata, geojsons_list[i], ax,
                    self.xf, self.xe, self.yf, self.ye, self.trim
                )

                # Compute lat/lon arrays (optional)
                if self.latlon:
                    lat, lon = interface.calc_lat_lon()
                    np.savez(file[:34] + "latlon", lat=lat, lon=lon)
                else:
                    lat, lon = 0, 0  # Dummy placeholders if not needed

                # Brightness temperature calculation
                brightness = interface.brightness_calc()
                unscaled.append(brightness)

                # Plot brightness + polygons (optional)
                if self.image:
                    interface.plot_polygons()
                    plt.contourf(
                        lon, lat, brightness,
                        levels=self.levels, cmap=self.cmap,
                        vmin=self.vmin, vmax=self.vmax
                    )

            # Save figure if plotting enabled
            if self.image:
                plt.savefig(file[:34], dpi=self.dpi)
                plt.close()

        # ---------------- Run Scaling ---------------- #
        main_scale = Scale(
            self.images, self.geojsons, self.xf, self.xe,
            self.yf, self.ye, unscaled, lst_images,
            self.trim, self.channel
        )
        main_scale.scaler_func()
