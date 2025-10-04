# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall

Purpose:
    Convert GOES ABI fixed grid projection (x,y) coordinates into
    latitude/longitude arrays for given netCDF ABI CMI files and
    save the lat/lon arrays into compressed .npz files.
"""

from pathlib import Path

import numpy as np
import xarray as xr


# -------------------------
# Core conversion function
# -------------------------
def abi_to_latlon(nc_path: str, out_path: str) -> None:
    """
    Convert ABI fixed grid projection (x,y) to lat/lon and save to .npz.

    :param nc_path: Path to ABI CMI netCDF file
    :param out_path: Output path (without extension, .npz will be added)
    """
    ds = xr.open_dataset(nc_path)

    # Extract grid coordinates
    x, y = np.meshgrid(ds.x.values, ds.y.values)
    proj = ds.goes_imager_projection

    # Projection constants
    major, minor = float(proj.semi_major_axis), float(proj.semi_minor_axis)
    h = float(proj.perspective_point_height) + major
    lambda_0 = np.radians(float(proj.longitude_of_projection_origin))

    # ABI projection equations (see GOES-R PUG, Volume 3)
    a = (np.sin(x)) ** 2 + (np.cos(x)) ** 2 * ((np.cos(y)) ** 2 + (major ** 2 / minor ** 2) * (np.sin(y)) ** 2)
    b = -2 * h * np.cos(x) * np.cos(y)
    c = h ** 2 - major ** 2

    # Distance from satellite to point
    r_s = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    # Satellite projection coordinates
    s_x = r_s * np.cos(x) * np.cos(y)
    s_y = -r_s * np.sin(x)
    s_z = r_s * np.cos(x) * np.sin(y)

    # Geodetic lat/lon
    lat = np.degrees(
        np.arctan((major ** 2 / minor ** 2) * (s_z / np.sqrt((h - s_x) ** 2 + s_y ** 2)))
    )
    lon = np.degrees(lambda_0 - np.arctan(s_y / (h - s_x)))

    # Save compressed .npz
    out_file = Path(out_path).with_suffix(".npz")
    np.savez_compressed(out_file, lat=lat, lon=lon)
    print(f"Saved lat/lon arrays → {out_file}")


# -------------------------
# Run conversions
# -------------------------
if __name__ == "__main__":
    files_to_convert = [
        ("//uahdata/rstor/cataloging/AL012022_20220602_1800_C13.nc", "AL_latlon"),
        ("//uahdata/rstor/cataloging/EP012020_20200427_1300_C13.nc", "EP_latlon_17"),
        ("//uahdata/rstor/cataloging/EP192023_20231107_1200_C13.nc", "EP_latlon_18"),
    ]

    for nc_file, out_name in files_to_convert:
        abi_to_latlon(nc_file, out_name)
