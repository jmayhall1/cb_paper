# coding=utf-8
"""
Optimized & Commented: 10/02/2025
@author: John Mark Mayhall

Purpose:
    Provides functions to:
      - Open ABI CMI NetCDF files (convert Kelvin → Celsius).
      - Optionally save raw/unscaled brightness temp arrays.
      - Apply sklearn StandardScaler transformations.
      - Save the scaled brightness temps as compressed NPZ.

Includes a multiprocessing-friendly wrapper.
"""

from multiprocessing import Lock
from pathlib import Path

import joblib
import numpy as np
import xarray as xr


# -------------------------
# Helper: open ABI netCDF
# -------------------------
def open_func(f: Path) -> np.ndarray:
    """
    Open a GOES ABI netCDF file and extract brightness temp in Celsius.

    :param f: Path to a netCDF file
    :return: (Y,X) numpy array of CMI values (float32, Celsius)
    """
    arr = xr.open_dataset(f.as_posix(), engine="h5netcdf").CMI.values
    return arr.astype(np.float32) - 273.15  # Kelvin → Celsius


# -------------------------
# Main conversion function
# -------------------------
def cmi_nc_to_scaled_npz(cmi_nc: Path, scaled_npz_dir: Path, scaler_path: Path,
                         unscaled_npz_dir: Path | None = None) -> str | None:
    """
    Open a single ABI CMI file, rescale using a provided StandardScaler,
    and save the result as a compressed .npz file.

    :param cmi_nc: ABI CMI netCDF file path
    :param scaled_npz_dir: Directory to save the scaled arrays
    :param scaler_path: Path to a sklearn StandardScaler object (saved with joblib)
    :param unscaled_npz_dir: Optional dir for saving unscaled arrays (before scaling)
    :return: Path to the newly created scaled .npz file, or None if failed
    """
    lock = Lock()
    try:
        # Load raw brightness temps (Celsius)
        unscaled = open_func(cmi_nc)

        # Save unscaled version if requested
        if unscaled_npz_dir is not None:
            unscaled_save_path = (unscaled_npz_dir.as_posix() + str(cmi_nc).replace("/rstor/jmayhall/cataloging",
                                                                                    "").replace(".nc",
                                                                                                "_unscaled"))
            print(f"Saving unscaled → {unscaled_save_path}")
            with lock:
                np.savez_compressed(unscaled_save_path, brightness=unscaled)

        # Load scaler and apply transformation
        scaler = joblib.load(scaler_path.as_posix())
        brightness = scaler.transform(unscaled.reshape(-1, unscaled.shape[-1]))
        brightness = brightness.reshape(unscaled.shape)

        # Replace any NaN values with fallback (10)
        brightness = np.nan_to_num(brightness, copy=False, nan=10)

        # Build scaled save path
        scaled_relpath = str(cmi_nc).replace("/rstor/jmayhall/cataloging/", "").replace(".nc", "_scaled")
        scaled_path = scaled_npz_dir / scaled_relpath

        print(f"Saving scaled → {scaled_path}")
        scaled_path.parent.mkdir(parents=True, exist_ok=True)  # ensure dirs exist
        with lock:
            np.savez_compressed(scaled_path.as_posix(), brightness=brightness)

        return scaled_path.as_posix()

    except Exception as e:
        print(f"[ERROR] Skipping {cmi_nc}: {e}")
        return None


# -------------------------
# Multiprocessing wrapper
# -------------------------
def mp_cmi_nc_to_scaled_npz(kwargs: dict) -> str | None:
    """
    Thin wrapper for multiprocessing Pool.map.
    Simply unpacks kwargs for cmi_nc_to_scaled_npz.
    """
    return cmi_nc_to_scaled_npz(**kwargs)
