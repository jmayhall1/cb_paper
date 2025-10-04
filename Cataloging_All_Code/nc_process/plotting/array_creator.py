# coding=utf-8
"""
Optimized & Commented: 10/02/2025
@author: John Mark Mayhall

Purpose:
    This script manages scaling of GOES ABI brightness temperature data.
      1. It inds any new .nc files (channels C13 & C08) that haven't yet been scaled.
      2. It prepares arguments for processing (including scaler paths and save dirs).
      3. It runs the conversion/scaling in parallel using multiprocessing.

The heavy lifting (conversion, scaling, plotting) is done in
`mp_cmi_nc_to_scaled_npz` imported from `nc_geojson_plotting`.
"""

from multiprocessing import Pool
from pathlib import Path

from nc_geojson_plotting import mp_cmi_nc_to_scaled_npz

# -------------------
# Paths & Configuration
# -------------------


if __name__ == '__main__':
    source_nc_dir = Path("/rstor/jmayhall/cataloging")
    scale_dir = source_nc_dir / "nc_process" / "plotting"

    c13_scaled_dir = scale_dir / "C13_scaled"
    c08_scaled_dir = scale_dir / "C08_scaled"
    c13_unscaled_dir = scale_dir / "C13_unscaled"

    c13_scaler_path = Path("data_scaler13.save")
    c08_scaler_path = Path("data_scaler8.save")

    num_workers = 128  # number of multiprocessing workers

    # -------------------
    # Identify already scaled files
    # -------------------
    done_13 = {s.stem for s in c13_scaled_dir.glob("*.npz")}
    done_8 = {s.stem for s in c08_scaled_dir.glob("*.npz")}

    # -------------------
    # Identify new .nc files to process
    # -------------------
    images = list(source_nc_dir.glob("*.nc"))
    images_13 = [p for p in images if "C13" in p.name]
    images_8 = [p for p in images if "C08" in p.name]

    not_done_13 = [p for p in images_13 if p.stem not in done_13]
    not_done_8 = [p for p in images_8 if p.stem not in done_8]

    print("Unprocessed C13 files:", not_done_13)
    print("Unprocessed C08 files:", not_done_8)
    print(f"Total C13 files to be scaled: {len(not_done_13)}")
    print(f"Total C08 files to be scaled: {len(not_done_8)}")

    # -------------------
    # Build argument dictionaries for processing
    # -------------------
    c13_args = [{
        "cmi_nc": p,
        "scaled_npz_dir": c13_scaled_dir,
        "scaler_path": c13_scaler_path,
        "unscaled_npz_dir": c13_unscaled_dir,
    } for p in not_done_13]

    c08_args = [{
        "cmi_nc": p,
        "scaled_npz_dir": c08_scaled_dir,
        "scaler_path": c08_scaler_path,
        "unscaled_npz_dir": None,  # channel 8 doesn’t use unscaled npz
    } for p in not_done_8]

    # -------------------
    # Run processing in parallel
    # -------------------
    if not (not_done_13 or not_done_8):
        print("No new files to process.")
    else:
        with Pool(num_workers) as pool:
            pool.map(mp_cmi_nc_to_scaled_npz, c13_args + c08_args)
