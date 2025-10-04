# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall

Purpose:
--------
Identify degraded GOES IR images (C08/C13 channels) using NetCDF DQF flags,
match them with associated files (NPZ/GeoJSON/polar files), and remove all
affected files if flagged as degraded.
"""

import glob
import os
from multiprocessing import Pool

import netCDF4
import pandas as pd


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================
def extract_metadata_from_filename(file: str):
    """
    Extract storm ID, date, and time from filename.
    Handles filename lengths of 29 or 30 characters.
    """
    fname = os.path.basename(file)
    s_id = fname[-29:-21] if len(fname) == 29 else fname[-30:-22]
    date = fname[-20:-12] if len(fname) == 29 else fname[-21:-13]
    time = fname[-11:-7]
    return s_id, date, time


def file_matches_ship(s_id: str, real_time: pd.Timestamp, ship_df: pd.DataFrame) -> bool:
    """
    Check if storm ID and timestamp exist in SHIPS dataset.
    """
    return not ship_df[(ship_df['atcf_id'] == s_id) & (ship_df.index == str(real_time))].empty


# ======================================================================
# PROCESSORS
# ======================================================================
def process_nc_file(file: str):
    """
    Check a NetCDF file for degraded quality using DQF flag.
    Identify corresponding AL/EP storm IDs and collect files to remove.
    """
    print('Processing', file)
    try:
        with netCDF4.Dataset(file) as ds:
            dqf_flag = ds.variables['DQF'].getncattr(
                'percent_focal_plane_temperature_threshold_exceeded_qf'
            )
    except Exception as e:
        print(f"Error reading NetCDF file {file}: {e}")
        return [], [], []

    # Skip files if no degradation
    if dqf_flag <= 0:
        return [], [], []

    print(f"Degraded file: {file}, DQF={dqf_flag}")

    # Extract metadata
    s_id, date, time = extract_metadata_from_filename(file)

    # Parse timestamp
    try:
        real_time = pd.to_datetime(f'{date}{time}', format='%Y%m%d%H%M')
    except Exception as e:
        # Attempt fallback parsing
        print(f"Potential Error {e}, reattempting with format='mixed'.")
        try:
            real_time = pd.to_datetime(f'{date}{time}', format='mixed')
        except Exception as e:
            print(f"Failed parsing time for {file}: {e}")
            return [], [], []

    al_removed, ep_removed, matched_files = [], [], []

    # Match storm ID to SHIPS database
    if 'AL' in file and file_matches_ship(s_id, real_time, al_ships):
        al_removed.append(f'{s_id}_{date}_{time}')
    elif 'EP' in file and file_matches_ship(s_id, real_time, ep_ships):
        ep_removed.append(f'{s_id}_{date}_{time}')

    # Match associated product files
    for item in total_files:
        if s_id in item and date in item and time in item:
            matched_files.append(item)

    # Add matching C08/C13 NetCDF pairs
    base_file = file.replace('C13', 'C08') if 'C13' in file else file
    matched_files += [base_file, base_file.replace('C08', 'C13')]

    return al_removed, ep_removed, matched_files


def process_image_file(file: str):
    """
    Process degraded image files (.jpg), checking against SHIPS dataset.
    Return storm IDs and associated removals.
    """
    s_id, date, time = file[-37:-29], file[-28:-20], file[-19:-15]
    real_time = pd.to_datetime(f'{date}{time}', format='%Y%m%d%H%M')

    if 'AL' in file and file_matches_ship(s_id, real_time, al_ships):
        return [f'{s_id}_{date}_{time}'], [], []
    elif 'EP' in file and file_matches_ship(s_id, real_time, ep_ships):
        return [], [f'{s_id}_{date}_{time}'], []
    else:
        return [], [], []


# ======================================================================
# MAIN EXECUTION
# ======================================================================
if __name__ == "__main__":
    # ======================================================================
    # CONFIG FLAGS
    # ======================================================================
    REMOVE_FILES = True  # If True, remove identified degraded files
    PLOT = False  # Placeholder for plotting (not currently used)

    # ======================================================================
    # FILE COLLECTION
    # ======================================================================
    # Candidate image files (original and removed images)
    images = glob.glob('/rstor/jmayhall/cataloging/nc_process/model_checker/*.jpg') + \
             glob.glob('/rstor/jmayhall/cataloging/nc_process/model_checker/removed_images/*.jpg')

    # Data products (NPZ arrays, GeoJSON labels, polar shear files)
    numpy_files = glob.glob('/rstor/jmayhall/cataloging/nc_process/geojson_transform/completed_arrays/*/*.npz')
    geo_files = glob.glob('/rstor/jmayhall/cataloging/nc_process/error_test/*.geojson')
    polar_files = glob.glob('/rstor/jmayhall/cataloging/nc_process/shear_process/shear_process_all/*.npz')
    total_files = numpy_files + geo_files + polar_files

    # Raw NetCDF files for GOES channels
    nc_files = (glob.glob('/rstor/jmayhall/cataloging/nc_file/*C08*.nc') +
                glob.glob('/rstor/jmayhall/cataloging/nc_file/*C13*.nc'))

    # SHIPS interpolated dataset for storm IDs and times
    ships = pd.read_csv('/rstor/jmayhall/cataloging/nc_process/model_checker/ships_interp.txt',
                        sep='\t', index_col=0)
    ep_ships = ships[ships['atcf_id'].str.contains('EP', na=False)]
    al_ships = ships[ships['atcf_id'].str.contains('AL', na=False)]
    ep_total, al_total = len(ep_ships), len(al_ships)
    # --- Process NetCDF files in parallel ---
    with Pool(processes=256) as pool:
        results = pool.map(process_nc_file, nc_files)

    all_al_removed, all_ep_removed, all_files_to_remove = set(), set(), set()

    # Aggregate removals
    for al_r, ep_r, rmv in results:
        all_al_removed.update(al_r)
        all_ep_removed.update(ep_r)
        all_files_to_remove.update(rmv)

    # --- Process degraded images (serially here, could be parallelized) ---
    for file in images:
        al_r, ep_r, rmv = process_image_file(file)
        all_al_removed.update(al_r)
        all_ep_removed.update(ep_r)

        # Also add related product files for this image
        for item in total_files:
            if all(part in item for part in [file[-37:-29], file[-28:-20], file[-19:-15]]):
                all_files_to_remove.add(item)

    # ==================================================================
    # SUMMARY STATISTICS
    # ==================================================================
    print("===========================================")
    print(f'EP removed: {len(all_ep_removed)} / {ep_total} ({(len(all_ep_removed) / ep_total) * 100:.2f}%)')
    print(f'AL removed: {len(all_al_removed)} / {al_total} ({(len(all_al_removed) / al_total) * 100:.2f}%)')
    all_total = ep_total + al_total
    all_removed = len(all_ep_removed) + len(all_al_removed)
    print(f'ALL removed: {all_removed} / {all_total} ({(all_removed / all_total) * 100:.2f}%)')
    print("===========================================")

    # ==================================================================
    # FILE REMOVAL
    # ==================================================================
    if REMOVE_FILES:
        pd.DataFrame(images, columns=['file']).to_csv('removed_files.txt', index=False)
        for file in all_files_to_remove:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error removing {file}: {e}")
