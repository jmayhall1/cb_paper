# coding=utf-8
"""
Count unique ATCF IDs by shear and intensity bins,
separated into Atlantic (AL) and Eastern Pacific (EP).

Shear bins:
    5 m/s bins using:
        5 * np.round(shear / 5)

Intensity bins:
    10 kt bins using:
        10 * (int(max_winds) / 10)
"""

import glob
import numpy as np
import pandas as pd


if __name__ == '__main__':

    # -----------------------------
    # Configuration
    # -----------------------------
    ONLINE = False
    BASE_DIR = '/rstor/jmayhall/' if ONLINE else '//uahdata/rstor/'
    DATA_DIR = f'{BASE_DIR}cataloging/nc_process/shear_process/shear_process_all/'

    file_list = glob.glob(f'{DATA_DIR}*.npz')
    ships_path = f'{DATA_DIR}ships_interp.txt'

    # -----------------------------
    # Load SHIPS data
    # -----------------------------
    ships_df = pd.read_csv(
        ships_path,
        sep='\t',
        index_col=0
    )

    ships_df.index = pd.to_datetime(ships_df.index)

    # -----------------------------
    # Build valid file list
    # -----------------------------
    valid_cases = []

    for file in file_list:

        atcf_id = file[-41:-33]
        date = file[-32:-24]
        time = file[-23:-19]

        valid_cases.append(
            (
                atcf_id,
                pd.to_datetime(
                    f'{date}{time}',
                    format='%Y%m%d%H%M'
                )
            )
        )

    valid_cases = pd.DataFrame(
        valid_cases,
        columns=['atcf_id', 'time']
    )

    valid_cases['time'] = pd.to_datetime(valid_cases['time'])

    # -----------------------------
    # Match SHIPS + NPZ
    # -----------------------------
    ships = (
        ships_df.reset_index()
        .rename(columns={'index': 'time'})
        .merge(
            valid_cases,
            on=['time', 'atcf_id'],
            how='inner'
        )
    )

    # -----------------------------
    # Compute shear magnitude
    # -----------------------------
    # Shear is converted to m/s
    ships['shear'] = np.sqrt(
        ships['shear_u'] ** 2 +
        ships['shear_v'] ** 2
    ) * 0.514444

    # -----------------------------
    # Remove invalid data
    # -----------------------------
    ships = ships.dropna(
        subset=['shear', 'max_winds']
    )

    # -----------------------------
    # Initialize dictionaries
    # -----------------------------
    # Each bin contains a SET of ATCF IDs.
    # This means repeated observations of the
    # same storm count only once per bin.

    shear_bins_al = {}
    shear_bins_ep = {}

    intensity_bins_al = {}
    intensity_bins_ep = {}

    # -----------------------------
    # Loop through observations
    # -----------------------------
    for _, row in ships.iterrows():

        atcf_id = row['atcf_id']
        shear = row['shear']
        intensity = row['max_winds']

        # -------------------------
        # Determine basin
        # -------------------------
        if atcf_id.startswith('AL'):

            shear_bins = shear_bins_al
            intensity_bins = intensity_bins_al

        elif atcf_id.startswith('EP'):

            shear_bins = shear_bins_ep
            intensity_bins = intensity_bins_ep

        else:
            continue

        # -------------------------
        # Shear bin: nearest 5 m/s
        # -------------------------
        shear_bin = 5 * np.round(shear / 5)

        # -------------------------
        # Intensity bin: 10 kt
        # -------------------------
        intensity_bin = int(10 * (np.round(intensity / 10)))

        # -------------------------
        # Add unique ATCF ID
        # -------------------------
        shear_bins.setdefault(
            shear_bin,
            set()
        ).add(atcf_id)

        intensity_bins.setdefault(
            intensity_bin,
            set()
        ).add(atcf_id)

    # ============================================================
    # PRINT SHEAR COUNTS
    # ============================================================

    print()
    print("============================================================")
    print("UNIQUE ATCF IDs BY SHEAR BIN")
    print("============================================================")
    print()
    print("                 Atlantic       Eastern Pacific")
    print("------------------------------------------------------------")

    all_shear_bins = sorted(
        set(shear_bins_al.keys()) |
        set(shear_bins_ep.keys())
    )

    for shear_bin in all_shear_bins:

        al_count = len(
            shear_bins_al.get(shear_bin, set())
        )

        ep_count = len(
            shear_bins_ep.get(shear_bin, set())
        )

        print(
            f"{shear_bin:5.1f} m/s"
            f"{al_count:15d}"
            f"{ep_count:20d}"
        )

    # ============================================================
    # PRINT INTENSITY COUNTS
    # ============================================================

    print()
    print("============================================================")
    print("UNIQUE ATCF IDs BY INTENSITY BIN")
    print("============================================================")
    print()
    print("                 Atlantic       Eastern Pacific")
    print("------------------------------------------------------------")

    all_intensity_bins = sorted(
        set(intensity_bins_al.keys()) |
        set(intensity_bins_ep.keys())
    )

    for intensity_bin in all_intensity_bins:

        al_count = len(
            intensity_bins_al.get(intensity_bin, set())
        )

        ep_count = len(
            intensity_bins_ep.get(intensity_bin, set())
        )

        print(
            f"{intensity_bin:3d} kt"
            f"{al_count:15d}"
            f"{ep_count:20d}"
        )

    print()
