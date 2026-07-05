# coding=utf-8
"""
Optimized: 10/02/2025
Updated: adds proper file-SHIPS matching + median lines

Processes tropical cyclone shear data:
1. Loads shear NPZ files and SHIPS dataset
2. Matches only valid file-SHIPS pairs (robust merge approach)
3. Computes shear magnitude
4. Splits by basin
5. Plots distributions with medians
"""

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    ships_df = pd.read_csv(ships_path, sep='\t', index_col=0)

    # convert SHIPS index to datetime explicitly
    ships_df.index = pd.to_datetime(ships_df.index)

    # -----------------------------
    # Build valid file list (QC step)
    # -----------------------------
    valid_cases = []

    for file in file_list:
        atcf_id = file[-41:-33]
        date = file[-32:-24]
        time = file[-23:-19]

        valid_cases.append(
            (
                atcf_id,
                pd.to_datetime(f'{date}{time}', format='%Y%m%d%H%M')
            )
        )

    valid_cases = pd.DataFrame(valid_cases, columns=['atcf_id', 'time'])
    valid_cases['time'] = pd.to_datetime(valid_cases['time'])

    # -----------------------------
    # Match SHIPS + NPZ (IMPORTANT FIX)
    # -----------------------------
    ships = (
        ships_df.reset_index()
        .rename(columns={'index': 'time'})
        .merge(valid_cases, on=['time', 'atcf_id'], how='inner')
    )

    # -----------------------------
    # Compute shear magnitude
    # -----------------------------
    ships['shear'] = np.sqrt(
        ships['shear_u'] ** 2 + ships['shear_v'] ** 2
    ) * 0.514444

    # Optional QC filter (keep if desired)
    ships = ships.dropna(subset=['shear'])

    # -----------------------------
    # Split by basin
    # -----------------------------
    atl = ships[ships['atcf_id'].str.startswith('AL')]
    ep = ships[ships['atcf_id'].str.startswith('EP')]

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(16, 8))

    bins = np.arange(0, 36, 2.5)

    # Histograms
    ax.hist(atl['shear'], bins=bins,
            color='blue', alpha=0.5, label='Atlantic')

    ax.hist(ep['shear'], bins=bins,
            color='green', alpha=0.5, label='Eastern Pacific')

    # -----------------------------
    # MEDIANS (NEW)
    # -----------------------------
    al_median = np.nanmedian(atl['shear'])
    ep_median = np.nanmedian(ep['shear'])

    ax.axvline(al_median, color='black', linewidth=3, label='AL Median Shear')
    ax.axvline(ep_median, linestyle='--', color='black', linewidth=3, label='EP Median Shear')

    # -----------------------------
    # Formatting
    # -----------------------------
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 3000)

    ax.set_title('TC Shear Distribution', fontsize=18)
    ax.set_xlabel(r'Shear ($m s^{-1}$)', fontsize=14)
    ax.set_ylabel('# of Images', fontsize=14)

    ax.legend(prop={'size': 12})

    plt.tight_layout()
    plt.savefig('shear_distribution_all.png', dpi=300)

    print("Saved: shear_distribution_all.png")