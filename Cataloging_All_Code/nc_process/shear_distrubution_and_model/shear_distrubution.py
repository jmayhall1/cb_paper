# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall

This script processes tropical cyclone shear data:
1. Loads shear process .npz files and the SHIPS interpolated dataset.
2. Matches each file to its corresponding SHIPS record.
3. Computes wind shear magnitude (sqrt(u^2 + v^2)).
4. Separates results by basin (Atlantic vs. Eastern Pacific).
5. Creates and saves a histogram of shear distribution.
"""

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # -----------------------------
    # Configuration
    # -----------------------------
    ONLINE = False
    BASE_DIR = '/rstor/jmayhall/' if ONLINE else '//uahdata/rstor/'
    DATA_DIR = f'{BASE_DIR}cataloging/nc_process/shear_process/shear_process_all/'

    # Input file paths
    file_list = glob.glob(f'{DATA_DIR}*.npz')  # shear files
    ships_path = f'{DATA_DIR}ships_interp.txt'  # SHIPS interpolated data

    # -----------------------------
    # Load SHIPS dataset
    # -----------------------------
    ships_df = pd.read_csv(
        ships_path, sep='\t', index_col=0
    )  # index is timestamps (string formatted)

    # -----------------------------
    # Process shear files
    # -----------------------------
    file_len = len(file_list)
    shear_list_AL, shear_list_EP = [], []

    for i, file in enumerate(file_list, start=1):
        print(f'Processing File {i} of {file_len}')

        # Extract timestamp and storm ID from file path
        # Adjust slicing carefully to match your file naming convention
        timestamp = str(pd.Timestamp(f'{file[79:87]}{file[88:92]}'))  # YYYYMMDD + HHMM
        s_id = file[70:78]  # storm ID (e.g., AL012022)

        # Match to SHIPS dataset
        time_ships = ships_df.loc[ships_df.index == timestamp]
        current_ships = time_ships[time_ships.atcf_id.str.contains(s_id)]

        if current_ships.empty:
            print(f'Warning: No SHIPS data found for {s_id} at {timestamp}')
            continue

        # Compute shear magnitude [m/s] (ships shear is in knots → convert to m/s)
        shear = np.sqrt(current_ships.shear_u ** 2 + current_ships.shear_v ** 2).values[0] * 0.514444

        # Separate by basin
        if 'AL' in s_id:
            shear_list_AL.append(shear)
        else:
            shear_list_EP.append(shear)

    # -----------------------------
    # Clean NaNs and convert to arrays
    # -----------------------------
    shear_list_AL = np.array(shear_list_AL)
    shear_list_EP = np.array(shear_list_EP)
    shear_list_AL = shear_list_AL[~np.isnan(shear_list_AL)]
    shear_list_EP = shear_list_EP[~np.isnan(shear_list_EP)]

    # -----------------------------
    # Plot shear distributions
    # -----------------------------
    fig, ax = plt.subplots(figsize=(16, 8))

    # Histogram for Atlantic
    ax.hist(shear_list_AL, bins=np.arange(0, 36, 2.5),
            color='blue', alpha=0.5, label='Atlantic')

    # Histogram for Eastern Pacific
    ax.hist(shear_list_EP, bins=np.arange(0, 36, 2.5),
            color='green', alpha=0.5, label='Eastern Pacific')

    # Axis settings
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 3000)
    ax.legend(prop={'size': 16})

    # Titles and labels
    fig.subplots_adjust(bottom=0.20)  # push plots up
    fig.suptitle('TC Shear Distribution', fontsize=20, y=0.95)
    fig.supxlabel(r'Shear ($m s^{-1}$)', fontsize=20, y=0.1)
    fig.supylabel('# of Images', fontsize=20, x=0.05)

    # Save figure
    plt.savefig('shear_distribution_all.png')
    print("Figure saved as shear_distribution_all.png")
