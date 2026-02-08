# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall

Determines transverse band existence in each quadrant of a tropical cyclone
based on the relative shear vector.
"""
import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
from shear_multi import mp_running
from wind_calc import Wind


def make_class_count_df() -> pd.DataFrame:
    """
    Creates an empty DataFrame for quadrant classification counts.
    Columns track presence/absence of bands and storm classification.
    """
    cols = [True, False, None, 'True Storms', 'False Storms', 'None Storms']
    return pd.DataFrame({col: [0] for col in cols}, columns=cols, index=[0])


if __name__ == '__main__':
    # ---------------------------
    # Thresholds and Flags
    # ---------------------------
    cut_off = 0.02
    pix_threshold = min_wind_threshold = -np.inf
    max_wind_threshold = np.inf
    online, save_state = True, False

    # ---------------------------
    # Directory setup
    # ---------------------------
    base_dir = '/rstor/jmayhall/' if online else '//uahdata/rstor/'

    paths = {
        'latlon': f'{base_dir}cataloging/nc_process/geojson_transform/completed_arrays/latlon_arrs/*.npz',
        'hurdat': f'{base_dir}cataloging/hurdat_update.txt',
        'indir': f'{base_dir}cataloging/nc_process/shear_process/shear_process_all/',
        'pf': 'ships_interp.txt',
        'model': f'{base_dir}Model_Training_Code/cnn_creation',
        'c8': f'{base_dir}cataloging/nc_process/geojson_transform/completed_arrays/C08_scaled/*',
        'c13': f'{base_dir}cataloging/nc_process/geojson_transform/completed_arrays/C13_scaled/*',
        'ships': f'{base_dir}cataloging/nc_process/shear_process/shear_process_all/ships_interp.txt'
    }

    # ---------------------------
    # Load data once
    # ---------------------------
    hurdat_df = pd.read_csv(paths['hurdat'], sep='\t')
    latlon_arrays = pd.DataFrame({'Name': glob.glob(paths['latlon'])})
    c8_scaled = pd.DataFrame({'Name': glob.glob(paths['c8'])})
    c13_scaled = glob.glob(paths['c13'])
    c13_length = len(c13_scaled)

    # ---------------------------
    # Initialize storage and counters
    # ---------------------------
    tc_des, tc_date, tc_loc, tc_pressure = [], [], [], []
    ur_occ = ul_occ = dr_occ = dl_occ = []
    diurnal_lst_total = [0] * 24
    upshear_right = upshear_left = downshear_right = downshear_left = 0
    path_len = len(paths['c13']) - 1

    # ---------------------------
    # Time dictionary
    # ---------------------------
    time_dict = {h: str(h) if h != 0 else 'zero' for h in range(25)}
    time_dict.update({1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
                      7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve',
                      13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen',
                      17: 'seventeen', 18: 'eighteen', 19: 'nineteen', 20: 'twenty',
                      21: 'twentyone', 22: 'twentytwo', 23: 'twentythree'})
    time_dict_df = pd.DataFrame({str(h): 0 for h in range(24)}, index=[0])

    # ---------------------------
    # Shear quadrant mapping
    # ---------------------------
    shear_dict = {
        **dict.fromkeys(range(0, 90), 'd_r'),
        0: ['d_r', 'd_l'], 90: ['d_r', 'u_r'],
        **dict.fromkeys(range(91, 180), 'u_r'),
        180: ['u_r', 'u_l'],
        **dict.fromkeys(range(181, 270), 'u_l'),
        270: ['u_l', 'd_l'],
        **dict.fromkeys(range(271, 360), 'd_l'),
        360: ['d_r', 'd_l']
    }

    # ---------------------------
    # Wind change and strength DataFrames
    # ---------------------------
    wind_change_df = pd.DataFrame({label: 0 for label in
                                   ['<=-10', '-5', 'Constant', '5', '>=10', 'None',
                                    '<=-10 Count', '-5 Count', 'Constant Count', '5 Count',
                                    '>=10 Count', 'None Count']},
                                  index=range(8))

    wind_dict = pd.read_csv(paths['ships'], sep='\t', index_col=0)

    strength_df = pd.DataFrame({
        'Count': np.zeros(34),
        'Storm Count': np.zeros(34)
    }, index=np.arange(5, 171, 5))

    # ---------------------------
    # RI and RW counts
    # ---------------------------
    ri_df_prev_count = make_class_count_df()
    rw_df_prev_count = make_class_count_df()
    ri_df_next_count = make_class_count_df()
    rw_df_next_count = make_class_count_df()

    # ---------------------------
    # Prepare multiprocessing arguments
    # ---------------------------
    needed_args = []
    for i, file in enumerate(c13_scaled):
        print(f'Preparing args for file {i + 1} of {c13_length}')
        needed_args.append({
            'i': i,
            'c13_length': c13_length,
            'file': file,
            'path_len': path_len,
            'paths': paths,
            'c8_scaled': c8_scaled,
            'time_dict': time_dict,
            'shear_dict': shear_dict,
            'cut_off': cut_off,
            'base_dir': base_dir
        })

    # ---------------------------
    # Run multiprocessing
    # ---------------------------
    NUM_WORKERS = 48
    with Pool(NUM_WORKERS) as pool:
        pool.map(mp_running, needed_args)
