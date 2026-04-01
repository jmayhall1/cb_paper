# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall

Memory-efficient code for plotting TCB data by shear, relative azimuth, and radius.
Uses multiprocessing for faster binning and grouping.
"""

import gc
import glob
from multiprocessing import Pool

import ephem
import numpy as np
import pandas as pd
from shear_plot_setup import Plotting

# -----------------------------
# GLOBAL (NEW)
# -----------------------------
GLOBAL_SHIPS_DF = None

def intensity_bin(val):
    if val < 64:
        return 0
    elif val < 95:
        return 1
    else:
        return 2

def shear_bin(val):
    if val < 5:
        return 0
    elif val < 11:
        return 1
    else:
        return 2

def time_bin(val):
    if val < 6:
        return 0
    elif val < 12:
        return 1
    elif val < 18:
        return 2
    else:
        return 3


def init_worker(ships):
    global GLOBAL_SHIPS_DF
    GLOBAL_SHIPS_DF = ships


# -----------------------------
# Helper Functions
# -----------------------------
def solartime(observer, sun=ephem.Sun()):
    sun.compute(observer)
    hour_angle = observer.sidereal_time() - sun.ra
    return ephem.hours(hour_angle + ephem.hours('12:00')).norm


def split_by_basin(bin_dict):
    al, ep = {}, {}
    for label, df in bin_dict.items():
        df.loc[:, 'ID'] = df['ID'].astype(str)  # no copy
        al[label] = df[df['ID'].str.startswith('AL')]
        ep[label] = df[df['ID'].str.startswith('EP')]
    return al, ep


def groupby_frequency(bin_dict, type_key):
    return {
        label: df.groupby(by=[type_key, 'Radius'])['Frequency'].sum().reset_index(name='Frequency')
        for label, df in bin_dict.items()
    }


def normalize_frequency(grouped, grouped_total, type_key):
    result = {}
    for label in grouped:
        df = grouped[label].set_index(['Radius', type_key])
        df_total = grouped_total[label].set_index(['Radius', type_key])

        norm_freq = df['Frequency'].divide(df_total['Frequency'], fill_value=0) * 100
        merged = norm_freq.reset_index()
        merged.columns = ['Radius', type_key, 'Frequency']
        result[label] = merged

    gc.collect()
    return result


# -----------------------------
# Binning Function
# -----------------------------
def binning(file: str, bin_degree_size: int, bin_rad_size: int, i_func: int,
            type_dict_func: dict, total_dict_func: dict = None):

    global GLOBAL_SHIPS_DF

    if not total_dict_func:
        total_dict_func = {0: 'degreetotal', 1: 'bearingtotal', 2: 'cardinaltotal'}

    df_storm_id = file[-41:-33]
    small_id = 'AL' if 'AL' in df_storm_id else 'EP'
    date, time = file[-32:-24], file[-23:-19]
    real_time = pd.to_datetime(f'{date}{time}', format='%Y%m%d%H%M')

    storm = GLOBAL_SHIPS_DF.loc[
        (GLOBAL_SHIPS_DF.index == real_time) &
        (GLOBAL_SHIPS_DF['atcf_id'] == df_storm_id)
    ]

    intensity = storm['max_winds'].values[0]
    shear = np.sqrt(storm.shear_u ** 2 + storm.shear_v ** 2).values[0] * 0.514444
    lon = float(storm.center_lon.values[0])
    storm_time = pd.Timestamp(storm.index[0])

    observer = ephem.Observer()
    observer.date = storm_time.to_pydatetime()
    observer.long = lon * np.pi / 180
    solar_time = pd.Timestamp(str(solartime(observer)))
    diurnal_hour = round(solar_time.hour + solar_time.minute / 60)

    # MEMORY FIX: mmap
    storage_array = np.load(file, mmap_mode='r')

    def process_array(array, column: str):
        # Select correct radius key
        rad_key = 'radius' if 'total' not in column else 'radiustotal'

        # Raw values
        theta = array[column]
        radius = array[rad_key]

        # Define bin edges (match your rounding logic)
        theta_bins = np.arange(0, 360 + bin_degree_size, bin_degree_size)
        radius_bins = np.arange(0, 1024 + bin_rad_size, bin_rad_size)

        # Compute 2D histogram
        hist, theta_edges, radius_edges = np.histogram2d(
            theta,
            radius,
            bins=[theta_bins, radius_bins]
        )

        # Convert to coordinates (bin centers)
        theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
        radius_centers = (radius_edges[:-1] + radius_edges[1:]) / 2

        # Flatten grid
        theta_grid, radius_grid = np.meshgrid(theta_centers, radius_centers, indexing='ij')

        df_temp = pd.DataFrame({
            column: theta_grid.ravel().astype('float32'),
            'Radius': radius_grid.ravel().astype('float32'),
            'Frequency': hist.ravel().astype('float32')
        })

        # Remove zero-frequency bins (huge memory saver)
        df_temp = df_temp[df_temp['Frequency'] > 0]

        # Add metadata
        df_temp['Intensity'] = intensity
        df_temp['Shear'] = shear
        df_temp['Time'] = diurnal_hour
        df_temp['ID'] = small_id

        return df_temp

    temp_data_func = process_array(storage_array, type_dict_func[i_func])
    temp_data_total = process_array(storage_array, total_dict_func[i_func])

    del storage_array

    return temp_data_func, temp_data_total


def process_and_plot(df, df_total, type_key):
    print('Processing plots...')

    intensity_bins = {'Weak': (0, 64), 'Moderate': (64, 95), 'Strong': (95, np.inf)}
    shear_bins = {'Low': (0, 5), 'Moderate': (5, 11), 'High': (11, np.inf)}
    time_bins = {'Night': (0, 6), 'Morning': (6, 12), 'Afternoon': (12, 18), 'Evening': (18, 24)}

    def plot_category(df, df_total, bins, titles, rows, cols):
        binned = {label: df[df[bins['column']] == label] for label in bins['dict']}
        total_binned = {label: df_total[df_total[bins['column']] == label] for label in bins['dict']}

        al, ep = split_by_basin(binned)
        al_total, ep_total = split_by_basin(total_binned)

        al_grouped = normalize_frequency(groupby_frequency(al, type_key),
                                         groupby_frequency(al_total, type_key), type_key)
        ep_grouped = normalize_frequency(groupby_frequency(ep, type_key),
                                         groupby_frequency(ep_total, type_key), type_key)

        interface = Plotting(list(al_grouped.values()) + list(ep_grouped.values()),
                             type_key, titles, rows, cols)
        interface.main_plot()
        gc.collect()

    # Intensity
    plot_category(df, df_total,
                  bins={'column': 'Intensity', 'dict': intensity_bins},
                  titles={0: 'Atlantic TD-TS', 1: 'Atlantic CAT 1-2', 2: 'Atlantic CAT 3-5',
                          3: 'Eastern Pacific TD-TS', 4: 'Eastern Pacific CAT 1-2', 5: 'Eastern Pacific CAT 3-5'},
                  rows=2, cols=3)

    # Shear
    plot_category(df, df_total,
                  bins={'column': 'Shear', 'dict': shear_bins},
                  titles={0: r'Atlantic <5 $\frac{m}{s}$ Shear', 1: r'Atlantic 5-10 $\frac{m}{s}$ Shear',
                          2: r'Atlantic >10 $\frac{m}{s}$ Shear', 3: r'Eastern Pacific <5 $\frac{m}{s}$ Shear',
                          4: r'Eastern Pacific 5-10 $\frac{m}{s}$ Shear',
                          5: r'Eastern Pacific >10 $\frac{m}{s}$ Shear'},
                  rows=2, cols=3)

    # Time
    plot_category(df, df_total,
                  bins={'column': 'Time', 'dict': time_bins},
                  titles={0: 'Atlantic 0-6 LST', 1: 'Atlantic 6-12 LST', 2: 'Atlantic 12-18 LST',
                          3: 'Atlantic 18-24 LST', 4: 'Eastern Pacific 0-6 LST', 5: 'Eastern Pacific 6-12 LST',
                          6: 'Eastern Pacific 12-18 LST', 7: 'Eastern Pacific 18-24 LST'},
                  rows=2, cols=4)


def mp_binning(args):
    return binning(*args)


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == '__main__':
    online = True
    base_path = '/rstor/jmayhall/' if online else '//uahdata/rstor/'

    file_list = glob.glob(f'{base_path}cataloging/nc_process/shear_process/shear_process_all/*.npz')
    print(f"Total files: {len(file_list)}")

    bin_degree_size = 1.0
    bin_rad_size = 25.0
    type_dict = {0: 'degree', 1: 'stormrel', 2: 'azimuth'}

    ships_path = f'{base_path}cataloging/nc_process/violin_plots/shear_process_all/ships_interp.txt'
    ships_df = pd.read_csv(ships_path, sep='\t', index_col=0, parse_dates=True)

    for i in [0, 1, 2]:
        print(f'Creating {type_dict[i]} Plot')

        bin_mp_args = [
            (file, bin_degree_size, bin_rad_size, i, type_dict)
            for file in file_list
        ]
        total_files = len(bin_mp_args)
        completed = 0
        from collections import defaultdict

        agg = defaultdict(float)
        agg_total = defaultdict(float)

        with Pool(processes=32, initializer=init_worker, initargs=(ships_df,)) as pool:
            for r in pool.imap_unordered(mp_binning, bin_mp_args):
                completed += 1
                print(f'Completed {completed}/{total_files}')

                if r is None:
                    continue

                d, dt = r

                angle_col = type_dict[i]

                # Column indices (safe + fast)
                col_idx = {col: idx for idx, col in enumerate(d.columns)}
                angle_idx = col_idx[angle_col]
                rad_idx = col_idx['Radius']
                int_idx = col_idx['Intensity']
                shear_idx = col_idx['Shear']
                time_idx = col_idx['Time']
                id_idx = col_idx['ID']
                freq_idx = col_idx['Frequency']

                # ----------- FUNC DATA -----------
                for row in d.itertuples(index=False, name=None):
                    key = (
                        row[angle_idx],
                        row[rad_idx],
                        intensity_bin(row[int_idx]),
                        shear_bin(row[shear_idx]),
                        time_bin(row[time_idx]),
                        row[id_idx]
                    )
                    agg[key] += row[freq_idx]

                # ----------- TOTAL DATA -----------
                for row in dt.itertuples(index=False, name=None):
                    key = (
                        row[angle_idx],
                        row[rad_idx],
                        intensity_bin(row[int_idx]),
                        shear_bin(row[shear_idx]),
                        time_bin(row[time_idx]),
                        row[id_idx]
                    )
                    agg_total[key] += row[freq_idx]

        # Concatenate only after all valid results are collected
        cols = [type_dict[i], 'Radius', 'Intensity', 'Shear', 'Time', 'ID', 'Frequency']

        data_df = pd.DataFrame(
            [(k[0], k[1], k[2], k[3], k[4], k[5], v) for k, v in agg.items()],
            columns=cols
        )

        data_total_df = pd.DataFrame(
            [(k[0], k[1], k[2], k[3], k[4], k[5], v) for k, v in agg_total.items()],
            columns=cols
        )

        intensity_map = {0: 'Weak', 1: 'Moderate', 2: 'Strong'}
        shear_map = {0: 'Low', 1: 'Moderate', 2: 'High'}
        time_map = {0: 'Night', 1: 'Morning', 2: 'Afternoon', 3: 'Evening'}

        for df in [data_df, data_total_df]:
            df['Intensity'] = df['Intensity'].map(intensity_map)
            df['Shear'] = df['Shear'].map(shear_map)
            df['Time'] = df['Time'].map(time_map)

        data_total_df = data_total_df.rename(columns={
            'degreetotal': 'degree',
            'bearingtotal': 'stormrel',
            'cardinaltotal': 'azimuth'
        })

        process_and_plot(data_df, data_total_df, type_dict[i])

        del data_df, data_total_df
        gc.collect()