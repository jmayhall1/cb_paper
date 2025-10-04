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
# Helper Functions
# -----------------------------
def solartime(observer, sun=ephem.Sun()):
    """
    Compute local solar time from an ephem.Observer object.

    :param observer: ephem.Observer object with .date and .long set
    :param sun: ephem.Sun() object
    :return: ephem.hours object representing solar time
    """
    sun.compute(observer)
    hour_angle = observer.sidereal_time() - sun.ra
    return ephem.hours(hour_angle + ephem.hours('12:00')).norm


def filter_bins(df, col, bin_dict):
    """
    Filter dataframe by bins defined in bin_dict.

    :param df: pandas DataFrame
    :param col: column name to bin
    :param bin_dict: dict of bin_name -> (min, max)
    :return: dict of bin_name -> filtered DataFrame
    """
    return {label: df[(df[col] >= rng[0]) & (df[col] < rng[1])] for label, rng in bin_dict.items()}


def split_by_basin(bin_dict):
    """
    Split each bin dataframe into the Atlantic (AL) and Eastern Pacific (EP) basins.

    :param bin_dict: dict of bin_name -> DataFrame
    :return: tuple of dicts (AL_bins, EP_bins)
    """
    al, ep = {}, {}
    for label, df in bin_dict.items():
        df = df.copy()
        df['ID'] = df['ID'].astype(str)
        al[label] = df[df['ID'].str.startswith('AL')]
        ep[label] = df[df['ID'].str.startswith('EP')]
    return al, ep


def groupby_frequency(bin_dict, type_key):
    """
    Group by the specified type_key and Radius, summing frequency.

    :param bin_dict: dict of bin_name -> DataFrame
    :param type_key: column name to group by
    :return: dict of grouped DataFrames
    """
    return {
        label: df.groupby(by=[type_key, 'Radius'])['Frequency'].sum().reset_index(name='Frequency')
        for label, df in bin_dict.items()
    }


def normalize_frequency(grouped, grouped_total, type_key):
    """
    Normalize frequencies by total for each bin.

    :param grouped: dict of grouped DataFrames
    :param grouped_total: dict of total grouped DataFrames
    :param type_key: grouping column
    :return: dict of normalized DataFrames
    """
    result = {}
    for label in grouped:
        df = grouped[label].set_index(['Radius', df.columns[0]])
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
            type_dict_func: dict, ships_df: pd.DataFrame,
            total_dict_func: dict = None):
    """
    Bin a single TCB .npz file for frequency analysis.

    :param file: path to npz file.
    :param bin_degree_size: size of azimuth/bin in degrees.
    :param bin_rad_size: size of radial bin.
    :param i_func: index for type_dict_func.
    :param type_dict_func: dict mapping index -> column name.
    :param ships_df: DataFrame of TC ship observations.
    :param total_dict_func: dict mapping i_func -> total column name.
    :return: tuple of (binned DataFrame, binned total DataFrame).
    """
    # Extract TC metadata from filename
    if not total_dict_func:
        total_dict_func = {0: 'degreetotal', 1: 'bearingtotal', 2: 'cardinaltotal'}
    df_storm_id = file[-41:-33]
    small_id = 'AL' if 'AL' in df_storm_id else 'EP'
    date, time = file[-32:-24], file[-23:-19]
    real_time = pd.to_datetime(f'{date}{time}', format='%Y%m%d%H%M')
    storm = ships_df.loc[(ships_df.index == real_time) & (ships_df['atcf_id'] == df_storm_id)]

    intensity = storm['max_winds'].values[0]
    shear = np.sqrt(storm.shear_u ** 2 + storm.shear_v ** 2).values[0] * 0.514444
    lon = float(storm.center_lon.values[0])
    storm_time = pd.Timestamp(storm.index[0])

    # Compute local solar time
    observer = ephem.Observer()
    observer.date = storm_time.to_pydatetime()
    observer.long = lon * np.pi / 180
    solar_time = pd.Timestamp(str(solartime(observer)))
    diurnal_hour = round(solar_time.hour + solar_time.minute / 60)

    # Load TCB arrays
    storage_array = np.load(file)

    # Process primary and total frequency arrays
    def process_array(array, column: str):
        """
        Function to process arrays based on the column name.
        :param array: Array to be processed.
        :param column: Column name that is to be processed.
        :return:
        """
        df_temp = pd.DataFrame([
            (np.round(array[column] / bin_degree_size) * bin_degree_size),
            (np.round(array['radius' if 'total' not in column else 'radiustotal'] / bin_rad_size) * bin_rad_size)
        ]).T
        df_temp.columns = [column, 'Radius']
        df_temp = df_temp.astype({'Radius': 'float32', column: 'float32'})
        df_temp = df_temp.groupby(by=[column, 'Radius']).size().reset_index(name='Frequency')
        df_temp['Intensity'] = intensity
        df_temp['Shear'] = shear
        df_temp['Time'] = diurnal_hour
        df_temp['ID'] = small_id
        return df_temp

    temp_data_func = process_array(storage_array, type_dict_func[i_func])
    temp_data_total = process_array(storage_array, total_dict_func[i_func])

    return temp_data_func, temp_data_total


def process_and_plot(df, df_total, type_key):
    """
    Process binned data and generate plots for intensity, shear, and time.
    """
    print('Processing plots...')
    # Define bins
    intensity_bins = {'Weak': (0, 64), 'Moderate': (64, 95), 'Strong': (95, np.inf)}
    shear_bins = {'Low': (0, 5), 'Moderate': (5, 11), 'High': (11, np.inf)}
    time_bins = {'Night': (0, 6), 'Morning': (6, 12), 'Afternoon': (12, 18), 'Evening': (18, 24)}

    # Helper to plot each category
    def plot_category(df: pd.DataFrame, df_total: pd.DataFrame, bins: dict, titles: dict, rows: int, cols: int):
        """
        Function to plot the data for each category.
        :param df: Dataframe based on TCB pixels.
        :param df_total: Dataframe based on total pixels.
        :param bins: List of bins to be plotted.
        :param titles: List of subplot titles.
        :param rows: Number of subplot rows.
        :param cols: Number of subplot columns.
        """
        binned = filter_bins(df, bins['column'], bins['dict'])
        total_binned = filter_bins(df_total, bins['column'], bins['dict'])
        al, ep = split_by_basin(binned)
        al_total, ep_total = split_by_basin(total_binned)
        al_grouped = normalize_frequency(groupby_frequency(al, type_key),
                                         groupby_frequency(al_total, type_key), type_key)
        ep_grouped = normalize_frequency(groupby_frequency(ep, type_key),
                                         groupby_frequency(ep_total, type_key), type_key)
        interface = Plotting(list(al_grouped.values()) + list(ep_grouped.values()), type_key, titles, rows, cols)
        interface.main_plot()
        gc.collect()

    # Intensity plots
    plot_category(df, df_total,
                  bins={'column': 'Intensity', 'dict': intensity_bins},
                  titles={0: 'Atlantic TD-TS', 1: 'Atlantic CAT 1-2', 2: 'Atlantic CAT 3-5',
                          3: 'Eastern Pacific TD-TS', 4: 'Eastern Pacific CAT 1-2', 5: 'Eastern Pacific CAT 3-5'},
                  rows=2, cols=3)
    # Shear plots
    plot_category(df, df_total,
                  bins={'column': 'Shear', 'dict': shear_bins},
                  titles={0: r'Atlantic <5 $\frac{m}{s}$ Shear', 1: r'Atlantic 5-10 $\frac{m}{s}$ Shear',
                          2: r'Atlantic >10 $\frac{m}{s}$ Shear', 3: r'Eastern Pacific <5 $\frac{m}{s}$ Shear',
                          4: r'Eastern Pacific 5-10 $\frac{m}{s}$ Shear',
                          5: r'Eastern Pacific >10 $\frac{m}{s}$ Shear'}, rows=2, cols=3)
    # Time plots
    plot_category(df, df_total,
                  bins={'column': 'Time', 'dict': time_bins},
                  titles={0: 'Atlantic 0-6 LST', 1: 'Atlantic 6-12 LST', 2: 'Atlantic 12-18 LST',
                          3: 'Atlantic 18-24 LST', 4: 'Eastern Pacific 0-6 LST', 5: 'Eastern Pacific 6-12 LST',
                          6: 'Eastern Pacific 12-18 LST', 7: 'Eastern Pacific 18-24 LST'}, rows=2, cols=4)


def mp_binning(args):
    """Wrapper for multiprocessing."""
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

    # Only process azimuth for now
    for i in [0, 1, 2]:
        print(f'Creating {type_dict[i]} Plot')
        bin_mp_args = [(file, bin_degree_size, bin_rad_size, i, type_dict, ships_df) for file in file_list]
        with Pool(processes=64) as pool:
            results = list(pool.imap_unordered(mp_binning, bin_mp_args))

        # Filter out None results
        results = [r for r in results if r is not None]

        # Separate into primary and total dataframes
        data, data_total = zip(*results)
        data_df = pd.concat(data, ignore_index=True)
        data_total_df = pd.concat(data_total, ignore_index=True)

        process_and_plot(data_df, data_total_df, type_dict[i])
        del results, data_df, data_total_df, data, data_total
        gc.collect()
