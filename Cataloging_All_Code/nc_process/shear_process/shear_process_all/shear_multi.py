# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall

Optimized and memory-efficient multiprocessing workflow for TCB shear calculation.
"""
import glob
import os
from datetime import datetime
from multiprocessing import Lock

import ephem
import numpy as np
import pandas as pd
from keras.models import load_model
from shear_calc import Shear
from shear_setup import Setup
from wind_calc import Wind


def solartime(observer, sun=ephem.Sun()):
    """
    Calculate Local Solar Time (LST) based on observer location.

    :param observer: ephem.Observer object
    :param sun: ephem.Sun object (default)
    :return: ephem.hours object representing LST
    """
    sun.compute(observer)
    hour_angle = observer.sidereal_time() - sun.ra
    # Add 12 hours to convert to local solar noon and normalize to 24h
    return ephem.hours(hour_angle + ephem.hours('12:00')).norm


def model_prediction(model_path: str, x_test: np.ndarray, x_test8: np.ndarray) -> np.ndarray:
    """
    Generate CNN predictions for given test arrays.

    :param model_path: Path to folder containing 'model.keras'
    :param x_test: Channel 13 data
    :param x_test8: Channel 8 data
    :return: Prediction array
    """
    model_file = os.path.join(model_path, 'model.keras')
    model = load_model(model_file, compile=False)
    return model.predict([x_test[None, :], x_test8[None, :]])


def running(i: int, c13_length: int, file: str, path_len: int, paths: dict, c8_scaled: pd.DataFrame, time_dict: dict,
            shear_dict: dict, cut_off: int, base_dir: str):
    """
    Process a single storm file for shear calculation (multiprocessing-friendly).

    :param i: File index.
    :param c13_length: Total number of files.
    :param file: Path to C13 file.
    :param path_len: Offset for extracting storm info from filename.
    :param paths: Dictionary containing file paths for model, ships, lat/lon.
    :param c8_scaled: DataFrame of scaled channel 8 files.
    :param time_dict: Dictionary for time indexing.
    :param shear_dict: Dictionary for shear quadrant.
    :param cut_off: Plotting cutoff.
    :param base_dir: Base directory for saving results.
    """
    lock = Lock()
    try:
        print(f'Processing file {i + 1} of {c13_length}')

        # Extract storm ID, date, and time from filename
        df_storm_id = file[path_len:path_len + 8]
        date = file[path_len + 9:path_len + 17]
        time = file[path_len + 18:path_len + 22]
        real_time = pd.to_datetime(f'{date}{time}', format='%Y%m%d%H%M')

        # Filter SHIPS data once using mask
        wind_df = pd.read_csv(paths['ships'], sep='\t', index_col=0)
        mask = (wind_df.index == str(real_time)) & (wind_df.atcf_id == df_storm_id)
        storm = wind_df.loc[mask]

        if storm.empty:
            print(f'No SHIPS data for {df_storm_id} at {real_time}')
            return

        center_lon = float(storm.center_lon.values[0])
        center_lat = float(storm.center_lat.values[0])

        # Skip storms outside bounds
        if ('AL' in df_storm_id and not (-105 < center_lon < -20)) or \
                ('EP' in df_storm_id and not (-140 < center_lon < -90)) or \
                not (5 < center_lat < 30):
            print(f'{df_storm_id} outside basin bounds, skipping.')
            return

        identifier = file[path_len:path_len + 22]

        # Load lat/lon arrays
        latlon_files = glob.glob(paths['latlon'])
        latlon_match = [f for f in latlon_files if identifier in f][0]
        with np.load(latlon_match) as data:
            lat = data["lat"]
            lon = data["lon"]

        # Load brightness data
        with np.load(file) as data:
            x_test = data["brightness"]
        c8_file = c8_scaled.loc[c8_scaled['Name'].str.contains(identifier), 'Name'].values[0]
        with np.load(str(c8_file)) as data:
            x_test8 = data["brightness"]

        # Generate prediction with thread-safe lock
        with lock:
            predict = None
            while predict is None:
                try:
                    predict = model_prediction(paths['model'], x_test, x_test8)
                except Exception as e:
                    print(f'Model load error: {e}')
                    predict = None

        # Initialize storm tracking
        interface1 = Setup(paths.get('indir') + paths.get('pf'), df_storm_id, lat, lon, center_lat, center_lon)
        azimuth, radius = interface1.polar_gridder()

        # Compute diurnal time
        storm_time = pd.Timestamp(storm.index.values[0])
        o = ephem.Observer()
        o.date = datetime(storm_time.year, storm_time.month, storm_time.day,
                          storm_time.hour, storm_time.minute, storm_time.second)
        o.long = center_lon * np.pi / 180
        diurnal_temp_time = round(pd.Timestamp(str(solartime(o))).hour +
                                  pd.Timestamp(str(solartime(o))).minute / 60) % 24

        # Relative shear calculation
        interface2 = Shear(azimuth, storm.shear_dir.values[0], storm.cardinal_dir.values[0],
                           None, None, None, None, None, df_storm_id, radius, diurnal_temp_time,
                           center_lon, time_dict, shear_dict)
        azi_shear_rel, bearing_rel, card_rel = interface2.vector_relative_azi()

        interface2 = Shear(None, None, None, azi_shear_rel, bearing_rel, card_rel,
                           cut_off, predict[0, :, :, 0], df_storm_id, radius, diurnal_temp_time,
                           center_lon, time_dict, shear_dict)

        (value_list, radius_arr, degree_arr, diurnal_lst, occ_list, ri_pixel_count,
         strength_pixel_count, wind_dif_pixel_count, storm_motion_rel, card_motion_rel,
         rad_total, degree_total, bear_total, card_total) = interface2.quad_probs()

        # Save compressed numpy arrays
        out_file = f'{base_dir}/cataloging/nc_process/shear_process/shear_process_all/' \
                   f'{df_storm_id}_{date}_{time}_shearandradius.npz'
        with lock:
            np.savez_compressed(out_file,
                                radius=radius_arr,
                                degree=degree_arr,
                                azimuth=card_motion_rel,
                                stormrel=storm_motion_rel,
                                radiustotal=rad_total,
                                degreetotal=degree_total,
                                bearingtotal=bear_total,
                                cardinaltotal=card_total)

    except (IndexError, AttributeError) as e:
        print(f'Error processing {file}: {e}')


def mp_running(kwargs):
    """
    Wrapper for multiprocessing with keyword arguments.
    """
    return running(**kwargs)
