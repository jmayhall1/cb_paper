# coding=utf-8
"""
Last Edited: 10/02/2025
@author: John Mark Mayhall
"""
from datetime import datetime

import ephem
import numpy as np
import pandas as pd
from keras.models import load_model
from wind_calc import Wind


# ----------------------------
# Solar time utility function
# ----------------------------
def solartime(observer: ephem.Observer, sun: ephem.Body = ephem.Sun()):
    """
    Compute local solar time at an observer's location.

    :param observer: ephem.Observer object with date, latitude, longitude.
    :param sun: ephem.Sun object (default).
    :return: Local solar time as ephem.hours.
    """
    sun.compute(observer)
    hour_angle = observer.sidereal_time() - sun.ra
    return ephem.hours(hour_angle + ephem.hours('12:00')).norm


# ----------------------------
# Keras model loader for multiprocessing
# ----------------------------
model = None


def init_worker(model_path: str = '/rstor/jmayhall/Model_Training_Code/cnn_creation/model.keras'):
    """
    Initialize Keras model for each worker in multiprocessing.
    Retries until the model is successfully loaded.
    """
    global model
    while model is None:
        try:
            model = load_model(model_path, compile=False)
        except Exception as e:
            print(f'Model load error: {e}')
            model = None


def model_prediction(x_test: np.ndarray, x_test8: np.ndarray) -> np.ndarray:
    """
    Run the CNN model on CH13 and CH08 brightness arrays.

    :param x_test: CH13 array
    :param x_test8: CH08 array
    :return: Predicted output array
    """
    return model.predict([x_test[None, :], x_test8[None, :]])


# ----------------------------
# Main processing function
# ----------------------------
def running(i: int, file: str, path_len: int, paths: dict, c8_scaled: pd.DataFrame):
    """
    Process a single satellite file to compute diurnal time, wind change, and TCB pixels.

    :param i: File index for logging
    :param file: Filepath of CH13 .npz
    :param path_len: Offset to extract identifier from filename
    :param paths: Dictionary with paths (e.g., 'ships')
    :param c8_scaled: DataFrame of scaled CH08 files
    :return: List [diurnal_time, rounded_wind, wind_change_list, TCB pixel count, storm_id] or None
    """
    identifier = file[path_len: path_len + 22]
    df_storm_id = identifier[:8]
    date, time = identifier[9:17], identifier[18:22]
    print(f'Processing file {i + 1}: {identifier}')

    # Convert timestamp
    real_time = pd.to_datetime(f'{date}{time}', format='%Y%m%d%H%M')

    # Load SHIPS data and filter once for storm
    wind_dict = pd.read_csv(paths['ships'], sep='\t', index_col=0)
    storm = wind_dict.query("index == @real_time.strftime('%Y-%m-%d %H:%M:%S') and atcf_id == @df_storm_id")
    if storm.empty:
        print(f'No SHIPS data for {df_storm_id} at {real_time}')
        return None

    # Extract storm center
    center_lat = storm.iloc[0].center_lat
    center_lon = storm.iloc[0].center_lon

    # Filter by basin and bounds
    if ('AL' in df_storm_id and not (-105 < center_lon < -20)) or \
            ('EP' in df_storm_id and not (-140 < center_lon < -90)) or \
            not (5 < center_lat < 30):
        print('Outside of bounds, skipping.')
        return None

    # Max wind
    wind = storm['max_winds'].values[0] if not storm.empty else None

    # Compute wind change
    wind_calc = Wind(df_storm_id, real_time, wind_dict)
    wind_dif_list = wind_calc.wind_change_calc()

    # Load CH13 brightness
    with np.load(file) as data:
        x_test = data["brightness"]

    # Match CH08 file
    c8_match = c8_scaled.loc[c8_scaled['Name'].str.contains(identifier), 'Name']
    if c8_match.empty:
        print(f'C08 file not found for {identifier}')
        return None

    with np.load(str(c8_match.values[0])) as data:
        x_test8 = data["brightness"]

    # Run prediction (retry until successful)
    predict = None
    while predict is None:
        try:
            predict = model_prediction(x_test, x_test8)
            # Optional: log unusually high predictions
            high_frac = (np.nansum(predict > 0.02) / (1024 * 1024)) * 100
            if high_frac > 60:
                print(f'High prediction {high_frac:.2f}% at {file}')
        except Exception as e:
            print(f'Model prediction error: {e}')
            predict = None

    # Compute Local Solar Time and round to 3-hour bin
    o = ephem.Observer()
    storm_time = pd.Timestamp(storm.index[0])
    o.date = datetime(*storm_time.timetuple()[:6])
    o.lon = np.deg2rad(center_lon)
    solar_time_hours = solartime(o).tuple()[3] + solartime(o).tuple()[4] / 60
    diurnal_time = np.round(solar_time_hours / 3) * 3
    if diurnal_time == 24:
        diurnal_time = 0

    # Round wind to nearest 10 kt
    rounded_wind = 10 * float(np.round(wind / 10))

    # Count TCB pixels
    tcb_pixels = np.nansum(predict[0, :, :, 0] > 0.02)

    return [diurnal_time, rounded_wind, wind_dif_list, tcb_pixels, df_storm_id]


def mp_running(kwargs: dict):
    """Wrapper for multiprocessing."""
    return running(**kwargs)
