# coding=utf-8
"""
Last Edited: 10/02/2025
@author: John Mark Mayhall
Purpose: Multiprocessing worker functions for TCB prediction using CNN.
Optimized and fully commented.
"""

from datetime import datetime

import ephem
import numpy as np
import pandas as pd
from keras.models import load_model


# -------------------------- Solar Time --------------------------

def solartime(observer: ephem.Observer, sun: ephem.Sun = ephem.Sun()):
    """
    Compute local solar time (LST) for an observer.

    :param observer: ephem.Observer with date, latitude, longitude set.
    :param sun: ephem.Sun() object.
    :return: Solar time as ephem.hours object.
    """
    sun.compute(observer)
    hour_angle = observer.sidereal_time() - sun.ra
    return ephem.hours(hour_angle + ephem.hours('12:00')).norm


# -------------------------- Model Loading --------------------------

model = None  # Global model object


def init_worker(model_path: str = '/rstor/jmayhall/Model_Training_Code/cnn_creation/model.keras'):
    """
    Initialize Keras model in each worker for multiprocessing.
    Retries until successfully loaded.
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
    Perform model prediction using loaded global Keras model.

    :param x_test: CH13 input array.
    :param x_test8: CH08 input array.
    :return: Prediction array.
    """
    return model.predict([x_test[None, :], x_test8[None, :]])


# -------------------------- Main Processing Function --------------------------

def running(i: int, file: str, path_len: int, paths: dict, c8_scaled: pd.DataFrame):
    """
    Process a single storm file for CNN TCB prediction.

    :param i: Index of the file in processing queue.
    :param file: Path to CH13 npz file.
    :param path_len: Base path length for extracting identifiers.
    :param paths: Dictionary of paths (including Statistical Hurricane Intensity Prediction Scheme/SHIPS file).
    :param c8_scaled: DataFrame containing CH08 file paths.
    :return: [diurnal_temp_time, tcb_pixels, df_storm_id] or None if invalid.
    """
    # Extract identifiers from filename
    identifier = file[path_len: path_len + 22]
    df_storm_id = identifier[:8]
    date = identifier[9:17]
    time = identifier[18:22]
    print(f'Processing file {i + 1}: {identifier}')

    # Convert to datetime
    real_time = pd.to_datetime(f'{date}{time}', format='%Y%m%d%H%M')

    # Load SHIPS data once per call
    wind_dict = pd.read_csv(paths['ships'], sep='\t', index_col=0)
    storm = wind_dict.query("index == @real_time.strftime('%Y-%m-%d %H:%M:%S') and atcf_id == @df_storm_id")

    if storm.empty:
        print(f'No SHIPS data for {df_storm_id} at {real_time}')
        return None

    center_lat = storm.iloc[0].center_lat
    center_lon = storm.iloc[0].center_lon

    # Basin & location filter
    if ('AL' in df_storm_id and not (-105 < center_lon < -20)) or \
            ('EP' in df_storm_id and not (-140 < center_lon < -90)) or \
            not (5 < center_lat < 30):
        print('Outside of bounds, skipping.')
        return None

    # Load CH13 data
    with np.load(file) as data:
        x_test = data["brightness"]

    # Match CH08 file
    c8_file_match = c8_scaled.loc[c8_scaled['Name'].str.contains(identifier), 'Name']
    if c8_file_match.empty:
        print(f'C08 file not found for {identifier}')
        return None

    with np.load(c8_file_match.values[0]) as data:
        x_test8 = data["brightness"]

    # Predict with retry in case of transient errors
    predict = None
    while predict is None:
        try:
            predict = model_prediction(x_test, x_test8)
        except Exception as e:
            print(f'Model prediction error: {e}')
            predict = None

    # Compute Local Solar Time (LST)
    storm_time = pd.Timestamp(storm.index[0])
    observer = ephem.Observer()
    observer.date = datetime(*storm_time.timetuple()[:6])
    observer.long = center_lon * np.pi / 180

    solar_time = ephem.hours(solartime(observer))
    # Convert to hours and round to nearest integer (diurnal bin)
    solar_hour = solar_time.norm * 24 / (2 * np.pi)  # ephem.hours -> hours
    diurnal_temp_time = int(round(solar_hour)) % 24  # wrap 24 -> 0

    # Count TCB pixels above threshold
    tcb_pixels = np.nansum(predict[0, :, :, 0] > 0.02)

    return [diurnal_temp_time, tcb_pixels, df_storm_id]


# -------------------------- Multiprocessing Wrapper --------------------------

def mp_running(kwargs: dict):
    """
    Wrapper to allow running() with multiprocessing pool.
    """
    return running(**kwargs)
