# coding=utf-8
"""
Last Edited: 07/10/2025
@author: John Mark Mayhall
"""
import ephem
import numpy as np
import os
import pandas as pd
from datetime import datetime
from keras.models import load_model
from wind_calc import Wind

model = None
def init_worker():
    global model
    while model is None:
        try:
            model = load_model(os.path.join('/rstor/jmayhall/Model_Training_Code/cnn_creation', 'model.keras'),
                               compile=False)
        except Exception as e:
            print(e)
            model = None


def model_prediction(x_test: np.ndarray, x_test8: np.ndarray) -> np.ndarray:
    """Load Keras model and predict from C13 and C08 input arrays."""
    return model.predict([x_test[None, :], x_test8[None, :]])


def running(i: int, file: str, path_len: int, paths: dict, c8_scaled: pd.DataFrame):
    """
    Main processing function for multiprocessing.
    :return: [diurnal_time, rounded_wind, wind_change_list, TCB pixel count] or None
    """
    identifier = file[path_len: path_len + 22]
    df_storm_id = identifier[:8]
    date = identifier[9:17]
    time = identifier[18:22]
    print(f'Processing file {i + 1}: {identifier}')

    real_time = pd.to_datetime(f'{date}{time}', format='%Y%m%d%H%M')
    wind_dict = pd.read_csv(paths['ships'], sep='\t', index_col=0)

    storm = wind_dict.query("index == @real_time.strftime('%Y-%m-%d %H:%M:%S') and atcf_id == @df_storm_id")
    if storm.empty:
        print(f'No SHIPS data for {df_storm_id} at {real_time}')
        return None

    center_lat = storm.iloc[0].center_lat
    center_lon = storm.iloc[0].center_lon

    # Basin and location filter
    if ('AL' in df_storm_id and not (-105 < center_lon < -20)) or \
            ('EP' in df_storm_id and not (-140 < center_lon < -90)) or \
            not (5 < center_lat < 30):
        print('Outside of bounds, skipping.')
        return None

    # Compute wind change
    wind_calc = Wind(df_storm_id, real_time, wind_dict)
    wind_dif_list = wind_calc.wind_change_calc()

    # Load CH13
    with np.load(file) as data:
        x_test = data["brightness"]

    # Match and load CH08
    c8_file_match = c8_scaled.loc[c8_scaled['Name'].str.contains(identifier), 'Name']
    if c8_file_match.empty:
        print(f'C08 file not found for {identifier}')
        return None

    with np.load(c8_file_match.values[0]) as data:
        x_test8 = data["brightness"]

    # Predict
    predict = model_prediction(x_test, x_test8)
    tcb_pixels = np.nansum(predict[0, :, :, 0] > 0.02)
    return [wind_dif_list, tcb_pixels, df_storm_id]


def mp_running(kwargs):
    """Wrapper for multiprocessing."""
    return running(**kwargs)