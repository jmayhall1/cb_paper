# coding=utf-8
"""
Last Edited: 10/02/2025
@author: John Mark Mayhall
Purpose: Multiprocessing worker functions for TCB prediction using CH13/CH08 brightness and environmental variables.
Optimized for clarity, memory efficiency, and robust error handling.
"""

import gc

import numpy as np
import pandas as pd
from keras.models import load_model

# -------------------------- Model Loading --------------------------

model = None  # Global model object for multiprocessing


def init_worker(model_path: str = '/rstor/jmayhall/Model_Training_Code/cnn_creation/model.keras'):
    """
    Initialize the Keras model for each worker.
    Retries until the model is loaded successfully.
    """
    global model
    while model is None:
        try:
            model = load_model(model_path, compile=False)
        except Exception as e:
            print(f"Model load error: {e}")
            model = None


def model_prediction(x_test: np.ndarray, x_test8: np.ndarray) -> np.ndarray:
    """
    Perform prediction using the global Keras model.

    :param x_test: CH13 brightness array
    :param x_test8: CH08 brightness array
    :return: Prediction array
    """
    return model.predict([x_test[None, :], x_test8[None, :]])


# -------------------------- Main Processing Function --------------------------

def running(file: str, path_len: int, paths: dict, c8_scaled: str, cut_off: float):
    """
    Process a single storm file for TCB prediction and environmental bins.

    :param file: Path to CH13 npz file
    :param path_len: Base path length for extracting identifiers
    :param paths: Dictionary containing paths (including SHIPS file)
    :param c8_scaled: Path to corresponding CH08 npz file
    :param cut_off: Threshold for TCB pixel detection
    :return: List of processed environmental bins and TCB pixel count or None if invalid
    """
    try:
        # ---------------- Extract identifiers ----------------
        df_storm_id = file[path_len: path_len + 8]
        date_str = file[path_len + 9: path_len + 17]
        time_str = file[path_len + 18: path_len + 22]
        real_time = pd.to_datetime(f'{date_str}{time_str}', format='%Y%m%d%H%M')

        # ---------------- Load storm SHIPS data ----------------
        wind_df = pd.read_csv(paths['ships'], sep='\t', index_col=0)
        storm = wind_df[(wind_df.index == str(real_time)) & (wind_df.atcf_id == df_storm_id)]

        if storm.empty:
            print(f"No matching SHIPS data for {df_storm_id} at {real_time}")
            return None

        # Extract environmental variables and storm center
        rhlo, rhmd, rhhi, sst = storm.iloc[0][['rhlo', 'rhmd', 'rhhi', 'dsst']]
        center_lat, center_lon = storm.iloc[0][['center_lat', 'center_lon']]

        # ---------------- Basin/location filter ----------------
        if not (
                ('AL' in df_storm_id and -105 < center_lon < -20) or
                ('EP' in df_storm_id and -140 < center_lon < -90)
        ) or not (5 < center_lat < 30):
            print(f"Storm {df_storm_id} outside of bounds. Skipping.")
            return None

        # ---------------- Load CH13 & CH08 brightness ----------------
        with np.load(file) as data:
            x_test = data['brightness']

        with np.load(c8_scaled) as data:
            x_test8 = data['brightness']

        # ---------------- Predict TCB pixels ----------------
        predict = None
        while predict is None:
            try:
                predict = model_prediction(x_test, x_test8)
            except Exception as e:
                print(f"Prediction failed, retrying: {e}")
                predict = None

        tcb_pixels = np.nansum(predict[0, :, :, 0] > cut_off)

        # ---------------- Round environmental variables ----------------
        rhlo_bin = 10 * float(np.round(rhlo / 10))
        rhmd_bin = 10 * float(np.round(rhmd / 10))
        rhhi_bin = 10 * float(np.round(rhhi / 10))
        sst_bin = np.round(sst)

        # ---------------- Cleanup ----------------
        del (predict, x_test, x_test8, wind_df, storm)
        gc.collect()

        return [rhlo_bin, rhmd_bin, rhhi_bin, sst_bin, tcb_pixels, df_storm_id]

    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None


# -------------------------- Multiprocessing Wrapper --------------------------

def mp_running(kwargs: dict):
    """
    Wrapper function for multiprocessing pool mapping.
    """
    return running(**kwargs)
