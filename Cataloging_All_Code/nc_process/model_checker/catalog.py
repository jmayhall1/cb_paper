# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall

Purpose:
--------
Run tropical cyclone (TC) satellite data through a pre-trained CNN model,
generate probability maps of transverse cirrus bands (TCB), overlay predictions
on brightness temperature images, and save annotated probability plots.
"""

import datetime
import glob
import os
import warnings
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# ======================================================================
# INITIALIZER (runs once per worker)
# ======================================================================
def init_worker(model_path: str):
    """
    Load the pre-trained Keras model once per worker.
    This avoids re-loading the model for every image processed.
    """
    global model
    while model is None:  # Retry loop in case of loading errors
        try:
            model = load_model(os.path.join(model_path, 'model.keras'), compile=False)
        except Exception as e:
            print(f"Error loading model in worker: {e}")
            model = None


# ======================================================================
# MODEL PREDICTION FUNCTION
# ======================================================================
def model_prediction(x_test: np.ndarray, x_test8: np.ndarray) -> np.ndarray:
    """
    Run CNN prediction on given inputs.

    Args:
        x_test  : Scaled brightness input (C13 channel).
        x_test8 : Additional scaled brightness input (C08 channel).

    Returns:
        np.ndarray: Probability map of shape (1024, 1024).
    """
    global model
    return model.predict([x_test[None, :], x_test8[None, :]])[0, :, :, 0]


# ======================================================================
# MAIN PLOTTING FUNCTION
# ======================================================================
def plotting(num: int, file: str, length: int,
             c8_files: dict, c13_unscaled_files: dict, latlon_files: dict,
             prob_ticks: list, cutoff: float):
    """
    Generate probability plot for a single satellite data file.

    Args:
        num : Index of the input file.
        file : Filename of the input file.
        length    : Total number of files being processed.
        c8_files  : Dictionary of C08 scaled files (by base_key).
        c13_unscaled_files: Dictionary of unscaled C13 files.
        latlon_files       : Dictionary of lat/lon arrays.
        prob_ticks : Tick marks for probability colorbar.
        cutoff     : Probability threshold for plotting.
    """
    print(f'Processing file {num + 1} of {length}')
    filename = os.path.basename(file)

    # Parse storm ID, date, and time from filename
    storm_id, date, time = filename[:8], filename[9:17], filename[18:22]
    base_key = filename[:-18]  # Shared prefix for related files

    try:
        # --- Load input arrays ---
        x_test = np.load(file)["brightness"].astype(np.float32)
        x_test8 = np.load(c8_files[f'{base_key}C08_scaled_cut.npz'])["brightness"].astype(np.float32)
        x_unscaled = np.load(c13_unscaled_files[f'{base_key}C13_scaled_cut.npz'])["brightness"].astype(np.float32)
        lat = np.load(latlon_files[f'{base_key}latlon.npz'])["lat"].astype(np.float32)
        lon = np.load(latlon_files[f'{base_key}latlon.npz'])["lon"].astype(np.float32)

        # --- Run CNN prediction ---
        predict = model_prediction(x_test, x_test8)

        # Fraction of pixels above cutoff (normalized by total pixels)
        total = np.nansum((predict >= cutoff).astype(int)) / (1024 * 1024)

        # Only generate plots if high confidence (> 60% of pixels exceed cutoff)
        if total >= 0.6:
            print(f'High Prediction ({total * 100:.2f}%) for file {file}')

            fig, ax = plt.subplots(figsize=(16, 8))
            extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]

            # Background: unscaled brightness temperatures
            ax.imshow(x_unscaled, cmap='Greys', extent=extent, aspect='auto')

            # Overlay probability contours (only where above cutoff)
            predict[predict <= cutoff] = 0
            ax.contour(np.flip(predict, axis=0), vmin=cutoff, vmax=1,
                       cmap='rainbow', extent=extent)

            # Axis styling
            ax.set_title(f'TC {storm_id}', fontsize=16)
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)

            # --- Add Colorbars ---
            grayscale_map = ScalarMappable(Normalize(vmin=-80, vmax=30), cmap='Greys')
            prob_map = ScalarMappable(Normalize(cutoff, 1), cmap='rainbow')

            cbar1 = fig.colorbar(grayscale_map, ax=ax, orientation="vertical",
                                 fraction=0.05, pad=0.08, ticks=[-80, -50, -25, 0, 30])
            cbar2 = fig.colorbar(prob_map, ax=ax, orientation="horizontal",
                                 fraction=0.05, pad=0.08, ticks=prob_ticks)

            cbar1.ax.tick_params(labelsize=16)
            cbar2.ax.tick_params(labelsize=16)
            cbar2.set_label('Probability of TCB', fontsize=16)
            cbar1.set_label('Brightness Temperatures (°C)', fontsize=16)

            # --- Title with formatted date/time ---
            formatted_date = datetime.datetime.strptime(date, '%Y%m%d').strftime('%b %d, %Y')
            formatted_time = datetime.datetime.strptime(time, '%H%M').strftime('%H:%M UTC')
            fig.suptitle(
                f"Transverse Cirrus Bands Probabilities for {formatted_date} at {formatted_time}",
                y=0.95, fontsize=20
            )

            # Save and close
            out_path = f"{storm_id}_{date}_{time}_prediction.jpg"
            plt.savefig(out_path, dpi=100)
            plt.close()

    except Exception as e:
        print(f'Error processing file {file}: {e}')


# Wrapper for multiprocessing
def mp_plotting(kwargs):
    """
    Multiprocessing wrapper function
    :param kwargs: Needed multiprocessing args
    :return: Nothing
    """
    return plotting(**kwargs)


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================
def _load_file_dict(path: str) -> dict:
    """
    Map filenames to their full paths for fast lookup.
    """
    return {os.path.basename(f): f for f in glob.glob(path)}


# ======================================================================
# MULTIPROCESSING EXECUTION
# ======================================================================
if __name__ == '__main__':
    # Suppress TensorFlow/Keras UserWarnings
    warnings.filterwarnings("ignore", category=UserWarning)
    # ========== GLOBALS ==========
    model = None  # Model loaded once per worker process
    # ======================================================================
    # CONFIGURATION
    # ======================================================================
    config = {
        "model_path": "/rstor/jmayhall/Model_Training_Code/cnn_creation",
        "latlon_path": "/rstor/jmayhall/cataloging/nc_process/geojson_transform/completed_arrays/latlon_arrs/*.npz",
        "cutoff": 0.02,
        "c8_path": "/rstor/jmayhall/cataloging/nc_process/geojson_transform/completed_arrays/C08_scaled/*",
        "c13_scaled_path": "/rstor/jmayhall/cataloging/nc_process/geojson_transform/completed_arrays/C13_scaled/*",
        "c13_unscaled_path": "/rstor/jmayhall/cataloging/nc_process/geojson_transform/completed_arrays/C13_scaled/*",
        "dataframe_path": "/rstor/jmayhall/cataloging/hurdat_update_interp.txt"
    }

    # ======================================================================
    # LOAD INPUT FILES
    # ======================================================================
    c8_files = _load_file_dict(config["c8_path"])
    c13_unscaled_files = _load_file_dict(config["c13_unscaled_path"])
    latlon_files = _load_file_dict(config["latlon_path"])
    c13_scaled_files = glob.glob(config["c13_scaled_path"])
    prob_ticks = [0.02, 0.2, 0.4, 0.6, 0.8, 1]

    # ======================================================================
    # CREATE ARGUMENT LIST FOR MULTIPROCESSING
    # ======================================================================
    needed_args = [{
        'num': num,
        'file': file,
        'length': len(c13_scaled_files),
        'c8_files': c8_files,
        'c13_unscaled_files': c13_unscaled_files,
        'latlon_files': latlon_files,
        'prob_ticks': prob_ticks,
        'cutoff': config["cutoff"]
    } for num, file in enumerate(c13_scaled_files)]
    NUM_WORKERS = 16
    with Pool(NUM_WORKERS, initializer=init_worker,
              initargs=(config["model_path"],)) as pool:
        pool.map(mp_plotting, needed_args)
