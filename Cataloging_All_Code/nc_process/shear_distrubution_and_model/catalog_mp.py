# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall

Purpose:
Run preprocessed C13/C08 TC imagery through a pre-trained Keras model,
plot probability overlays on original brightness temperature images,
and save the resulting figures. Handles special case of AL032023/AL042023.
"""

import datetime
import os
from multiprocessing import Lock

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# -----------------------------
# Model prediction helper
# -----------------------------
def model_prediction(model_path: str, x_test: np.ndarray, x_test8: np.ndarray) -> np.ndarray:
    """
    Load Keras model from disk and predict probability map for a given test input.

    :param model_path: Path to the folder containing 'model.keras'.
    :param x_test: C13 test array.
    :param x_test8: C08 test array.
    :return: Probability array (height x width).
    """
    model = load_model(os.path.join(model_path, 'model.keras'), compile=False)
    return model.predict([x_test[None, :], x_test8[None, :]])[0, :, :, 0]


# -----------------------------
# Plotting routine
# -----------------------------
def plotting(num, file, length, model_path, c8_files, c13_unscaled_files, latlon_files, prob_ticks, cutoff):
    """
    Main plotting routine for a single TC image.

    :param num: Index of the current file.
    :param file: Path to C13 scaled npz file.
    :param length: Total number of files.
    :param model_path: Path to trained model.
    :param c8_files: Dict of C08 scaled file paths.
    :param c13_unscaled_files: Dict of C13 unscaled file paths.
    :param latlon_files: Dict of lat/lon file paths.
    :param prob_ticks: List of probability ticks for colorbar.
    :param cutoff: Minimum probability cutoff for plotting.
    """
    print(f'Processing file {num + 1} of {length}')
    lock = Lock()
    filename = os.path.basename(file)

    # Extract metadata
    storm_id, date, time = filename[:8], filename[9:17], filename[18:22]

    # Handle special dual-storm case
    storm_pairs = [(storm_id, file)]
    if storm_id == 'AL032023' and date == '20230622' and time == '1200':
        storm_pairs.append(('AL042023', file.replace('AL032023', 'AL042023')))

    for sid, file_path in storm_pairs:
        try:
            # Load test arrays
            x_test = np.load(file_path)["brightness"].astype(np.float32)
            x_test8 = np.load(c8_files.get(f'{file_path[:-18]}C08_scaled_cut.npz'))["brightness"].astype(np.float32)
            x_unscaled = np.load(c13_unscaled_files.get(f'{file_path[:-18]}'
                                                        f'C13_scaled_cut.npz'))["brightness"].astype(np.float32)
            lat = np.load(latlon_files.get(f'{file_path[:-18]}latlon.npz'))["lat"].astype(np.float32)
            lon = np.load(latlon_files.get(f'{file_path[:-18]}latlon.npz'))["lon"].astype(np.float32)

            # Model prediction with retry
            predict = None
            while predict is None:
                try:
                    predict = model_prediction(model_path, x_test, x_test8)
                except Exception as e:
                    print(f'Model prediction error: {e}')
                    predict = None

            # Mask probabilities below cutoff
            predict[predict <= cutoff] = 0

            # Setup plot
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
            ax.imshow(x_unscaled, cmap='Greys', extent=extent, aspect='auto')
            ax.contour(np.flip(predict, axis=0), vmin=cutoff, vmax=1, cmap='rainbow', extent=extent)

            # Titles & axes
            ax.set_title(f'TC {sid}', fontsize=16)
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)

            # Colorbars
            grayscale_map = ScalarMappable(Normalize(vmin=-80, vmax=30), cmap='Greys')
            prob_map = ScalarMappable(Normalize(cutoff, 1), cmap='rainbow')
            cbar1 = fig.colorbar(grayscale_map, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08,
                                 ticks=[-80, -50, -25, 0, 30])
            cbar2 = fig.colorbar(prob_map, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08, ticks=prob_ticks)
            cbar1.ax.tick_params(labelsize=16)
            cbar2.ax.tick_params(labelsize=16)
            cbar1.set_label('Brightness Temperatures (\N{degree sign}C)', fontsize=16)
            cbar2.set_label('Probability of TCB', fontsize=16)

            # Figure title
            formatted_date = datetime.datetime.strptime(str(date), '%Y%m%d').strftime('%b %d, %Y')
            formatted_time = datetime.datetime.strptime(str(time), '%H%M').strftime('%H:%M UTC')
            fig.suptitle(f"Transverse Cirrus Bands Probabilities for {formatted_date} at {formatted_time}", y=0.95,
                         fontsize=20)

            # Save figure
            lock.acquire()
            plt.savefig(f"model_prediction_{sid}.jpg", dpi=100)
            plt.close()
            lock.release()

        except EOFError:
            print(f'EOFError encountered. File: {file_path}, Filename slice: {file_path[:-18]}')


# -----------------------------
# Multiprocessing wrapper
# -----------------------------
def mp_plotting(kwargs):
    """
    Wrapper for multiprocessing.

    :param kwargs: Dictionary of plotting arguments.
    :return: Result of plotting function.
    """
    return plotting(**kwargs)
