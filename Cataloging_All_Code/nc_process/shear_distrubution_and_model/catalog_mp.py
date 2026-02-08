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
import gc
import io
import os
from multiprocessing import Lock

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import zstandard as zstd
from keras.models import load_model
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# -----------------------------
# Model prediction helper
# -----------------------------
model = None  # Global model object for multiprocessing


def init_worker(model_path: str = '/rstor/jmayhall/Model_Training_Code/cnn_creation/model.keras'):
    """
    Initialize the Keras model ONCE per worker.
    Uses spawn-safe initialization and prevents TF memory/thread leaks.
    """

    global model

    # Clear any prior TF state (in case worker restarted via maxtasksperchild)
    tf.keras.backend.clear_session()

    # OPTIONAL: Restrict TF from grabbing all threads/CPU
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    # Load model with retry logic
    for attempt in range(100):
        try:
            model = load_model(model_path, compile=False)
            return
        except Exception as e:
            print(f"[Worker init] Model load attempt {attempt + 1}/100 failed: {e}")
            model = None

    # If still failed:
    raise RuntimeError("Worker failed to load model after 100 attempts.")


def model_prediction(x_test: np.ndarray, x_test8: np.ndarray) -> np.ndarray:
    """
    Load Keras model from disk and predict probability map for a given test input.

    :param x_test: C13 test array.
    :param x_test8: C08 test array.
    :return: Probability array (height x width).
    """
    return model.predict([x_test[None, :], x_test8[None, :]])[0, :, :, 0]


# -----------------------------
# Plotting routine
# -----------------------------
def plotting(num, file, length, prob_ticks, cutoff, plot):
    """
    Main plotting routine for a single TC image.

    :param num: Index of the current file.
    :param file: Path to C13 scaled npz file.
    :param length: Total number of files.
    :param prob_ticks: List of probability ticks for colorbar.
    :param cutoff: Minimum probability cutoff for plotting.
    :param plot: Bool to decide whether to plot.
    """
    print(f'Processing file {num + 1} of {length}')
    lock = Lock()
    filename = os.path.basename(file)

    # Extract metadata
    storm_id, date, time = filename[:8], filename[9:17], filename[18:22]
    try:
        # Load test arrays
        x_test = np.load(file)["brightness"].astype(np.float32)
        c8_file, c13_unscaled_file, latlon_file = (file[:-18].replace('C13', 'C08'),
                                                   file[:-18].replace('scaled', 'unscaled'),
                                                   file[:-18].replace('C13_scaled', 'latlon_arrs'))
        x_test8 = np.load(f'{c8_file}C08_scaled_cut.npz')["brightness"].astype(np.float32)
        lat = np.load(f'{latlon_file}latlon.npz')["lat"].astype(np.float32)
        lon = np.load(f'{latlon_file}latlon.npz')["lon"].astype(np.float32)

        # Model prediction with retry
        predict = None
        while predict is None:
            try:
                predict = model_prediction(x_test, x_test8)
            except Exception as e:
                print(f'Model prediction error: {e}')
                predict = None

        # Round floats
        predict = predict.astype(np.float32).round(4)
        lat = lat.astype(np.float32).round(4)
        lon = lon.astype(np.float32).round(4)
        print('Setup complete')


        # ---- Save to a temporary .npz in memory, not on disk ----
        if plot==0 or plot==3:
            print('Running plot = 0')
            buffer = io.BytesIO()
            np.savez(buffer, probability=predict, lat=lat, lon=lon)
            raw_npz = buffer.getvalue()

            # ---- Compress with zstd (no pickle!) ----
            cctx = zstd.ZstdCompressor(level=22)

            out_path = (
                f'/rstor/jmayhall/cataloging/nc_process/shear_distrubution_and_model/tcb_probs/'
                f'{storm_id}_{date}_{time}_probs.zst'
            )

            with open(out_path, 'wb') as f:
                f.write(cctx.compress(raw_npz))

        elif plot==1 or plot==3:
            print('Running plot = 1')
            predict = None
            while predict is None:
                try:
                    predict = model_prediction(x_test, x_test8)
                except Exception as e:
                    print(f'Model prediction error: {e}')
                    predict = None
            x_unscaled = np.load(f'{c13_unscaled_file}C13_unscaled_cut.npz')["brightness"].astype(np.float32)
            predict[predict <= cutoff] = 0
            # Setup plot
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
            ax.imshow(x_unscaled, cmap='Greys', extent=extent, aspect='auto')
            ax.contour(np.flip(predict, axis=0), vmin=cutoff, vmax=1, cmap='rainbow', extent=extent)

            # Titles & axes
            ax.set_title(f'TC {storm_id}', fontsize=16)
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
            plt.savefig(f"model_prediction_{storm_id}_{date}_{time}.jpg", dpi=100)
            plt.close()
            lock.release()
        elif plot==2:
            print('Running plot = 2')
            x_unscaled = np.load(f'{c13_unscaled_file}C13_unscaled_cut.npz')["brightness"].astype(np.float32)
            # Setup plot
            fig, ax = plt.subplots(1, 1, figsize=(9, 7))
            extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
            im = ax.imshow(x_unscaled, cmap='Greys', extent=extent, aspect='auto')

            # Titles & axes
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)
            ax.set_xlabel('Longitude', fontsize=16)
            ax.set_ylabel('Latitude', fontsize=16)
            print('Setting axes')

            # Reserve space at bottom
            fig.subplots_adjust(bottom=0.15)  # Move axes up to make room
            cbar1 = fig.colorbar(im, orientation='horizontal', fraction=0.05, pad=0.15)
            cbar1.ax.tick_params(labelsize=16)
            cbar1.set_label('Brightness Temperatures (\N{degree sign}C)', fontsize=16)
            print('Setting colorbar')

            # Figure title
            formatted_date = datetime.datetime.strptime(str(date), '%Y%m%d').strftime('%b %d, %Y')
            formatted_time = datetime.datetime.strptime(str(time), '%H%M').strftime('%H:%M UTC')
            if storm_id == 'AL012022':
                fig.suptitle(f"TC Alex ({storm_id}) on {formatted_date} at {formatted_time}", y=0.95,
                             fontsize=20)
            elif storm_id == 'AL132023':
                fig.suptitle(f"TC Lee ({storm_id}) on {formatted_date} at {formatted_time}", y=0.95,
                             fontsize=20)
            elif storm_id == 'AL202019':
                fig.suptitle(f"TC Sebastian ({storm_id}) on {formatted_date} at {formatted_time}", y=0.95,
                             fontsize=20)
            else:
                fig.suptitle(f"TC {storm_id} on {formatted_date} at {formatted_time}", y=0.95,
                             fontsize=20)
            print('Setting title')

            # Save figure
            lock.acquire()
            print('Saving File')
            plt.savefig(f"model_prediction_{storm_id}_{date}_{time}.jpg", dpi=100)
            plt.close()
            lock.release()
        del (predict, lat, lon, x_test, x_test8, c8_file, c13_unscaled_file,
             latlon_file, storm_id, date, time, filename)
        gc.collect()

    except EOFError:
        print(f'EOFError encountered. File: {file}, Filename slice: {file[:-18]}')


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
