# coding=utf-8
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import zstandard as zstd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from vtkmodules.util.colors import deep_pink

al032023_lats = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/model_prediction_files/'
                        f'AL032023_20230622_1200_latlon.npz')["lat"].astype(np.float32)
al032023_lons = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/model_prediction_files/'
                        f'AL032023_20230622_1200_latlon.npz')["lon"].astype(np.float32)
al032023_c13 = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/model_prediction_files/'
                       f'AL032023_20230622_1200_C13_unscaled_cut.npz')["brightness"].astype(np.float32)
al032023_c8 = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/model_prediction_files/'
                      f'AL032023_20230622_1200_C08_unscaled_cut.npz')["brightness"].astype(np.float32)

al042023_lats = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/model_prediction_files/'
                        f'AL042023_20230622_1200_latlon.npz')["lat"].astype(np.float32)
al042023_lons = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/model_prediction_files/'
                        f'AL042023_20230622_1200_latlon.npz')["lon"].astype(np.float32)
al042023_c13 = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/model_prediction_files/'
                       f'AL042023_20230622_1200_C13_unscaled_cut.npz')["brightness"].astype(np.float32)
al042023_c8 = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/model_prediction_files/'
                      f'AL042023_20230622_1200_C08_unscaled_cut.npz')["brightness"].astype(np.float32)

# ---- Read compressed file ----
with open('C:/Users/jmayhall/PycharmProjects/PythonProject/model_prediction_files/AL032023_20230622_1200_probs.zst', 'rb') as f:
    compressed = f.read()

# ---- Decompress ----
dctx = zstd.ZstdDecompressor()
raw_npz = dctx.decompress(compressed)

# ---- Load NPZ from memory ----
buffer = io.BytesIO(raw_npz)
with np.load(buffer) as data:
    al032023_prob = data['probability']

with open('C:/Users/jmayhall/PycharmProjects/PythonProject/model_prediction_files/AL042023_20230622_1200_probs.zst', 'rb') as f:
    compressed = f.read()

# ---- Decompress ----
dctx = zstd.ZstdDecompressor()
raw_npz = dctx.decompress(compressed)

# ---- Load NPZ from memory ----
buffer = io.BytesIO(raw_npz)
with np.load(buffer) as data:
    al042023_prob = data['probability']

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
cutoff = 0.02
prob_ticks = [0.02, 0.2, 0.4, 0.6, 0.8, 1.0]
lat, lon, storm_id, x_unscaled, predict, date, time = None, None, None, None, None, None, None
for i, ax in enumerate(axes.ravel()):
    if i == 0:
        lon, lat, predict, x_unscaled, storm_id, date, time = (al032023_lons, al032023_lats, al032023_prob,
                                                               al032023_c13, 'AL032023', 'June 22nd, 2023,',
                                                               '12:00 UTC')
        print(np.nanmax(x_unscaled))
        print(np.nanmin(x_unscaled))
        ax.set_title(f'TC Bret ({storm_id})', fontsize=16)
    if i == 1:
        lon, lat, predict, x_unscaled, storm_id, date, time = (al042023_lons, al042023_lats, al042023_prob,
                                                               al042023_c13, 'AL042023', 'June 22nd, 2023,',
                                                               '12:00 UTC')
        ax.set_title(f'TC Cindy ({storm_id})', fontsize=16)
    if i == 2:
        lon, lat, predict, x_unscaled, storm_id, date, time = (al032023_lons, al032023_lats, al032023_prob,
                                                               al032023_c8, 'AL032023', 'June 22nd, 2023,',
                                                               '12:00 UTC')
        print(np.nanmax(x_unscaled))
        print(np.nanmin(x_unscaled))
        ax.set_title(f'TC Bret ({storm_id})', fontsize=16)
    if i == 3:
        lon, lat, predict, x_unscaled, storm_id, date, time = (al042023_lons, al042023_lats, al042023_prob,
                                                               al042023_c8, 'AL042023', 'June 22nd, 2023,',
                                                               '12:00 UTC')
        ax.set_title(f'TC Cindy ({storm_id})', fontsize=16)
    predict[predict <= cutoff] = 0
    extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
    if i == 0 or i == 1:
        ax.imshow(x_unscaled, cmap='Greys', extent=extent, aspect='auto', vmin=-70, vmax=20)
        grayscale_map = ScalarMappable(Normalize(vmin=-70, vmax=20), cmap='Greys')
    if i == 2 or i == 3:
        ax.imshow(x_unscaled, cmap='Greys', extent=extent, aspect='auto', vmin=-70, vmax=-30)
        grayscale_map = ScalarMappable(Normalize(vmin=-70, vmax=-30), cmap='Greys')
    ax.contour(np.flip(predict, axis=0), vmin=cutoff, vmax=1, cmap='rainbow', extent=extent)

    # Titles & axes
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Colorbars
    prob_map = ScalarMappable(Normalize(cutoff, 1), cmap='rainbow')
    if i == 0:
        cbar1 = fig.colorbar(grayscale_map, ax=ax, orientation="horizontal", ticks=[-70, -50, -25, 0, 20])
        cbar1.ax.tick_params(labelsize=16)
        cbar1.set_label('Brightness Temperatures (\N{degree sign}C, CH13)', fontsize=16)

    if i == 1:
        cbar2 = fig.colorbar(prob_map, ax=ax, orientation="horizontal", ticks=prob_ticks)
        cbar2.ax.tick_params(labelsize=16)
        cbar2.set_label('Probability of CB', fontsize=16)

    if i == 2:
        cbar1 = fig.colorbar(grayscale_map, ax=ax, orientation="horizontal", ticks=[-70, -60, -50 ,-40, -30])
        cbar1.ax.tick_params(labelsize=16)
        cbar1.set_label('Brightness Temperatures (\N{degree sign}C, CH08)', fontsize=16)

    if i == 3:
        cbar2 = fig.colorbar(prob_map, ax=ax, orientation="horizontal", ticks=prob_ticks)
        cbar2.ax.tick_params(labelsize=16)
        cbar2.set_label('Probability of CB', fontsize=16)

    # Figure title
    fig.suptitle(f"Cirrus Band Probabilities for {date} at {time}",
                 fontsize=20)

    # Save figure
plt.tight_layout()
plt.savefig('model_prediction.jpg', dpi=300)
