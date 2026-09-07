# coding=utf-8
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import zstandard as zstd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

AL202019_lats = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                        f'AL202019_20191120_1200_latlon.npz')["lat"].astype(np.float32)
AL202019_lons = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                        f'AL202019_20191120_1200_latlon.npz')["lon"].astype(np.float32)
AL202019_c13 = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                       f'AL202019_20191120_1200_C13_unscaled_cut.npz')["brightness"].astype(np.float32)

AL202019_lats_2 = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                        f'AL202019_20191121_0000_latlon.npz')["lat"].astype(np.float32)
AL202019_lons_2 = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                        f'AL202019_20191121_0000_latlon.npz')["lon"].astype(np.float32)
AL202019_c13_2 = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                       f'AL202019_20191121_0000_C13_unscaled_cut.npz')["brightness"].astype(np.float32)
# ---- Read compressed file ----
with open('C:/Users/jmayhall/PycharmProjects/PythonProject/AL202019_20191120_1200_probs.zst', 'rb') as f:
    compressed = f.read()

# ---- Decompress ----
dctx = zstd.ZstdDecompressor()
raw_npz = dctx.decompress(compressed)

# ---- Load NPZ from memory ----
buffer = io.BytesIO(raw_npz)
with np.load(buffer) as data:
    AL202019_prob = data['probability']

with open('C:/Users/jmayhall/PycharmProjects/PythonProject/AL202019_20191121_0000_probs.zst', 'rb') as f:
    compressed = f.read()

# ---- Decompress ----
dctx = zstd.ZstdDecompressor()
raw_npz = dctx.decompress(compressed)

# ---- Load NPZ from memory ----
buffer = io.BytesIO(raw_npz)
with np.load(buffer) as data:
    AL202019_prob_2 = data['probability']

fig, axes = plt.subplots(1, 2, figsize=(12, 8))
cutoff = 0.02
prob_ticks = [0.02, 0.2, 0.4, 0.6, 0.8, 1.0]
lat, lon, storm_id, x_unscaled, predict, date, time = None, None, None, None, None, None, None
for i, ax in enumerate(axes.ravel()):
    if i == 0:
        lon, lat, predict, x_unscaled, storm_id, date, time = (AL202019_lons, AL202019_lats, AL202019_prob,
                                                               AL202019_c13, 'AL202019', 'November 20th, 2019,',
                                                               '12:00 UTC')
        print(np.nanmax(x_unscaled))
        print(np.nanmin(x_unscaled))
        ax.set_title(f'AL202019 for {date} at {time}', fontsize=16)
    if i == 1:
        lon, lat, predict, x_unscaled, storm_id, date, time = (AL202019_lons_2, AL202019_lats_2, AL202019_prob_2,
                                                               AL202019_c13_2, 'AL202019', 'November 21st, 2019,',
                                                               '00:00 UTC')
        ax.set_title(f'AL202019 for {date} at {time}', fontsize=16)
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
        cbar1.set_label('Brightness Temperatures (\N{degree sign}C)', fontsize=16)

    if i == 1:
        cbar2 = fig.colorbar(prob_map, ax=ax, orientation="horizontal", ticks=prob_ticks)
        cbar2.ax.tick_params(labelsize=16)
        cbar2.set_label('Probability of TCB', fontsize=16)

    # Figure title
    fig.suptitle(f"Transverse Cirrus Bands Probabilities for AL202019 (Sebastian)",
                 fontsize=20)

    # Save figure
plt.tight_layout()
plt.savefig('sebastian_prediction.jpg', dpi=300)
