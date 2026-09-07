# coding=utf-8
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import zstandard as zstd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from vtkmodules.util.colors import deep_pink

al052023_lats = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                        f'AL052023_20230712_0000_latlon.npz')["lat"].astype(np.float32)
al052023_lons = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                        f'AL052023_20230712_0000_latlon.npz')["lon"].astype(np.float32)
al052023_c13 = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                       f'AL052023_20230712_0000_C13_unscaled_cut.npz')["brightness"].astype(np.float32)

al272020_lats = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                        f'AL272020_20201021_1200_latlon.npz')["lat"].astype(np.float32)
al272020_lons = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                        f'AL272020_20201021_1200_latlon.npz')["lon"].astype(np.float32)
al272020_c13 = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                       f'AL272020_20201021_1200_C13_unscaled_cut.npz')["brightness"].astype(np.float32)

ep132019_lats = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                        f'EP132019_20190918_0600_latlon.npz')["lat"].astype(np.float32)
ep132019_lons = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                        f'EP132019_20190918_0600_latlon.npz')["lon"].astype(np.float32)
ep132019_c13 = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                       f'EP132019_20190918_0600_C13_unscaled_cut.npz')["brightness"].astype(np.float32)

ep192023_lats = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                        f'EP192023_20231102_1800_latlon.npz')["lat"].astype(np.float32)
ep192023_lons = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                        f'EP192023_20231102_1800_latlon.npz')["lon"].astype(np.float32)
ep192023_c13 = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                       f'EP192023_20231102_1800_C13_unscaled_cut.npz')["brightness"].astype(np.float32)

fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
lat, lon, storm_id, x_unscaled, predict, date, time = None, None, None, None, None, None, None
for i, ax in enumerate(axes.ravel()):
    if i == 0:
        lon, lat, x_unscaled, storm_id, date, time = (al052023_lons, al052023_lats, al052023_c13, 'AL052023',
                                                      'July 12th, 2023,', '00:00 UTC')
        ax.set_title(f'TC Don ({storm_id})', fontsize=16)
        extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
        ax.imshow(x_unscaled, cmap='Greys', extent=extent, aspect='auto', vmin=-70, vmax=20)
        grayscale_map = ScalarMappable(Normalize(vmin=-70, vmax=20), cmap='Greys')
    if i == 1:
        lon, lat, x_unscaled, storm_id, date, time = (al272020_lons, al272020_lats, al272020_c13, 'AL272020',
                                                      'October 21st, 2020,', '12:00 UTC')
        ax.set_title(f'TC Epsilon ({storm_id})', fontsize=16)
        extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
        ax.imshow(x_unscaled, cmap='Greys', extent=extent, aspect='auto', vmin=-70, vmax=20)
        grayscale_map = ScalarMappable(Normalize(vmin=-70, vmax=20), cmap='Greys')
    if i == 2:
        lon, lat, x_unscaled, storm_id, date, time = (ep132019_lons, ep132019_lats, ep132019_c13, 'EP132019',
                                                      'September 18th, 2019,', '06:00 UTC')
        ax.set_title(f'TC Kiko ({storm_id})', fontsize=16)
        extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
        ax.imshow(x_unscaled, cmap='Greys', extent=extent, aspect='auto', vmin=-70, vmax=20)
        grayscale_map = ScalarMappable(Normalize(vmin=-70, vmax=20), cmap='Greys')
    if i == 3:
        lon, lat, x_unscaled, storm_id, date, time = (ep192023_lons, ep192023_lats, ep192023_c13, 'EP192023',
                                                      'November 2nd, 2023,', '18:00 UTC')
        ax.set_title(f'TC Pilar ({storm_id})', fontsize=16)
        extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
        ax.imshow(x_unscaled, cmap='Greys', extent=extent, aspect='auto', vmin=-70, vmax=20)
        grayscale_map = ScalarMappable(Normalize(vmin=-70, vmax=20), cmap='Greys')

    # Titles & axes
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

cbar = fig.colorbar(
    grayscale_map,
    ax=axes.ravel(),
    orientation='horizontal',
    pad=0.08,
    fraction=0.05,
    ticks=[-70, -50, -25, 0, 20]
)

cbar.ax.tick_params(labelsize=16)
cbar.set_label('Brightness Temperature (°C)', fontsize=16)

fig.suptitle(f"Transverse Cirrus Bands Examples",
             fontsize=20)

    # Save figure
plt.savefig('tcb_example.png', dpi=300)
z
