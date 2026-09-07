# coding=utf-8
import xarray as xr
import numpy as np
from pyproj import Geod

# Open GOES-16 ABI L1b file
ds = xr.open_dataset("AL152020_20200902_0300_C13.nc")
al032023_lats = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                        f'AL032023_20230622_1200_latlon.npz')["lat"].astype(np.float32)
al032023_lons = np.load(f'C:/Users/jmayhall/PycharmProjects/PythonProject/'
                        f'AL032023_20230622_1200_latlon.npz')["lon"].astype(np.float32)

# Projection info
proj_info = ds['goes_imager_projection']
H = proj_info.perspective_point_height
lon0 = proj_info.longitude_of_projection_origin

geod = Geod(ellps="WGS84")

ny, nx = al032023_lats.shape

dx = np.full((ny, nx), np.nan, dtype=np.float32)
dy = np.full((ny, nx), np.nan, dtype=np.float32)

# x-direction (east–west)
_, _, dist_x = geod.inv(
    al032023_lons[:, :-1],
    al032023_lats[:, :-1],
    al032023_lons[:, 1:],
    al032023_lats[:, 1:]
)
dx[:, :-1] = dist_x / 1000.0  # km

# y-direction (north–south)
_, _, dist_y = geod.inv(
    al032023_lons[:-1, :],
    al032023_lats[:-1, :],
    al032023_lons[1:, :],
    al032023_lats[1:, :]
)
dy[:-1, :] = dist_y / 1000.0  # km
print(f'Lon Min: {np.nanmin(al032023_lons)}')
print(f'Lon Max: {np.nanmax(al032023_lons)}')
print(f'Lat Min: {np.nanmin(al032023_lats)}')
print(f'Lat Max: {np.nanmax(al032023_lats)}')
print('Nadir pixel size: 2 km')
print(f'dx Min: {np.nanmin(dx)} km')
print(f'dx Max: {np.nanmax(dx)} km')
print(f'dy Min: {np.nanmin(dy)} km')
print(f'dy Max: {np.nanmax(dy)} km')
