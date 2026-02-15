# coding=utf-8
"""
Optimized: 10/01/2025
@author: John Mark Mayhall

Purpose:
--------
Map tropical cyclones (TCs) on average sea surface temperature (SST) fields.
TD, TS, CAT1-2, CAT3-5 categories are plotted on top of SST.
"""

import glob
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def load_ships_data(file_path: Path):
    """Load SHIPS data and convert index to datetime for fast lookups."""
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    df.index = pd.to_datetime(df.index)
    df.atcf_id = df.atcf_id.astype(str)
    return df


def extract_tc_points(ships_df: pd.DataFrame, npz_files: list):
    """
    Extract lat/lon/wind points from NPZ files for TCs present in SHIPS data.
    Returns lists of lats, lons, and max winds.
    """
    lats, lons, winds = [], [], []
    valid_ids = set(ships_df.atcf_id)

    for file in npz_files:
        fname = Path(file).name
        atcf_id = fname[-41:-33]
        date_str = fname[-32:-19]  # e.g., 20220831_00
        timestamp = pd.Timestamp(
            year=int(date_str[:4]),
            month=int(date_str[4:6]),
            day=int(date_str[6:8]),
            hour=int(date_str[-4:-2])
        )

        if atcf_id in valid_ids and timestamp in ships_df.index:
            row = ships_df.loc[(ships_df.index == timestamp) & (ships_df.atcf_id == atcf_id)]
            lats.append(row.center_lat.values[0])
            lons.append(row.center_lon.values[0])
            winds.append(row.max_winds.values[0])

    return lats, lons, winds


def categorize_tc(lats: list, lons: list, winds: list) -> dict:
    """Categorize TC points by wind speed into TD, TS, CAT1-2, CAT3-5."""
    categories = {
        'TD': ([], []),
        'TS': ([], []),
        'CAT12': ([], []),
        'CAT35': ([], [])
    }

    for lat, lon, w in zip(lats, lons, winds):
        if w < 34:
            categories['TD'][0].append(lat)
            categories['TD'][1].append(lon)
        elif 34 <= w < 65:
            categories['TS'][0].append(lat)
            categories['TS'][1].append(lon)
        elif 65 <= w < 96:
            categories['CAT12'][0].append(lat)
            categories['CAT12'][1].append(lon)
        elif w >= 96:
            categories['CAT35'][0].append(lat)
            categories['CAT35'][1].append(lon)
    return categories


def compute_sst_average(file_pattern: str, months: list):
    """Compute seasonal average SST from netCDF files."""
    files = glob.glob(file_pattern)
    count_arr = None
    file_count = 0

    for file in files:
        fname = Path(file).name
        date_str = fname.split('.')[2]  # '199508'
        month = int(date_str[4:6])

        if month in months:
            ds = xr.open_dataset(file)['TSKINWTR'][0].values  # first timestep
            if count_arr is None:
                count_arr = np.zeros_like(ds, dtype=float)
            count_arr += ds
            file_count += 1

    if file_count > 0:
        count_arr /= file_count
        count_arr -= 273.15  # Convert K -> °C
        np.clip(count_arr, 18, 30, out=count_arr)  # Limit SST to 18-30°C
    return count_arr


def plot_tc_sst(categories: dict, sst_avg, lon: list, lat: list, extent: tuple):
    """Plot TC tracks over averaged SST."""
    latn, lats, lonw, lone = extent

    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([lonw, lone, lats, latn], crs=ccrs.PlateCarree())
    ax.set_facecolor(cfeature.COLORS['water'])

    # Add map features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle='--')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.STATES)
    ax.add_feature(cfeature.RIVERS)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=2, color='black', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-140, -120, -100, -80, -60, -40, -20])
    gl.ylocator = mticker.FixedLocator([0, 10, 20, 30])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Plot SST
    sst_plot = ax.contourf(lon, lat, sst_avg, cmap='Greys', vmin=18, vmax=30,
                           levels=np.arange(18, 30.01, 0.1))

    # Plot TC categories
    ax.scatter(categories['TD'][1], categories['TD'][0], color='yellow', s=10, label='TD')
    ax.scatter(categories['TS'][1], categories['TS'][0], color='orange', s=10, label='TS')
    ax.scatter(categories['CAT12'][1], categories['CAT12'][0], color='red', s=10, label='CAT 1-2')
    ax.scatter(categories['CAT35'][1], categories['CAT35'][0], color='magenta', s=10, label='CAT 3-5')

    ax.set_title('Atlantic and Eastern Pacific Analysis Cases\n' +
                 r'with 1995-2024 May-Nov Sea Surface Temperatures (SSTs) in $^\circ C$')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(loc='upper right', framealpha=0.4)

    # Colorbar
    degree_sign = u'\N{DEGREE SIGN}'
    fig.colorbar(sst_plot, ax=ax, location='bottom', label=f'SST ({degree_sign}C)',
                ticks=np.arange(18, 31, 1),
                format=mticker.FixedFormatter(['≤18', '19', '20', '21', '22', '23',
                                               '24', '25', '26', '27', '28', '29', '≥30']),
                extend='both')
    plt.tight_layout()
    plt.savefig('tc_tracks_sst.png', dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    # Load SHIPS data for AL and EP basins
    al_file = Path('//uahdata/rstor/cataloging/nc_process/shear_process/shear_process_all/ships_interp_AL.txt')
    ep_file = Path('//uahdata/rstor/cataloging/nc_process/shear_process/shear_process_all/ships_interp_EP.txt')
    al_df = load_ships_data(al_file)
    ep_df = load_ships_data(ep_file)

    # Gather all NPZ files from both basins
    npz_files = glob.glob(str(al_file.parent / '*.npz')) + glob.glob(str(ep_file.parent / '*.npz'))

    # Extract lat/lon/wind for TCs
    lats, lons, winds = [], [], []
    for df in [al_df, ep_df]:
        lat_tmp, lon_tmp, wind_tmp = extract_tc_points(df, npz_files)
        lats.extend(lat_tmp)
        lons.extend(lon_tmp)
        winds.extend(wind_tmp)

    # Categorize TCs
    categories = categorize_tc(lats, lons, winds)

    # SST averaging
    sst_avg = compute_sst_average('//uahdata/rstor/cataloging/nc_process/map_creating/data/*._nc_',
                                  months=list(np.arange(5, 12, 1)))

    # Get lon/lat for plotting from one SST file
    sample_ds = xr.open_dataset(glob.glob('//uahdata/rstor/cataloging/nc_process/map_creating/data/*._nc_')[0])
    lon, lat = sample_ds.lon.values, sample_ds.lat.values

    # Plot
    plot_tc_sst(categories, sst_avg, lon, lat, extent=(35.1, -0.1, -145.1, -14.9))
