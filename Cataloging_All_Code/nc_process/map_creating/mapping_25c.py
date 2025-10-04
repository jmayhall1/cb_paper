# coding=utf-8
"""
Optimized: 10/01/2025
@author: John Mark Mayhall

Purpose:
--------
Map tropical cyclones (TCs) occurring over ~25°C SST currently, in the past 24 hours, or in the next 24 hours.
TD, TS, CAT1-2, CAT3-5 categories are plotted.
"""

import glob
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def load_ships_data(file_path: Path) -> pd.DataFrame:
    """
    Load SHIPS data with datetime index for fast lookup.
    """
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    df.index = pd.to_datetime(df.index)
    df.atcf_id = df.atcf_id.astype(str)
    return df


def extract_tc_25c_points(ships_df: pd.DataFrame, npz_files: list, target_sst: float = 25.0, hours_range: int = 24):
    """
    Extract TC points that are currently, previously, or next 24 hours over ~25°C SST.
    Returns lists of latitudes, longitudes, and wind speeds.
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
            # Check current SST
            row = ships_df.loc[(ships_df.index == timestamp) & (ships_df.atcf_id == atcf_id)]
            if np.round(row.dsst.values[0]) == target_sst:
                # Add current point
                lats.append(row.center_lat.values[0])
                lons.append(row.center_lon.values[0])
                winds.append(row.max_winds.values[0])

                # Check past and future 24 hours
                for delta in range(1, hours_range + 1):
                    for offset in [-delta, delta]:
                        new_time = timestamp + pd.Timedelta(hours=offset)
                        new_row = ships_df.loc[(ships_df.index == new_time) & (ships_df.atcf_id == atcf_id)]
                        if not new_row.empty and np.round(new_row.dsst.values[0]) != target_sst:
                            lats.append(new_row.center_lat.values[0])
                            lons.append(new_row.center_lon.values[0])
                            winds.append(new_row.max_winds.values[0])

    return lats, lons, winds


def categorize_tc(lats: list, lons: list, winds: list) -> dict:
    """
    Categorize TC points by wind speed into TD, TS, CAT1-2, CAT3-5.
    Returns a dictionary of lists.
    """
    categories = {'TD': ([], []), 'TS': ([], []), 'CAT12': ([], []), 'CAT35': ([], [])}
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


def plot_tc_25c(categories: dict, extent: tuple = (35.1, -0.1, -145.1, -14.9), output_file: str = 'tc_tracks_25C.png'):
    """
    Plot TC points over the Atlantic & East Pacific region.
    """
    latn, lats, lonw, lone = extent
    fig, ax = plt.subplots(figsize=(10, 4), subplot_kw={'projection': ccrs.PlateCarree()})
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
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='black', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-140, -120, -100, -80, -60, -40, -20])
    gl.ylocator = mticker.FixedLocator([0, 10, 20, 30])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Plot TCs by category
    ax.scatter(*categories['TD'][1], *categories['TD'][0], color='yellow', s=10, label='TD')
    ax.scatter(*categories['TS'][1], *categories['TS'][0], color='orange', s=10, label='TS')
    ax.scatter(*categories['CAT12'][1], *categories['CAT12'][0], color='red', s=10, label='CAT 1-2')
    ax.scatter(*categories['CAT35'][1], *categories['CAT35'][0], color='magenta', s=10, label='CAT 3-5')

    ax.set_title(
        'Training, Validation, and Analysis Cases\nAtlantic & East Pacific TC Images\n'
        'Over SST ~25°C Currently or ±24 Hours'
    )
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(loc='upper right', framealpha=0.4)
    plt.savefig(output_file, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    # Paths to SHIPS data
    al_file = Path('//uahdata/rstor/cataloging/nc_process/shear_process/shear_process_all/ships_interp_AL.txt')
    ep_file = Path('//uahdata/rstor/cataloging/nc_process/shear_process/shear_process_all/ships_interp_EP.txt')

    # Load SHIPS data
    al_df = load_ships_data(al_file)
    ep_df = load_ships_data(ep_file)

    # Gather all NPZ files
    npz_files = glob.glob(str(al_file.parent / '*.npz')) + glob.glob(str(ep_file.parent / '*.npz'))

    # Extract TC points over 25C SST
    lats, lons, winds = [], [], []
    for df in [al_df, ep_df]:
        lat_tmp, lon_tmp, wind_tmp = extract_tc_25c_points(df, npz_files)
        lats.extend(lat_tmp)
        lons.extend(lon_tmp)
        winds.extend(wind_tmp)

    # Categorize TCs
    categories = categorize_tc(lats, lons, winds)

    # Plot
    plot_tc_25c(categories)
