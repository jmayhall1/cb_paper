# coding=utf-8
"""
Last Edited: 10/02/2025
@author: John Mark Mayhall
Purpose: Determine if transverse bands exist in each quadrant of a tropical cyclone.
"""
import glob
from multiprocessing import Pool
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def load_ships_data(file_path: Path) -> pd.DataFrame:
    """Load SHIPS interpolated data and convert index to datetime."""
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


def get_needed_files(file_pattern: str, ships_df: pd.DataFrame, lon_bounds: tuple, lat_bounds: tuple) -> list:
    """Filter .npz files for valid ATCF IDs and timestamp within bounds."""
    valid_ids = set(ships_df.atcf_id)
    needed_files = []

    for file in glob.glob(file_pattern):
        fname = Path(file).name
        atcf_id = fname[-41:-33]
        date_str = fname[-32:-19]  # e.g., 20220831_00
        ts = pd.Timestamp(year=int(date_str[:4]),
                          month=int(date_str[4:6]),
                          day=int(date_str[6:8]),
                          hour=int(date_str[-4:-2]))

        if (atcf_id in valid_ids and ts in ships_df.index and
                lon_bounds[0] < ships_df.center_lon.values[0] < lon_bounds[1] and
                lat_bounds[0] < ships_df.center_lat.values[0] < lat_bounds[1]):
            needed_files.append(file.replace('shear_process/shear_process_all', 'tcb_mapping/files'))
    return needed_files


def process_file(file_mp: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process a single .npz file and return TCB pixel counts and total pixel counts."""
    data = np.load(file_mp)
    lon_arr = np.round(data['lon'].flatten(), 1)
    lat_arr = np.round(data['lat'].flatten(), 1)
    pixel_arr = np.round(data['pixels'][0, :, :, 0].flatten(), 1)

    mask = pixel_arr > PIXEL_THRESHOLD
    filtered_lons = lon_arr[mask]
    filtered_lats = lat_arr[mask]

    pixel_counts = pd.crosstab(filtered_lats, filtered_lons)
    total_counts = pd.crosstab(lat_arr, lon_arr)

    pixel_df = pd.DataFrame(0, index=LAT_RANGE, columns=LON_RANGE)
    total_df = pd.DataFrame(0, index=LAT_RANGE, columns=LON_RANGE)
    pixel_df.update(pixel_counts)
    total_df.update(total_counts)

    return pixel_df, total_df


def run_parallel(file_list: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run multiprocessing over all needed files and sum results."""
    pixel_df = pd.DataFrame(0, index=LAT_RANGE, columns=LON_RANGE)
    total_df = pd.DataFrame(0, index=LAT_RANGE, columns=LON_RANGE)

    with Pool(NUM_WORKERS) as pool:
        for i, (px_df, tc_df) in enumerate(pool.imap_unordered(process_file, file_list)):
            print(f"Processing file {i + 1} of {len(file_list)}")
            pixel_df = pixel_df.add(px_df)
            total_df = total_df.add(tc_df)

    return pixel_df, total_df


def plot_heatmaps(pixel_df: pd.DataFrame, total_df: pd.DataFrame, output_file: str):
    """Plot original, normalized, and sample count heatmaps."""
    extent = [-141, -19, 4, 31]

    fig, axes = plt.subplots(
        3, 1,
        figsize=(10, 13),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    fig.suptitle(
        'Heatmap of the number of TCB Pixels from 2019–2023\n'
        'from Atlantic and Eastern Pacific Tropical Cyclones',
        fontsize=18, y=0.93
    )

    fig.subplots_adjust(top=0.90, bottom=0.08, hspace=0.05)

    # -------------------------
    # Panel A: Raw TCB counts
    # -------------------------
    im1 = axes[0].imshow(
        pixel_df.to_numpy(),
        cmap='gist_heat',
        origin='lower',
        extent=extent,
        vmin=0,
        vmax=35_000,
        transform=ccrs.PlateCarree()
    )
    axes[0].set_title('Raw TCB Pixel Counts', fontsize=16)

    # -------------------------
    # Panel B: Normalized %
    # -------------------------
    normalized = np.nan_to_num(
        pixel_df.to_numpy() / total_df.to_numpy() * 100
    )

    im2 = axes[1].imshow(
        normalized,
        cmap='gist_heat',
        origin='lower',
        extent=extent,
        vmin=0,
        vmax=40,
        transform=ccrs.PlateCarree()
    )
    axes[1].set_title('Normalized (% of Pixels that are TCBs)', fontsize=16)

    # -------------------------
    # Panel C: Sample Count (Denominator)
    # -------------------------
    im3 = axes[2].imshow(
        total_df.to_numpy(),
        cmap='gist_heat',
        origin='lower',
        extent=extent,
        vmin=0,
        vmax=160_000,
        transform=ccrs.PlateCarree()
    )
    axes[2].set_title('Total Pixel Samples per Bin', fontsize=16)

    # -------------------------
    # Common Map Features
    # -------------------------
    for ax in axes:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_facecolor(cfeature.COLORS['water'])
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE, edgecolor='white')
        ax.add_feature(cfeature.BORDERS, linestyle='--', edgecolor='white')
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.STATES, edgecolor='white')
        ax.add_feature(cfeature.RIVERS)

        gl = ax.gridlines(draw_labels=True, linewidth=1, color='white',
                          alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator([-140, -120, -100, -80, -60, -40, -20])
        gl.ylocator = mticker.FixedLocator([0, 10, 20, 30])
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}

    # -------------------------
    # Colorbars
    # -------------------------
    cbar1 = fig.colorbar(im1, ax=axes[0], orientation="horizontal", pad=0.15)
    cbar1.set_label('Number of TCB Pixels', fontsize=14)

    cbar2 = fig.colorbar(im2, ax=axes[1], orientation="horizontal", pad=0.15)
    cbar2.set_label('% of Pixels that are TCBs', fontsize=14)

    cbar3 = fig.colorbar(im3, ax=axes[2], orientation="horizontal", pad=0.15)
    cbar3.set_label('Total Number of Pixel Samples', fontsize=14)

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    LAT_RANGE = np.round(np.arange(31, 3.9, -1), 1)
    LON_RANGE = np.round(np.arange(-141, -18.9, 1), 1)
    NUM_WORKERS = 64
    PIXEL_THRESHOLD = 0.02
    # --- Atlantic ---
    ships_al = load_ships_data(Path('/rstor/jmayhall/cataloging/nc_process/shear_process/'
                                    'shear_process_all/ships_interp_AL.txt'))
    files_al = get_needed_files('/rstor/jmayhall/cataloging/nc_process/shear_process/'
                                    'shear_process_all/AL*.npz',
                                ships_al, lon_bounds=(-105, -20), lat_bounds=(5, 30))

    # --- Eastern Pacific ---
    ships_ep = load_ships_data(Path('/rstor/jmayhall/cataloging/nc_process/shear_process/'
                                    'shear_process_all/ships_interp_EP.txt'))
    files_ep = get_needed_files('/rstor/jmayhall/cataloging/nc_process/shear_process/'
                                    'shear_process_all/EP*.npz',
                                ships_ep, lon_bounds=(-140, -90), lat_bounds=(5, 30))

    all_files = files_al + files_ep

    pixel_df, tc_df = run_parallel(all_files)
    plot_heatmaps(pixel_df, tc_df, "tcb_heatmap.png")
