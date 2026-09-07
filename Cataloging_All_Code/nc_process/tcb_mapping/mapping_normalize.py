# coding=utf-8
"""
Last Edited: 10/02/2025
@author: John Mark Mayhall
Purpose: Determine if cirrus bands exist in each quadrant of a tropical cyclone.
"""
import glob
from multiprocessing import Pool
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter
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

        if atcf_id in valid_ids and ts in ships_df.index:
            # FIX: Get the row(s) specifically for this timestamp instead of the 0th row of the whole df
            row = ships_df.loc[ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]  # Handle case if there are duplicate timestamps

            if (lon_bounds[0] < row.center_lon < lon_bounds[1] and
                    lat_bounds[0] < row.center_lat < lat_bounds[1]):
                needed_files.append(file.replace('shear_process/shear_process_all', 'tcb_mapping/files'))

    return needed_files


def process_file(file_mp: str) -> tuple[np.ndarray, np.ndarray]:
    """Process a single .npz file and return 2D numpy arrays of counts."""
    data = np.load(file_mp)
    lon_arr = np.round(data['lon'].flatten(), 1)
    lat_arr = np.round(data['lat'].flatten(), 1)
    pixel_arr = np.round(data['pixels'][0, :, :, 0].flatten(), 1)

    # Use NumPy's histogram2d for a massive speedup over pd.crosstab
    # Note: np.histogram2d requires bins to be monotonically increasing
    # Use np.linspace to guarantee exact edge counts.
    # We need 124 edges for 123 bins (centers -141 to -19)
    lon_bins = np.linspace(-141.5, -18.5, 124)

    # We need 29 edges for 28 bins (centers 4 to 31)
    # np.histogram2d needs them in ascending order
    lat_bins = np.linspace(3.5, 31.5, 29)

    total_counts, _, _ = np.histogram2d(lat_arr, lon_arr, bins=[lat_bins, lon_bins])

    mask = pixel_arr > PIXEL_THRESHOLD
    pixel_counts, _, _ = np.histogram2d(lat_arr[mask], lon_arr[mask], bins=[lat_bins, lon_bins])

    # Flip vertically to match your original descending latitude index (31 to 4)
    return np.flipud(pixel_counts), np.flipud(total_counts)


def run_parallel(file_list: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run multiprocessing over all needed files and sum results."""
    # Accumulate directly in NumPy arrays for speed
    pixel_total = np.zeros((len(LAT_RANGE), len(LON_RANGE)))
    tc_total = np.zeros((len(LAT_RANGE), len(LON_RANGE)))

    with Pool(NUM_WORKERS) as pool:
        for i, (px_arr, tc_arr) in enumerate(pool.imap_unordered(process_file, file_list)):
            if i % 10 == 0:  # Print less frequently to avoid I/O bottlenecks
                print(f"Processing file {i + 1} of {len(file_list)}")
            pixel_total += px_arr
            tc_total += tc_arr

    # Convert to DataFrames only at the very end
    pixel_df = pd.DataFrame(pixel_total, index=LAT_RANGE, columns=LON_RANGE)
    total_df = pd.DataFrame(tc_total, index=LAT_RANGE, columns=LON_RANGE)
    return pixel_df, total_df


def plot_heatmaps(pixel_df: pd.DataFrame, total_df: pd.DataFrame, output_file: str):
    """Plot original, normalized, and sample count heatmaps."""
    extent = [-141, -19, 4, 31]

    # FIX: Use layout='constrained' and an aspect-appropriate figsize
    fig, axes = plt.subplots(
        3, 1,
        figsize=(10, 8),
        subplot_kw={'projection': ccrs.PlateCarree()},
        layout='constrained'
    )

    fig.suptitle(
        'Heatmap of the number of CB Pixels from 2019–2023\n'
        'from Atlantic and Eastern Pacific Tropical Cyclones',
        fontsize=16
    )

    # -------------------------
    # Panel A: Raw CB counts
    # -------------------------
    im1 = axes[0].imshow(
        pixel_df.to_numpy(), cmap='gist_heat', origin='lower',
        extent=extent, transform=ccrs.PlateCarree()
    )
    axes[0].set_title('Raw CB Pixel Counts', fontsize=14)

    # -------------------------
    # Panel B: Normalized %
    # -------------------------
    # Prevent division by zero runtime warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = np.nan_to_num((pixel_df.to_numpy() / total_df.to_numpy()) * 100)

    im2 = axes[1].imshow(
        normalized, cmap='gist_heat', origin='lower',
        extent=extent, transform=ccrs.PlateCarree()
    )
    axes[1].set_title('Normalized (% of Pixels that are CBs)', fontsize=14)

    # -------------------------
    # Panel C: Sample Count (Denominator)
    # -------------------------
    im3 = axes[2].imshow(
        total_df.to_numpy(), cmap='gist_heat', origin='lower',
        extent=extent, transform=ccrs.PlateCarree()
    )
    axes[2].set_title('Total Pixel Samples per Bin', fontsize=14)

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

        gl = ax.gridlines(draw_labels=True, linewidth=1, color='white', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator([-140, -120, -100, -80, -60, -40, -20])
        gl.ylocator = mticker.FixedLocator([0, 10, 20, 30])
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

    # -------------------------
    # Colorbars (Horizontal)
    # -------------------------
    # shrink=0.55 prevents the colorbar from forcing the subplot to be too wide
    # aspect=30 makes the colorbar thinner vertically
    # pad controls the distance between the map and its colorbar

    cbar1 = fig.colorbar(im1, ax=axes[0], orientation="horizontal", shrink=0.55, aspect=30, pad=0.02)
    cbar1.set_label('Number of CB Pixels', fontsize=12)

    cbar2 = fig.colorbar(im2, ax=axes[1], orientation="horizontal", shrink=0.55, aspect=30, pad=0.02)
    cbar2.set_label('% of Pixels that are CBs', fontsize=12)

    cbar3 = fig.colorbar(im3, ax=axes[2], orientation="horizontal", shrink=0.55, aspect=30, pad=0.02)
    cbar3.set_label('Total Number of Pixel Samples', fontsize=12)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))

    cbar1.ax.xaxis.set_major_formatter(formatter)
    cbar2.ax.xaxis.set_major_formatter(formatter)
    cbar3.ax.xaxis.set_major_formatter(formatter)

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
    plot_heatmaps(pixel_df, tc_df, "CB_heatmap.png")