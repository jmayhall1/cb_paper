# coding=utf-8
"""
Optimized: 10/02/2025
Updated: adds proper file-SHIPS matching + median lines + 3 panel vertical split

Processes tropical cyclone shear data:
1. Loads shear NPZ files and SHIPS dataset
2. Matches only valid file-SHIPS pairs (robust merge approach)
3. Computes shear magnitude
4. Splits by intensity regime and basin
5. Plots distributions with medians across 3 vertical panels
"""

import glob

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

if __name__ == '__main__':

    # -----------------------------
    # Configuration
    # -----------------------------
    ONLINE = False
    BASE_DIR = '/rstor/jmayhall/' if ONLINE else '//uahdata/rstor/'
    DATA_DIR = f'{BASE_DIR}cataloging/nc_process/shear_process/shear_process_all/'

    file_list = glob.glob(f'{DATA_DIR}*.npz')
    ships_path = f'{DATA_DIR}ships_interp.txt'

    # -----------------------------
    # Load SHIPS data
    # -----------------------------
    ships_df = pd.read_csv(ships_path, sep='\t', index_col=0)

    # convert SHIPS index to datetime explicitly
    ships_df.index = pd.to_datetime(ships_df.index)

    # -----------------------------
    # Build valid file list (QC step)
    # -----------------------------
    valid_cases = []

    for file in file_list:
        atcf_id = file[-41:-33]
        date = file[-32:-24]
        time = file[-23:-19]

        valid_cases.append(
            (
                atcf_id,
                pd.to_datetime(f'{date}{time}', format='%Y%m%d%H%M')
            )
        )

    valid_cases = pd.DataFrame(valid_cases, columns=['atcf_id', 'time'])
    valid_cases['time'] = pd.to_datetime(valid_cases['time'])

    # -----------------------------
    # Match SHIPS + NPZ (IMPORTANT FIX)
    # -----------------------------
    ships = (
        ships_df.reset_index()
        .rename(columns={'index': 'time'})
        .merge(valid_cases, on=['time', 'atcf_id'], how='inner')
    )

    # -----------------------------
    # Compute shear magnitude
    # -----------------------------
    ships['shear'] = np.sqrt(
        ships['shear_u'] ** 2 + ships['shear_v'] ** 2
    ) * 0.514444

    # Optional QC filter (keep if desired)
    ships = ships.dropna(subset=['shear', 'max_winds'])

    # -----------------------------
    # Intensity Regimes Configuration
    # -----------------------------
    intensity_splits = [
        {"name": "TD/TS (0-63kt)", "title": "TD–TS", "condition": ships['max_winds'] < 64},
        {"name": "Cat 1-2 (64-95 kt)", "title": "CAT 1–2",
         "condition": (ships['max_winds'] >= 64) & (ships['max_winds'] < 96)},
        {"name": "Cat 3-5 (>95 kt)", "title": "CAT 3–5", "condition": ships['max_winds'] >= 96}
    ]

    # -----------------------------
    # Plot
    # -----------------------------
    # Switched to 3 rows, 1 column. sharex=True keeps the x-axis clean
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)
    fig.suptitle("TC Shear Distribution by Intensity", fontsize=20, y=0.98)

    bin_width = 2.5
    bins = np.arange(0, 36, bin_width)

    for i, (ax, split) in enumerate(zip(axes, intensity_splits)):
        # Filter ships data for current intensity regime
        regime_data = ships[split["condition"]]

        # Split by basin
        atl = regime_data[regime_data['atcf_id'].str.startswith('AL')]
        ep = regime_data[regime_data['atcf_id'].str.startswith('EP')]

        basins = [
            (atl, "Atlantic", "tab:blue", "-", 1),
            (ep, "Eastern Pacific", "tab:green", "-", 0.45),
        ]

        for data, label, color, ls, al in basins:
            if len(data) == 0:
                continue

            # Histogram
            ax.hist(
                data["shear"],
                bins=bins,
                density=True,
                color=color,
                alpha=al,
                edgecolor="black",
                linewidth=0.7,
                label=f"{label} Histogram",
            )

            # Median
            median = np.nanmedian(data["shear"])
            ax.axvline(
                median,
                color=color,
                linestyle=ls,
                linewidth=2.5,
                zorder=8,
            )

            # KDE scaled to histogram counts
            if len(data) > 1:  # KDE requires more than 1 data point
                x = np.linspace(0, 35, 500)
                try:
                    kde = gaussian_kde(data["shear"].dropna())
                    ax.plot(
                        x,
                        kde(x),
                        color=color,
                        linestyle=ls,
                        linewidth=3,
                        zorder=10,
                        label=f"{label} KDE",
                        path_effects=[
                            pe.Stroke(linewidth=5, foreground='black'),
                            pe.Normal(),
                        ],
                    )
                except np.linalg.LinAlgError:
                    pass  # Skip KDE if data is completely uniform/singular

        # -----------------------------
        # Formatting per subplot
        # -----------------------------
        ax.set_xlim(0, 35)
        ax.set_title(split["title"], fontsize=16)
        ax.set_ylabel("Probability Density", fontsize=14)

        # Only add the x-axis label to the very bottom subplot
        if i == 2:
            ax.set_xlabel(r"Shear ($m\,s^{-1}$)", fontsize=14)

        ax.grid(True, linestyle="--", linewidth=1)
        ax.set_axisbelow(True)

    # -----------------------------
    # Shared Legend
    # -----------------------------
    legend_lines = [
        Line2D([0], [0], color="tab:blue", linewidth=3, linestyle="-", label="Atlantic KDE / Median"),
        Line2D([0], [0], color="tab:green", linewidth=3, linestyle="-", label="Eastern Pacific KDE / Median"),
        Line2D([0], [0], color="tab:blue", linewidth=8, alpha=0.45, label="Atlantic Histogram"),
        Line2D([0], [0], color="tab:green", linewidth=8, alpha=0.45, label="Eastern Pacific Histogram"),
    ]

    # Adjusted for vertical layout: placed at the bottom, 2 columns so it fits nicely under the plot
    fig.legend(
        handles=legend_lines,
        frameon=True,
        fontsize=12,
        loc="upper right",
        framealpha=1,
        bbox_to_anchor=(0.97, 0.93)
    )

    plt.tight_layout()
    # Add extra padding at the bottom so the legend doesn't get cut off
    plt.subplots_adjust(bottom=0.08)
    plt.savefig("intensity_shear_dist.png", dpi=300, bbox_inches="tight")