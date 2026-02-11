# coding=utf-8
"""
Last Edited: 10/02/2025
@author: John Mark Mayhall
Purpose: Identify transverse bands in TC quadrants based on shear vector.
"""
import glob
from collections import defaultdict
from multiprocessing import Pool

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from scipy.stats import mannwhitneyu
from shear_multi import mp_running, init_worker


def group_func(data: list, group_labels: list):
    """
    Group pixel data by rounded intensity change bins.

    Converts raw pixel counts to percentage of domain (1024x1024).
    Returns:
        grouped_data : list of lists (pixel % per bin)
        grouped_labels : list of lists (bin labels)
    """
    df = pd.DataFrame({
        "label": group_labels,
        "data": data
    }).dropna()

    # Convert to % once (vectorized)
    df["data"] = (df["data"] / (1024 * 1024)) * 100

    grouped = df.groupby("label")["data"].apply(list).sort_index()

    return grouped.tolist(), [[k] * len(v) for k, v in grouped.items()]


def compute_pval_grid(data: list) -> np.ndarray:
    """
    Compute symmetric Mann-Whitney U p-value matrix.
    Only computes upper triangle and mirrors.
    """
    n = len(data)
    p_grid = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i, n):
            _, p = mannwhitneyu(data[i], data[j], alternative='two-sided')
            p_grid[i, j] = p
            p_grid[j, i] = p

    return p_grid


def clean_func(list1: list, list2: list) -> list:
    """
    Function for removing None values from list
    :param list1: First list to be cleaned
    :param list2: Second list to be cleaned
    :return: The two cleaned lists
    """
    none_indices = ([i for i, v in enumerate(list1) if v is None] +
                    [i for i, v in enumerate(list2) if v is None])

    # Remove None values
    cleaned_list1 = [v for i, v in enumerate(list1) if i not in none_indices]
    cleaned_list2 = [v for i, v in enumerate(list2) if i not in none_indices]
    return cleaned_list1, cleaned_list2

def split_basin(results: dict):
    """
    Function to split analysis results by basin.
    :param results: Analysis results
    :return: Split Results
    """
    results_al = {k: [] for k in results}
    results_ep = {k: [] for k in results}
    for idx, storm_id in enumerate(results['id_list']):
        target = results_al if 'AL' in storm_id else results_ep if 'EP' in storm_id else None
        if target:
            target['id_list'].append(storm_id)
            for key in ['shear_pixel', 'shear_count']:
                target[key].append(results[key][idx])
    return results_al, results_ep


# ================= Clean data =================
def clean_all(results: dict) -> dict:
    """
    Function to run help clean function
    :param results: Analysis results
    :return: Cleaned analysis results
    """
    results['shear_pixel'], results['shear_count'] = clean_func(results['shear_pixel'],
                                                                        results['shear_count'])
    return results


def adjacent_values(sorted_array: np.ndarray, q1: float, q3: float) -> tuple[float, float]:
    """Compute whisker limits for violin/box plots."""
    iqr = q3 - q1
    upper = min(max(sorted_array[sorted_array <= q3 + 1.5 * iqr], default=q3), max(sorted_array))
    lower = max(min(sorted_array[sorted_array >= q1 - 1.5 * iqr], default=q1), min(sorted_array))
    return lower, upper


def plot_violin_2panel_shear(data1, labels1, data2, labels2, suptitle, xlabel, ylabel, filename):
    """
    Plot two side-by-side violin plots: violins + quartiles from 3-hour bins, hourly medians as a line.
    """

    def plot_single_violin(ax: plt.axis, data: list, labels: list, panel_title: str):
        # --- Bin data into 3-hour intervals ---
        bin_centers = list(range(0, 31, 5))  # 0, 3, ..., 21
        bins = defaultdict(list)
        for group, label_group in zip(data, labels):
            for val, lab in zip(group, label_group):
                binned_lab = 5 * np.round(lab / 5)  # ensures 24 → 0 bin
                bins[binned_lab].append(val)

        # --- Prepare violin data ---
        violin_positions = np.array(sorted(bins.keys()), dtype=float)
        violin_data = [np.asarray(bins[bc], dtype=float) for bc in violin_positions]

        # Compute quartiles and whiskers for each group manually
        quartile1, medians, quartile3 = [], [], []
        whiskers_min, whiskers_max = [], []

        for group in violin_data:
            group = np.asarray(group, dtype=float)
            group = group[np.isfinite(group)]

            if group.size == 0:
                quartile1.append(np.nan)
                medians.append(np.nan)
                quartile3.append(np.nan)
                whiskers_min.append(np.nan)
                whiskers_max.append(np.nan)
                continue

            q1, med, q3 = np.percentile(group, [25, 50, 75])
            whisk_min, whisk_max = adjacent_values(group, q1, q3)

            quartile1.append(float(q1))
            medians.append(float(med))
            quartile3.append(float(q3))
            whiskers_min.append(float(whisk_min))
            whiskers_max.append(float(whisk_max))

        # --- Plot violins ---
        ax.tick_params(axis='both', labelsize=18)
        parts = ax.violinplot(
            violin_data, positions=violin_positions, showmeans=False,
            showmedians=False, showextrema=False, widths=3
        )
        for pc in parts['bodies']:
            pc.set_facecolor('#FFA500')
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        ax.scatter(violin_positions, medians, color='blue', s=80, zorder=3)
        ax.vlines(violin_positions, quartile1, quartile3, color='k', lw=2)
        ax.vlines(violin_positions, whiskers_min, whiskers_max, color='k', lw=1)

        # --- Formatting ---
        ax.set_xlim((-5, 35))
        ax.set_xticks(range(0, 31, 5))
        ax.set_ylim((0, 60))
        ax.set_title(panel_title, fontsize=28)
        ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)

        # --- Sample counts ---
        for pos, values in zip(violin_positions, violin_data):
            if len(values) == 0:
                continue
            y = min(max(values) + 2, 60)
            ax.text(
                pos, y, f'{len(values)}', ha='center', fontsize=18, fontweight='bold',
                rotation=90, va='center',
                path_effects=[path_effects.Stroke(linewidth=2, foreground='white'),
                              path_effects.Normal()]
            )

    # --- Create horizontal 2-panel plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plot_single_violin(axes[0], data1, labels1, 'Atlantic')
    plot_single_violin(axes[1], data2, labels2, 'Eastern Pacific')

    fig.suptitle(suptitle, fontsize=28, y=0.98)
    fig.supxlabel(xlabel, fontsize=25, y=0)
    fig.supylabel(ylabel, fontsize=25, x=0.05)
    plt.savefig(filename)
    return fig, axes


def plot_contourf_2panel_shear(data1: list, labels1: list, data2: list, labels2: list, suptitle: str, xlabel: str,
                         ylabel: str, filename: str):
    """
    Plot p-values after binning data into 3-hour intervals.
    """

    def bin_data(data: list, labels: list):
        """
        Bin data into 3-hourly bins centered on 0, 3, ..., 21.
        The 24-hour bin wraps into 0.
        """
        bins = defaultdict(list)  # key: (x_bin, y_bin) => list of values

        for group, label_group in zip(data, labels):
            for val, (x_lab, y_lab) in zip(group, zip(label_group, label_group)):
                x_bin = 5 * np.round(x_lab / 5)
                y_bin = 5 * np.round(y_lab / 5)
                bins[(x_bin, y_bin)].append(val)

        bin_centers = list(range(0, 31, 5))
        grid = [[bins.get((x, y), []) for x in bin_centers] for y in bin_centers]
        return grid, bin_centers

    # Bin and compute p-value grids
    binned1, centers1 = bin_data(data1, labels1)
    binned2, centers2 = bin_data(data2, labels2)

    pvals1 = compute_pval_grid(binned1)
    pvals2 = compute_pval_grid(binned2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Common tick marks
    tick_marks = range(0, 31, 5)

    # Left panel (Atlantic)
    x, y = np.meshgrid(centers1, centers1)
    z = np.array(pvals1)
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = z.ravel()
    mask = z_flat < 0.05
    x_masked = x_flat[mask]
    y_masked = y_flat[mask]
    axes[0].scatter(x_masked, y_masked, c='black', s=200, label='p < 0.05')
    axes[0].set_xlim((-5, 35))
    axes[0].set_ylim((-5, 35))
    axes[0].set_xticks(tick_marks)
    axes[0].set_yticks(tick_marks)
    axes[0].set_yticklabels(tick_marks)
    axes[0].set_title('Atlantic', fontsize=28)
    axes[0].grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)
    axes[0].tick_params(axis='both', labelsize=18)

    # Right panel (Eastern Pacific)
    x, y = np.meshgrid(centers2, centers2)
    z = np.array(pvals2)
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = z.ravel()
    mask = z_flat < 0.05
    x_masked = x_flat[mask]
    y_masked = y_flat[mask]
    axes[1].scatter(x_masked, y_masked, c='black', s=200, label='p < 0.05')
    axes[1].set_xlim((-5, 35))
    axes[1].set_ylim((-5, 35))
    axes[1].set_xticks(tick_marks)
    axes[1].set_yticks(tick_marks)
    axes[1].set_yticklabels(tick_marks)
    axes[1].set_title('Eastern Pacific', fontsize=28)
    axes[1].grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)
    axes[1].tick_params(axis='both', labelsize=18)

    # Final layout
    fig.subplots_adjust(bottom=0.25)
    fig.suptitle(suptitle, fontsize=28, y=0.98)
    fig.supxlabel(xlabel, fontsize=28, y=0.13)
    fig.supylabel(ylabel, fontsize=28, x=0.05)

    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=20, framealpha=0)
    plt.savefig(filename)
    return fig, axes


if __name__ == '__main__':
    # ======================== Config ========================
    cut_off = 0.02
    ri_thresh, rw_thresh = 30, -20
    pix_threshold = min_wind_threshold = -np.inf
    max_wind_threshold = np.inf
    online, save_state = True, False
    base_dir = '/rstor/jmayhall/' if online else '//uahdata/rstor/'

    paths = {
        'latlon': f'{base_dir}cataloging/nc_process/geojson_transform/completed_arrays/latlon_arrs/*.npz',
        'hurdat': f'{base_dir}cataloging/hurdat_update.txt',
        'model': f'{base_dir}Model_Training_Code/cnn_creation',
        'c8': f'{base_dir}cataloging/nc_process/geojson_transform/completed_arrays/C08_scaled/*',
        'c13': f'{base_dir}cataloging/nc_process/geojson_transform/completed_arrays/C13_scaled/*',
        'ships': f'{base_dir}cataloging/nc_process/violin_plots/shear_process_all/ships_interp.txt'
    }

    # ===================== Load Data ========================
    hurdat_df = pd.read_csv(paths['hurdat'], sep='\t')
    latlon_arrays = pd.DataFrame({'Name': glob.glob(paths['latlon'])})
    c8_scaled = pd.DataFrame({'Name': glob.glob(paths['c8'])})
    c13_scaled = glob.glob(paths['c13'])
    path_len = len(paths['c13']) - 1
    wind_dict = pd.read_csv(paths['ships'], sep='\t', index_col=0)

    # Normalize for plotting
    norm = mpl.colors.Normalize(vmin=0.049, vmax=1)
    sm = ScalarMappable(cmap='gist_ncar', norm=norm)
    sm.set_array([])

    # =================== Parallel Args =======================
    needed_args = [{
        'i': i,
        'file': file,
        'path_len': path_len,
        'paths': paths,
        'c8_scaled': c8_scaled
    } for i, file in enumerate(c13_scaled)]

    # ================== Collect Results =======================
    results = {
        'shear_pixel': [], 'shear_count': [],
        'id_list': []
    }

    # Parallel processing
    with Pool(12, initializer=init_worker) as pool:
        for result in pool.map(mp_running, needed_args):
            if result is None:
                continue
            shear_count, pixels, atcf_id = result

            # Append results
            results['shear_pixel'].append(pixels)
            results['shear_count'].append(shear_count)

            results['id_list'].append(atcf_id)

    results_al, results_ep = split_basin(results)
    results = clean_all(results)
    results_al = clean_all(results_al)
    results_ep = clean_all(results_ep)

    # ==================== Violin & Contour Plots for Basins =====================
    data_al, labels_al = group_func(results_al['shear_pixel'], results_al['shear_count'])
    data_ep, labels_ep = group_func(results_ep['shear_pixel'], results_ep['shear_count'])
    fig1, _ = plot_violin_2panel_shear(data_al, labels_al, data_ep, labels_ep,
                       suptitle='TCB Occurrences vs Vertical Wind Shear',
                       xlabel=r'Vertical Wind Shear ($\frac{m}{s}$)',
                       ylabel='Percentage of Storm Pixels with TCBs',
                       filename='shear_violin_ALEP.png')
    fig2, _ = plot_contourf_2panel_shear(data_al, labels_al, data_ep, labels_ep,
                         suptitle='TC Vertical Wind Shear Mann-Whitney P-Values',
                         xlabel=r'Vertical Wind Shear ($\frac{m}{s}$)',
                         ylabel=r'Vertical Wind Shear ($\frac{m}{s}$)',
                         filename='shear_mannwhitney_ALEP.png')

    img1 = fig_to_rgb(fig1)
    img2 = fig_to_rgb(fig2)

    fig, ax = plt.subplots(
        figsize=(img1.shape[1] / 100, (img1.shape[0] + img2.shape[0]) / 100)
    )

    ax.imshow(np.vstack([img1, img2]))
    ax.axis("off")

    plt.savefig("shear_combined.png", dpi=300, bbox_inches="tight")
    plt.close()