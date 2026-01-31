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
from matplotlib.ticker import FuncFormatter
from scipy.stats import mannwhitneyu
from shear_multi import mp_running, init_worker


# ----------------------------
# Axis labeling functions
# ----------------------------
def label_every_10(x, pos):
    """Return tick label every 10 units."""
    _ = pos  # Adding dummy variable to ensure pos not used warning is not present.
    return f"{int(x)}" if x % 10 == 0 else ''


def label_every_20(x: int, pos) -> str:
    """
    Tick Formatter
    :param x: Number of ticks
    :param pos: Position
    :return: Tick labels
    """
    _ = pos  # Adding dummy variable to ensure pos not used warning is not present.
    """Return tick label every 20 units."""
    return f"{int(x)}" if x % 20 == 0 else ''


# ----------------------------
# Data grouping and cleaning
# ----------------------------
def group_func(data: list[float], group_labels: list[int]) -> tuple[list[list[float]], list[list[int]]]:
    """
    Group data by labels and convert to percentage.

    :param data: Values to be grouped
    :param group_labels: Labels corresponding to values
    :return: Tuple of (grouped data in percent, grouped labels)
    """
    grouped = defaultdict(list)
    for val, label in zip(data, group_labels):
        grouped[label].append((val / (1024 * 1024)) * 100)

    final_data = [grouped[k] for k in sorted(grouped.keys())]
    final_labels = [[k] * len(grouped[k]) for k in sorted(grouped.keys())]

    # Debug: log high percentages
    flat = [item for sublist in final_data for item in sublist]
    if flat and np.max(flat) > 60:
        print(f'High final data: {final_data}')

    return final_data, final_labels


def clean_func(list1: list, list2: list) -> tuple[list, list]:
    """
    Remove None values from two parallel lists.

    :param list1: First list
    :param list2: Second list
    :return: Cleaned lists
    """
    valid_indices = [i for i, (v1, v2) in enumerate(zip(list1, list2)) if v1 is not None and v2 is not None]
    return [list1[i] for i in valid_indices], [list2[i] for i in valid_indices]

def fig_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)


# ----------------------------
# Violin/Box plot helpers
# ----------------------------
def adjacent_values(sorted_array: np.ndarray, q1: float, q3: float) -> tuple[float, float]:
    """Compute whisker limits for violin/box plots."""
    iqr = q3 - q1
    upper = min(max(sorted_array[sorted_array <= q3 + 1.5 * iqr], default=q3), max(sorted_array))
    lower = max(min(sorted_array[sorted_array >= q1 - 1.5 * iqr], default=q1), min(sorted_array))
    return lower, upper


def plot_violin_2panel_diurnal(data1, labels1, data2, labels2, suptitle, xlabel, ylabel, filename):
    """
    Plot two side-by-side violin plots: violins + quartiles from 3-hour bins, hourly medians as a line.
    """

    def plot_single_violin(ax: plt.axis, data: list, labels: list, panel_title: str):
        # --- Bin data into 3-hour intervals ---
        bin_centers = list(range(0, 24, 3))  # 0, 3, ..., 21
        bins = defaultdict(list)
        for group, label_group in zip(data, labels):
            for val, hr in zip(group, label_group):
                binned_hr = (3 * (hr // 3)) % 24  # ensures 24 → 0 bin
                bins[binned_hr].append(val)

        # --- Prepare violin data ---
        violin_positions = np.array(sorted(bins.keys()), dtype=float)
        violin_data = [np.asarray(bins[bc], dtype=float) for bc in violin_positions]

        # Compute quartiles and whiskers for each group manually
        quartile1, medians, quartile3 = [], [], []
        whiskers_min, whiskers_max = [], []

        for group in violin_data:
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
        ax.tick_params(axis='both', labelsize=14)
        parts = ax.violinplot(
            violin_data, positions=violin_positions, showmeans=False,
            showmedians=False, showextrema=False, widths=2.5
        )
        for pc in parts['bodies']:
            pc.set_facecolor('#FFA500')
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        ax.scatter(violin_positions, medians, color='blue', s=80, zorder=3)
        ax.vlines(violin_positions, quartile1, quartile3, color='k', lw=2)
        ax.vlines(violin_positions, whiskers_min, whiskers_max, color='k', lw=1)

        # --- Formatting ---
        ax.set_xlim((-2, 24))
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels(np.arange(0, 24, 3), rotation=45, ha='right')
        ax.set_ylim((0, 60))
        ax.set_title(panel_title, fontsize=16)
        ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)

        # --- Sample counts ---
        for pos, values in zip(violin_positions, violin_data):
            if len(values) == 0:
                continue
            y = min(max(values) + 2, 60)
            ax.text(
                pos, y, f'{len(values)}', ha='center', fontsize=14, fontweight='bold',
                rotation=90, va='center',
                path_effects=[path_effects.Stroke(linewidth=2, foreground='white'),
                              path_effects.Normal()]
            )

    # --- Create horizontal 2-panel plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plot_single_violin(axes[0], data1, labels1, 'Atlantic')
    plot_single_violin(axes[1], data2, labels2, 'Eastern Pacific')

    fig.suptitle(suptitle, fontsize=20)
    fig.supxlabel(xlabel, fontsize=20)
    fig.supylabel(ylabel, fontsize=20, x=0.05)
    plt.savefig(filename)
    return fig, axes


def plot_contourf_2panel_diurnal(data1: list, labels1: list, data2: list, labels2: list, suptitle: str, xlabel: str,
                         ylabel: str, filename: str):
    """
    Plot p-values after binning data into 3-hour intervals.
    """

    def bin_data_by_3hr(data: list, labels: list):
        """
        Bin data into 3-hourly bins centered on 0, 3, ..., 21.
        The 24-hour bin wraps into 0.
        """
        bins = defaultdict(list)  # key: (x_bin, y_bin) => list of values

        for group, label_group in zip(data, labels):
            for val, (x_hr, y_hr) in zip(group, zip(label_group, label_group)):
                # Wrap 24 -> 0
                x_bin = (int(x_hr) % 24) // 3 * 3
                y_bin = (int(y_hr) % 24) // 3 * 3
                bins[(x_bin, y_bin)].append(val)

        bin_centers = list(range(0, 24, 3))
        grid = [[bins.get((x, y), []) for x in bin_centers] for y in bin_centers]
        return grid, bin_centers

    # Bin and compute p-value grids
    binned1, centers1 = bin_data_by_3hr(data1, labels1)
    binned2, centers2 = bin_data_by_3hr(data2, labels2)

    pvals1 = compute_pval_grid(binned1)
    pvals2 = compute_pval_grid(binned2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Common tick marks
    tick_marks = range(0, 24, 3)

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
    axes[0].set_xlim((-1, 24))
    axes[0].set_ylim((-1, 24))
    axes[0].set_xticks(tick_marks)
    axes[0].set_yticks(tick_marks)
    axes[0].set_xticklabels(tick_marks, rotation=45, ha='right')
    axes[0].set_yticklabels(tick_marks)
    axes[0].set_title('Atlantic', fontsize=16)
    axes[0].grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)

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
    axes[1].set_xlim((-1, 24))
    axes[1].set_ylim((-1, 24))
    axes[1].set_xticks(tick_marks)
    axes[1].set_yticks(tick_marks)
    axes[1].set_xticklabels(tick_marks, rotation=45, ha='right')
    axes[1].set_yticklabels(tick_marks)
    axes[1].set_title('Eastern Pacific', fontsize=16)
    axes[1].grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)

    # Final layout
    fig.subplots_adjust(bottom=0.25)
    fig.suptitle(suptitle, fontsize=20, y=0.95)
    fig.supxlabel(xlabel, fontsize=20, y=0.15)
    fig.supylabel(ylabel, fontsize=20, x=0.05)

    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=16)
    plt.savefig(filename)
    return fig, axes

def plot_violin_2panel_intensity(data1, labels1, data2, labels2, suptitle, xlabel, ylabel, filename):
    """
    Plot two side-by-side violin plots: violins + quartiles from 3-hour bins, hourly medians as a line.
    """

    def plot_single_violin(ax: plt.axis, data: list, labels: list, panel_title: str):
        # --- Bin data into 3-hour intervals ---
        bin_centers = list(range(20, 161, 10))  # 0, 3, ..., 21
        bins = defaultdict(list)
        for group, label_group in zip(data, labels):
            for val, lab in zip(group, label_group):
                binned_lab = (10 * (lab // 10))  # ensures 24 → 0 bin
                bins[binned_lab].append(val)

        # --- Prepare violin data ---
        violin_positions = np.array(sorted(bins.keys()), dtype=float)
        violin_data = [np.asarray(bins[bc], dtype=float) for bc in violin_positions]

        # Compute quartiles and whiskers for each group manually
        quartile1, medians, quartile3 = [], [], []
        whiskers_min, whiskers_max = [], []

        for group in violin_data:
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
        ax.tick_params(axis='both', labelsize=14)
        parts = ax.violinplot(
            violin_data, positions=violin_positions, showmeans=False,
            showmedians=False, showextrema=False, widths=2.5
        )
        for pc in parts['bodies']:
            pc.set_facecolor('#FFA500')
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        ax.scatter(violin_positions, medians, color='blue', s=80, zorder=3)
        ax.vlines(violin_positions, quartile1, quartile3, color='k', lw=2)
        ax.vlines(violin_positions, whiskers_min, whiskers_max, color='k', lw=1)

        # --- Formatting ---
        ax.set_xlim((15, 165))
        ax.set_xticks(range(20, 161, 10))
        ax.set_xticklabels(np.arange(20, 161, 10), rotation=45, ha='right')
        ax.set_ylim((0, 60))
        ax.set_title(panel_title, fontsize=16)
        ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)

        # --- Sample counts ---
        for pos, values in zip(violin_positions, violin_data):
            if len(values) == 0:
                continue
            y = min(max(values) + 2, 60)
            ax.text(
                pos, y, f'{len(values)}', ha='center', fontsize=14, fontweight='bold',
                rotation=90, va='center',
                path_effects=[path_effects.Stroke(linewidth=2, foreground='white'),
                              path_effects.Normal()]
            )

    # --- Create horizontal 2-panel plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plot_single_violin(axes[0], data1, labels1, 'Atlantic')
    plot_single_violin(axes[1], data2, labels2, 'Eastern Pacific')

    fig.suptitle(suptitle, fontsize=20)
    fig.supxlabel(xlabel, fontsize=20)
    fig.supylabel(ylabel, fontsize=20, x=0.05)
    plt.savefig(filename)
    return fig, axes


def plot_contourf_2panel_intensity(data1: list, labels1: list, data2: list, labels2: list, suptitle: str, xlabel: str,
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
                x_bin = int(x_hr) // 10 * 10
                y_bin = int(y_hr)// 10 * 10
                bins[(x_bin, y_bin)].append(val)

        bin_centers = list(range(20, 161, 10))
        grid = [[bins.get((x, y), []) for x in bin_centers] for y in bin_centers]
        return grid, bin_centers

    # Bin and compute p-value grids
    binned1, centers1 = bin_data(data1, labels1)
    binned2, centers2 = bin_data(data2, labels2)

    pvals1 = compute_pval_grid(binned1)
    pvals2 = compute_pval_grid(binned2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Common tick marks
    tick_marks = range(20, 161, 10)

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
    axes[0].set_xlim((15, 161))
    axes[0].set_ylim((15, 165))
    axes[0].set_xticks(tick_marks)
    axes[0].set_yticks(tick_marks)
    axes[0].set_xticklabels(tick_marks, rotation=45, ha='right')
    axes[0].set_yticklabels(tick_marks)
    axes[0].set_title('Atlantic', fontsize=16)
    axes[0].grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)

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
    axes[1].set_xlim((15, 165))
    axes[1].set_ylim((15, 165))
    axes[1].set_xticks(tick_marks)
    axes[1].set_yticks(tick_marks)
    axes[1].set_xticklabels(tick_marks, rotation=45, ha='right')
    axes[1].set_yticklabels(tick_marks)
    axes[1].set_title('Eastern Pacific', fontsize=16)
    axes[1].grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)

    # Final layout
    fig.subplots_adjust(bottom=0.25)
    fig.suptitle(suptitle, fontsize=20, y=0.95)
    fig.supxlabel(xlabel, fontsize=20, y=0.15)
    fig.supylabel(ylabel, fontsize=20, x=0.05)

    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=16)
    plt.savefig(filename)
    return fig, axes


def compute_pval_grid(grid: list[list[list[float]]]) -> np.ndarray:
    """
    Compute Mann-Whitney p-values for a 2D grid of lists-of-values.
    Returns a np.array of shape (n_bins, n_bins).
    """
    n = len(grid)
    pvals = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(n):
            data_i = grid[i]
            data_j = grid[j]
            # flatten row into single list of values
            vals_i = [v for cell in data_i for v in (cell if isinstance(cell, list) else [cell])]
            vals_j = [v for cell in data_j for v in (cell if isinstance(cell, list) else [cell])]

            if vals_i and vals_j:  # both non-empty
                _, p = mannwhitneyu(vals_i, vals_j, alternative='two-sided')
                pvals[i, j] = p
    return pvals


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
    results_al = {k: [] if k != 'wind_change' else {f'{h:+}': {'pixel': [], 'count': []} for h in
                                                    [-24, -18, -12, -6, 6, 12, 18, 24]} for k in results}
    results_ep = {k: [] if k != 'wind_change' else {f'{h:+}': {'pixel': [], 'count': []} for h in
                                                    [-24, -18, -12, -6, 6, 12, 18, 24]} for k in results}
    for idx, storm_id in enumerate(results['id_list']):
        target = results_al if 'AL' in storm_id else results_ep if 'EP' in storm_id else None
        if target:
            target['id_list'].append(storm_id)
            for key in ['diurnal_pixel', 'diurnal_count', 'intensity_pixel', 'intensity_count']:
                target[key].append(results[key][idx])
            for h in target['wind_change']:
                target['wind_change'][h]['pixel'].append(results['wind_change'][h]['pixel'][idx])
                target['wind_change'][h]['count'].append(results['wind_change'][h]['count'][idx])
    return results_al, results_ep


results_al, results_ep = split_basin(results)


# ================= Clean data =================
def clean_all(results: dict) -> dict:
    """
    Function to run help clean function
    :param results: Analysis results
    :return: Cleaned analysis results
    """
    results['diurnal_pixel'], results['diurnal_count'] = clean_func(results['diurnal_pixel'],
                                                                    results['diurnal_count'])
    results['intensity_pixel'], results['intensity_count'] = clean_func(results['intensity_pixel'],
                                                                        results['intensity_count'])
    for h in results['wind_change']:
        results['wind_change'][h]['pixel'], results['wind_change'][h]['count'] = clean_func(
            results['wind_change'][h]['pixel'], results['wind_change'][h]['count'])
    return results


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
        'diurnal_pixel': [], 'diurnal_count': [],
        'intensity_pixel': [], 'intensity_count': [],
        'wind_change': {f'{h:+}': {'pixel': [], 'count': []} for h in [-24, -18, -12, -6, 6, 12, 18, 24]},
        'id_list': []
    }

    # Parallel processing
    with Pool(24, initializer=init_worker) as pool:
        for result in pool.map(mp_running, needed_args):
            if result is None:
                continue
            diurnal_count, intensity_count, pixels, atcf_id = result

            # Append results
            results['diurnal_pixel'].append(pixels)
            results['diurnal_count'].append(diurnal_count)
            results['intensity_pixel'].append(pixels)
            results['intensity_count'].append(int(intensity_count))

            results['id_list'].append(atcf_id)

    results = clean_all(results)
    results_al = clean_all(results_al)
    results_ep = clean_all(results_ep)

    # ==================== Violin & Contour Plots for Basins =====================
    data_al, labels_al = group_func(results_al['diurnal_pixel'], results_al['diurnal_count'])
    data_ep, labels_ep = group_func(results_ep['diurnal_pixel'], results_ep['diurnal_count'])
    fig1, _ = plot_violin_2panel_diurnal(data_al, labels_al, data_ep, labels_ep,
                       suptitle='TCB Occurrences vs Diurnal Cycle Stage',
                       xlabel='Local Solar Time',
                       ylabel='Percentage of Storm Pixels with TCBs',
                       filename='diurnal_violin_ALEP.png')
    fig2, _ = plot_contourf_2panel_diurnal(data_al, labels_al, data_ep, labels_ep,
                         suptitle='Diurnal Cycle Mann-Whitney P-Values',
                         xlabel='Local Solar Time',
                         ylabel='Local Solar Time',
                         filename='diurnal_mannwhitney_ALEP.png')

    img1 = fig_to_rgb(fig1)
    img2 = fig_to_rgb(fig2)

    fig, ax = plt.subplots(
        figsize=(img1.shape[1] / 100, (img1.shape[0] + img2.shape[0]) / 100)
    )

    ax.imshow(np.vstack([img1, img2]))
    ax.axis("off")

    plt.savefig("diurnal_combined.png", dpi=300, bbox_inches="tight")
    plt.close('all')

    data_al, labels_al = group_func(results_al['intensity_pixel'], results_al['intensity_count'])
    data_ep, labels_ep = group_func(results_ep['intensity_pixel'], results_ep['intensity_count'])
    fig1, _ = plot_violin_2panel_intensity(data_al, labels_al, data_ep, labels_ep,
                       suptitle='TCB Occurrences vs TC Intensity',
                       xlabel='TC Current Wind Speed (kts)',
                       ylabel='Percentage of Storm Pixels with TCBs',
                       filename='intensity_violin_ALEP.png')
    fig2, _ = plot_contourf_2panel_intensity(data_al, labels_al, data_ep, labels_ep,
                         suptitle='TC Intensity Mann-Whitney P-Values',
                         xlabel='Intensity (kts)',
                         ylabel='Intensity (kts)',
                         filename='intensity_mannwhitney_ALEP.png')

    img1 = fig_to_rgb(fig1)
    img2 = fig_to_rgb(fig2)

    fig, ax = plt.subplots(
        figsize=(img1.shape[1] / 100, (img1.shape[0] + img2.shape[0]) / 100)
    )

    ax.imshow(np.vstack([img1, img2]))
    ax.axis("off")

    plt.savefig("intensity_combined.png", dpi=300, bbox_inches="tight")
    plt.close()