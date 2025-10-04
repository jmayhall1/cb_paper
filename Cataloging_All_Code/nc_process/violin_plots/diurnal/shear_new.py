# coding=utf-8
"""
Last Edited: 10/02/2025
@author: John Mark Mayhall
Purpose: Identify transverse bands in TC quadrants based on shear vector.
Optimized version with improved readability, vectorization, and comments.
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


# -------------------------- Utility Functions --------------------------

def group_func(data: list, group_labels: list) -> tuple[list[list[float]], list[list[int]]]:
    """
    Group data by corresponding labels and convert pixel counts to percentages.

    :param data: List of numeric values (pixels).
    :param group_labels: List of group labels corresponding to data.
    :return: (grouped_data, grouped_labels) as nested lists.
    """
    # Sort by labels for consistent grouping
    sorted_pairs = sorted(zip(group_labels, data))
    group_labels, data = zip(*sorted_pairs)

    grouped_data_dict = defaultdict(list)
    grouped_labels_dict = defaultdict(list)

    for label, val in zip(group_labels, data):
        grouped_data_dict[label].append(val)
        grouped_labels_dict[label].append(label)

    keys_sorted = sorted(grouped_data_dict.keys())
    grouped_data = [[(v / (1024 * 1024)) * 100 for v in grouped_data_dict[k]] for k in keys_sorted]
    grouped_labels = [grouped_labels_dict[k] for k in keys_sorted]

    # Debug check for unusually high percentages
    flat_vals = [item for sublist in grouped_data for item in sublist]
    if flat_vals and np.max(flat_vals) > 60:
        print(f'High final data detected: {grouped_data}')

    return grouped_data, grouped_labels


def adjacent_values(sorted_array: np.ndarray, q1: float, q3: float) -> tuple[float, float]:
    """
    Compute adjacent values (whiskers) for box/violin plots.

    :param sorted_array: Sorted numeric array.
    :param q1: First quartile.
    :param q3: Third quartile.
    :return: (lower_whisker, upper_whisker)
    """
    iqr = q3 - q1
    upper_adj = q3 + 1.5 * iqr
    lower_adj = q1 - 1.5 * iqr
    upper = max((x for x in sorted_array if x <= upper_adj), default=q3)
    lower = min((x for x in sorted_array if x >= lower_adj), default=q1)
    return lower, upper


def clean_func(list1: list, list2: list) -> tuple[list, list]:
    """
    Remove None values from two lists simultaneously.

    :param list1: First list.
    :param list2: Second list.
    :return: Cleaned lists.
    """
    valid_indices = [i for i, (v1, v2) in enumerate(zip(list1, list2)) if v1 is not None and v2 is not None]
    return [list1[i] for i in valid_indices], [list2[i] for i in valid_indices]


# -------------------------- Plotting Functions --------------------------

def plot_violin(data: list, labels: list, title: str, xlabel: str, ylabel: str, filename: str):
    """
    Plot a single violin plot with quartiles, whiskers, medians, and sample counts.
    """
    labels_unique = np.unique(np.concatenate([np.array(l) for l in labels])).astype(int)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.tick_params(labelsize=16)
    ax.set_ylim(0, 60)

    showmedians_bool, showmeans_bool = False, False
    parts = ax.violinplot(data, positions=labels_unique, showmeans=showmeans_bool, showmedians=showmedians_bool,
                          showextrema=False,
                          widths=0.8)
    for pc in parts['bodies']:
        pc.set_facecolor('#FFA500')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    # Compute quartiles, whiskers, and medians
    quartile1, medians, quartile3, whiskers_min, whiskers_max = [], [], [], [], []
    for group in data:
        group_sorted = np.sort(group)
        q1, med, q3 = np.percentile(group_sorted, [25, 50, 75])
        w_min, w_max = adjacent_values(group_sorted, q1, q3)
        quartile1.append(q1)
        medians.append(med)
        quartile3.append(q3)
        whiskers_min.append(w_min)
        whiskers_max.append(w_max)

    ax.scatter(labels_unique, medians, color='blue', s=100, zorder=3)
    ax.vlines(labels_unique, quartile1, quartile3, color='k', lw=5)
    ax.vlines(labels_unique, whiskers_min, whiskers_max, color='k', lw=1)

    ax.set_xlim(-2, 24)
    ax.set_xticks(range(0, 24, 1))
    num_list = range(0, 24, 1)
    ax.set_xticklabels([str(item) for item in num_list], rotation=45, ha='right')

    # Display sample counts above violins
    y_offset = 0.05 * (max(max(d) for d in data) - min(min(d) for d in data))
    for i, d in enumerate(data):
        text_y = min(max(d) + y_offset, 100)
        ax.text(labels_unique[i], text_y, f'{len(d)}', ha='center', fontsize=16, fontweight='bold',
                rotation=90, va='center',
                path_effects=[path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.grid(True, linestyle='--', linewidth=1.0, alpha=1)
    plt.savefig(filename)
    plt.close()


def plot_violin_2panel(data1, labels1, data2, labels2, suptitle, xlabel, ylabel, filename):
    """
    Plot side-by-side violin plots for two datasets (e.g., Atlantic vs Eastern Pacific).
    """

    def plot_panel(ax, data: list, labels: list, panel_title: str):
        """
        FUnction to help plot subplot
        :param ax: Matplotlib axis
        :param data: Data to be plotted
        :param labels: Data labels
        :param panel_title: Titles of the subplot panels
        :return: Nothing
        """
        # Bin into 3-hour intervals
        bins = defaultdict(list)
        for group, lbls in zip(data, labels):
            for val, hr in zip(group, lbls):
                bins[(hr // 3) * 3].append(val)

        positions = sorted(bins.keys())
        violin_data = [bins[p] for p in positions]

        parts = ax.violinplot(violin_data, positions=positions, showmeans=False, showmedians=False, showextrema=False,
                              widths=2.5)
        for pc in parts['bodies']:
            pc.set_facecolor('#FFA500')
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        # Quartiles and whiskers
        quartile1, quartile3 = [], []
        whiskers_min, whiskers_max = [], []
        for vals in violin_data:
            if not vals:
                quartile1.append(np.nan)
                quartile3.append(np.nan)
                whiskers_min.append(np.nan)
                whiskers_max.append(np.nan)
                continue
            q1, q3 = np.percentile(vals, [25, 75])
            quartile1.append(q1)
            quartile3.append(q3)
            whiskers_min.append(min(vals))
            whiskers_max.append(max(vals))

        ax.vlines(positions, quartile1, quartile3, color='k', lw=5)
        ax.vlines(positions, whiskers_min, whiskers_max, color='k', lw=1)

        # Hourly medians
        hourly_medians = []
        for hr in range(24):
            vals_hr = [v for g, l in zip(data, labels) for v, h in zip(g, l) if h == hr]
            hourly_medians.append(np.nan if not vals_hr else np.percentile(vals_hr, 50))
        ax.plot(range(24), hourly_medians, color='blue', linewidth=3, marker='o')

        # Formatting
        ax.set_xlim(-2, 24)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels(range(0, 24, 3), rotation=45, ha='right')
        ax.set_ylim(0, 60)
        ax.set_title(panel_title, fontsize=16)
        ax.grid(True, linestyle='--', linewidth=1.0, alpha=1)

        # Sample counts
        for pos, vals in zip(positions, violin_data):
            if vals:
                ax.text(pos, min(max(vals) + 2, 60), f'{len(vals)}', ha='center', fontsize=14, fontweight='bold',
                        rotation=90, va='center', path_effects=[path_effects.Stroke(linewidth=2, foreground='white'),
                                                                path_effects.Normal()])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plot_panel(axes[0], data1, labels1, 'Atlantic')
    plot_panel(axes[1], data2, labels2, 'Eastern Pacific')
    fig.suptitle(suptitle, fontsize=20)
    fig.supxlabel(xlabel, fontsize=20)
    fig.supylabel(ylabel, fontsize=20, x=0.05)
    plt.savefig(filename)
    plt.close()


def compute_pval_grid(grid: list[list[list[float]]]) -> np.ndarray:
    """
    Compute Mann-Whitney U p-values for a 2D grid of lists-of-values.
    """
    n = len(grid)
    pvals = np.full((n, n), np.nan)
    mannwhitney_type = 'two-sided'

    for i in range(n):
        for j in range(n):
            vals_i = [v for cell in grid[i] for v in (cell if isinstance(cell, list) else [cell])]
            vals_j = [v for cell in grid[j] for v in (cell if isinstance(cell, list) else [cell])]
            if vals_i and vals_j:
                _, pvals[i, j] = mannwhitneyu(vals_i, vals_j, alternative=mannwhitney_type)
    return pvals


def plot_contourf_2panel(data1, labels1, data2, labels2, suptitle, xlabel, ylabel, filename):
    """
    Plot Mann-Whitney p-values as 2-panel scatter plots (binned 3-hourly).
    """

    def bin_data_3hr(data: list, labels: list):
        """
        Function to bin data in three hour bins.
        :param data: Data to be binned
        :param labels: Center time labels
        :return: Binned data and centers
        """
        bins = defaultdict(list)
        for g, lbls in zip(data, labels):
            for val, hr in zip(g, lbls):
                bins[(hr // 3) * 3].append(val)
        centers = sorted({k for k in bins})
        grid = [[bins.get((x, y), []) for x in centers] for y in centers]
        return grid, centers

    binned1, centers1 = bin_data_3hr(data1, labels1)
    binned2, centers2 = bin_data_3hr(data2, labels2)
    pvals1, pvals2 = compute_pval_grid(binned1), compute_pval_grid(binned2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    tick_marks = range(0, 24, 3)

    for ax, pvals, centers, title in zip(axes, [pvals1, pvals2], [centers1, centers2], ['Atlantic', 'Eastern Pacific']):
        x, y = np.meshgrid(centers, centers)
        mask = pvals.ravel() < 0.05
        ax.scatter(x.ravel()[mask], y.ravel()[mask], c='black', s=200, label='p < 0.05')
        ax.set_xlim(-1, 24)
        ax.set_ylim(-1, 24)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(tick_marks, rotation=45, ha='right')
        ax.set_yticklabels(tick_marks)
        ax.set_title(title, fontsize=16)
        ax.grid(True, linestyle='--', linewidth=1.0, alpha=1)

    fig.subplots_adjust(bottom=0.25)
    fig.suptitle(suptitle, fontsize=20, y=0.95)
    fig.supxlabel(xlabel, fontsize=20, y=0.15)
    fig.supylabel(ylabel, fontsize=20, x=0.05)
    fig.legend(['p < 0.05'], loc='upper right', fontsize=16)
    plt.savefig(filename)
    plt.close()


def plot_mann_whitney_contour(data, labels, filename, xlabel, ylabel, title, test_type='p'):
    """
    Plot a 2D Mann-Whitney matrix as a scatter of significant values.
    """
    unique_labels = sorted(set([i for sublist in labels for i in sublist]))
    n = len(unique_labels)
    grouped_data = {label[0]: d for label, d in zip(labels, data)}
    mannwhitney_type = 'two-sided'

    stat_matrix = np.zeros((n, n))
    for i, l1 in enumerate(unique_labels):
        for j, l2 in enumerate(unique_labels):
            u_stat, p_value = mannwhitneyu(grouped_data[l1], grouped_data[l2], alternative=mannwhitney_type)
            stat_matrix[i, j] = p_value if test_type == 'p' else u_stat

    x, y = np.meshgrid(unique_labels, unique_labels)
    mask = stat_matrix.ravel() < 0.05

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(x.ravel()[mask], y.ravel()[mask], c='black', s=200, label='p < 0.05')
    ax.set_xlim(-1, 24)
    ax.set_ylim(-1, 24)
    ax.set_xticks(range(0, 24, 1))
    ax.set_yticks(range(0, 24, 1))
    num_list = range(0, 24, 1)
    ax.set_xticklabels([str(item) for item in num_list], rotation=45, ha='right')
    ax.set_xlabel(xlabel, fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    ax.set_title(title, fontsize=17)
    ax.grid(True, linestyle='--', linewidth=1.0, alpha=1)
    ax.legend(loc='upper right', fontsize=16)
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    # -------------------------- Configuration --------------------------
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

    # -------------------------- Load Data --------------------------

    hurdat_df = pd.read_csv(paths['hurdat'], sep='\t')
    latlon_arrays = pd.DataFrame({'Name': glob.glob(paths['latlon'])})
    c8_scaled = pd.DataFrame({'Name': glob.glob(paths['c8'])})
    c13_scaled = glob.glob(paths['c13'])
    path_len = len(paths['c13']) - 1
    wind_dict = pd.read_csv(paths['ships'], sep='\t', index_col=0)

    # Color mapping
    norm = mpl.colors.Normalize(vmin=0.049, vmax=1)
    sm = ScalarMappable(cmap='gist_ncar', norm=norm)
    sm.set_array([])

    # -------------------------- Parallel Processing --------------------------

    needed_args = [{'i': i, 'file': file, 'path_len': path_len, 'paths': paths, 'c8_scaled': c8_scaled}
                   for i, file in enumerate(c13_scaled)]

    results = {'diurnal_pixel': [], 'diurnal_count': [], 'id_list': []}

    with Pool(4, initializer=init_worker) as pool:
        for res in pool.map(mp_running, needed_args):
            if res is None:
                continue
            diurnal_count, pixels, atcf_id = res
            results['diurnal_pixel'].append(pixels)
            results['diurnal_count'].append(diurnal_count)
            results['id_list'].append(atcf_id)

    # Split into the Atlantic and Eastern Pacific basins.
    results_AL = {'diurnal_pixel': [], 'diurnal_count': [], 'id_list': []}
    results_EP = {'diurnal_pixel': [], 'diurnal_count': [], 'id_list': []}

    for idx, storm_id in enumerate(results['id_list']):
        target = results_AL if 'AL' in storm_id else results_EP if 'EP' in storm_id else None
        if target:
            target['id_list'].append(storm_id)
            target['diurnal_pixel'].append(results['diurnal_pixel'][idx])
            target['diurnal_count'].append(results['diurnal_count'][idx])

    # Clean None values
    results['diurnal_pixel'], results['diurnal_count'] = clean_func(results['diurnal_pixel'], results['diurnal_count'])
    results_AL['diurnal_pixel'], results_AL['diurnal_count'] = clean_func(results_AL['diurnal_pixel'],
                                                                          results_AL['diurnal_count'])
    results_EP['diurnal_pixel'], results_EP['diurnal_count'] = clean_func(results_EP['diurnal_pixel'],
                                                                          results_EP['diurnal_count'])

    # -------------------------- Plotting --------------------------

    # Combined Atlantic & Eastern Pacific
    data, labels = group_func(results['diurnal_pixel'], results['diurnal_count'])
    plot_violin(data, labels,
                title='TCB Occurrences vs Diurnal Cycle Stage\nfor 2019-2023 Atlantic & Eastern Pacific TCs',
                xlabel='Local Solar Time',
                ylabel='Percentage of Storm Pixels with TCBs',
                filename='diurnal_violin_all.png')
    plot_mann_whitney_contour(data, labels, 'diurnal_mannwhitney_all.png', 'LST', 'P-Value',
                              'Diurnal Mann-Whitney P-Values for Atlantic and Eastern Pacific TCs')

    # Separate AL & EP
    data1, labels1 = group_func(results_AL['diurnal_pixel'], results_AL['diurnal_count'])
    data2, labels2 = group_func(results_EP['diurnal_pixel'], results_EP['diurnal_count'])

    plot_violin_2panel(data1, labels1, data2, labels2,
                       suptitle='TCB Occurrences vs Diurnal Cycle Stage',
                       xlabel='Local Solar Time',
                       ylabel='Percentage of Storm Pixels with TCBs',
                       filename='diurnal_violin_ALEP.png')
    plot_contourf_2panel(data1, labels1, data2, labels2,
                         suptitle='Diurnal Cycle Mann-Whitney P-Values',
                         xlabel='Local Solar Time',
                         ylabel='Local Solar Time',
                         filename='diurnal_mannwhitney_ALEP.png')
