# coding=utf-8
"""
Last Edited: 10/02/2025
@author: John Mark Mayhall
Purpose: Identify transverse bands in TC quadrants based on shear vector.
"""
from collections import defaultdict

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


# ----------------------------
# Violin/Box plot helpers
# ----------------------------
def adjacent_values(sorted_array: np.ndarray, q1: float, q3: float) -> tuple[float, float]:
    """Compute whisker limits for violin/box plots."""
    iqr = q3 - q1
    upper = min(max(sorted_array[sorted_array <= q3 + 1.5 * iqr], default=q3), max(sorted_array))
    lower = max(min(sorted_array[sorted_array >= q1 - 1.5 * iqr], default=q1), min(sorted_array))
    return lower, upper


def plot_violin(data: list[list[float]], labels: list[list[int]], title: str, xlabel: str,
                ylabel: str, filename: str):
    """
    Plot a single violin plot.

    :param data: Nested list of values
    :param labels: Nested list of labels
    :param title: Plot title
    :param xlabel: X-axis label
    :param ylabel: Y-axis label
    :param filename: Output filename
    """
    labels_flat = np.unique(np.concatenate([np.array(sublist) for sublist in labels]))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.tick_params(labelsize=16)
    ax.set_ylim(0, 60)

    # Width adjustment
    widths = 2 if 'Diurnal' in title else 8
    means_bool, medians_bool = False, False
    parts = ax.violinplot(data, positions=labels_flat, showmeans=means_bool, showmedians=medians_bool,
                          showextrema=False, widths=widths)
    for pc in parts['bodies']:
        pc.set_facecolor('#FFA500')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    # Compute medians and whiskers
    quartile1, medians, quartile3, whiskers_min, whiskers_max = [], [], [], [], []
    for group in data:
        group = np.sort(group)
        q1, med, q3 = np.percentile(group, [25, 50, 75])
        whisk_min, whisk_max = adjacent_values(group, q1, q3)
        quartile1.append(q1)
        medians.append(med)
        quartile3.append(q3)
        whiskers_min.append(whisk_min)
        whiskers_max.append(whisk_max)

    ax.scatter(labels_flat, medians, marker='o', color='blue', s=100, zorder=3)
    ax.vlines(labels_flat, quartile1, quartile3, color='k', lw=5)
    ax.vlines(labels_flat, whiskers_min, whiskers_max, color='k', lw=1)

    # Axis formatting
    if 'Diurnal' in title:
        ax.set_xlim(-2, 23)
        ax.set_xticks(range(0, 22, 3))
        num_list = range(0, 22, 3)
        ax.set_xticklabels([str(item) for item in num_list], rotation=45, ha='right')
    else:
        ax.set_xlim(15, 165)
        ax.set_xticks(range(20, 161, 10))
        ax.xaxis.set_major_formatter(FuncFormatter(label_every_10))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.axvline(65, color='orange', linestyle='--', lw=2, label='TS/HUR Transition')
        ax.axvline(95, color='magenta', linestyle='--', lw=2, label='HUR/MAJ HUR Transition')
        ax.legend(fontsize=12, loc='upper right').set_alpha(0.5)

    # Annotate sample size
    ymax = max([np.max(d) for d in data])
    y_offset = 0.05 * ymax
    for i, d in enumerate(data):
        text_y = min(max(d) + y_offset, 100)
        ax.text(labels_flat[i], text_y, str(len(d)), ha='center', fontsize=16,
                fontweight='bold', rotation=90, va='center',
                path_effects=[path_effects.Stroke(linewidth=2, foreground='white'),
                              path_effects.Normal()])

    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.grid(True, linestyle='--', linewidth=1.0, alpha=1)
    plt.savefig(filename)
    plt.close()


# ----------------------------
# Mann-Whitney and contourf
# ----------------------------
def compute_pval_grid(data: list[list[float]]) -> np.ndarray:
    """Compute pairwise Mann-Whitney p-values for a list of groups."""
    n = len(data)
    p_grid = np.empty((n, n))
    mannwhitneyu_type = 'two-sided'
    for i in range(n):
        for j in range(n):
            _, p = mannwhitneyu(data[i], data[j], alternative=mannwhitneyu_type)
            p_grid[i, j] = p
    return p_grid


def plot_mann_whitney_contour(data: list[list[float]], labels: list[list[int]], filename: str,
                              xlabel: str, ylabel: str, title: str, test_type='p'):
    """
    Plot Mann-Whitney U test results as scatter of significant pairs (p < 0.05).

    :param data: Nested list of groups
    :param labels: Nested list of labels
    :param filename: Output file
    :param xlabel: X-axis label
    :param ylabel: Y-axis label
    :param title: Plot title
    :param test_type: 'p' for p-value, 'u' for U statistic
    """
    unique_labels = sorted(set([item for sublist in labels for item in sublist]))
    n = len(unique_labels)
    grouped_data = {label[0]: d for label, d in zip(labels, data)}
    mannwhitneyu_type = 'two-sided'

    stat_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            group1 = grouped_data[unique_labels[i]]
            group2 = grouped_data[unique_labels[j]]
            u_stat, p_value = mannwhitneyu(group1, group2, alternative=mannwhitneyu_type)
            stat_matrix[i, j] = p_value if test_type == 'p' else u_stat

    # Mask significant values
    x, y = np.meshgrid(unique_labels, unique_labels)
    mask = stat_matrix.ravel() < 0.05

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(x.ravel()[mask], y.ravel()[mask], c='black', s=200, label='p < 0.05')
    ax.set_xlabel(xlabel, fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    ax.set_title(title, fontsize=17)
    ax.grid(True, linestyle='--', linewidth=1, alpha=1)
    ax.legend(loc='upper right', fontsize=16)
    plt.savefig(filename)
    plt.close()


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
    with Pool(12, initializer=init_worker) as pool:
        for result in pool.map(mp_running, needed_args):
            if result is None:
                continue
            diurnal_count, intensity_count, wind_change_counts, pixels, atcf_id = result

            # Append results
            results['diurnal_pixel'].append(pixels)
            results['diurnal_count'].append(diurnal_count)
            results['intensity_pixel'].append(pixels)
            results['intensity_count'].append(int(intensity_count))

            for h, wind_change in zip([-24, -18, -12, -6, 6, 12, 18, 24], wind_change_counts):
                rounded_wind = int(np.round(wind_change / 20) * 20) if wind_change is not None else None
                key = f'{h:+}'
                results['wind_change'][key]['pixel'].append(pixels)
                results['wind_change'][key]['count'].append(rounded_wind)

            results['id_list'].append(atcf_id)


    # ================= Split Atlantic & Eastern Pacific =================
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


    results = clean_all(results)
    results_al = clean_all(results_al)
    results_ep = clean_all(results_ep)


    # ================= Helper to plot past/future wind changes =================
    def plot_wind_change(results: dict, hours: list, fname_prefix: str, title_prefix: str):
        """
        Helper function for plotting wind change data.
        :param results: Dictionary of results
        :param hours: List of hours
        :param fname_prefix: Filename prefix
        :param title_prefix: Title prefix
        """
        ax = None
        """Generic function for plotting wind change violin plots and p-values."""
        data_list = [group_func(results['wind_change'][f'{h:+}']['pixel'],
                                results['wind_change'][f'{h:+}']['count'])[0] for h in hours]
        x_labels = [group_func(results['wind_change'][f'{h:+}']['pixel'],
                               results['wind_change'][f'{h:+}']['count'])[1] for h in hours]
        labels_list = [f"{h}hr" for h in hours]

        # Violin plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'{title_prefix} TCB Occurrences vs TC Intensity Change', fontsize=20)
        fig.supxlabel(r'TC Wind Speed Change ($\frac{dv}{dt}, kts$)', fontsize=20)
        fig.supylabel('Percentage of Pixels with TCBs', fontsize=20)

        for ax, data, label, labels in zip(axes.flatten(), data_list, labels_list, x_labels):
            labels = np.unique(np.concatenate([np.array(sublist) for sublist in labels]))
            ax.tick_params(axis='both', labelsize=16)
            parts = ax.violinplot(data, positions=labels, showmeans=False, showmedians=False, showextrema=False,
                                  widths=8)
            for pc in parts['bodies']:
                pc.set_facecolor('#FFA500')
                pc.set_edgecolor('black')
                pc.set_alpha(1)

            # Quartiles and whiskers
            quartile1, medians, quartile3, whiskers_min, whiskers_max = [], [], [], [], []
            for group in data:
                group = np.sort(group)
                q1, med, q3 = np.percentile(group, [25, 50, 75])
                whisk_min, whisk_max = adjacent_values(group, q1, q3)
                quartile1.append(q1)
                medians.append(med)
                quartile3.append(q3)
                whiskers_min.append(whisk_min)
                whiskers_max.append(whisk_max)
            ax.scatter(labels, medians, marker='o', color='blue', s=100, zorder=3)
            ax.vlines(labels, quartile1, quartile3, color='k', linestyle='-', lw=5)
            ax.vlines(labels, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

            # Axis limits
            ax.set_ylim((0, 60))
            if '6' in label:
                ax.set_xlim((-65, 45))
                ax.set_xticks(range(-60, 41, 20))
            elif '12' in label:
                ax.set_xlim((-105, 65))
                ax.set_xticks(range(-100, 61, 20))
            elif '18' in label:
                ax.set_xlim((-85, 85))
                ax.set_xticks(range(-80, 81, 20))
            elif '24' in label:
                ax.set_xlim((-105, 105))
                ax.set_xticks(range(-100, 101, 20))
            ax.xaxis.set_major_formatter(FuncFormatter(label_every_20))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            # Add transition lines
            ax.axvline(x=-20, color='blue', linestyle='--', lw=2, label='RW Transition')
            ax.axvline(x=30, color='magenta', linestyle='--', lw=2, label='RI Transition')
            ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)

        handles, labels_ = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels_, loc='upper right', fontsize=16)
        legend.set_alpha(0.4)
        plt.savefig(f'{fname_prefix}_all.png')
        plt.close()

        # Mann-Whitney P-values
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'{title_prefix} TC Intensity Change Mann-Whitney P-Values', fontsize=20)
        for ax, data, label, labels in zip(axes.flatten(), data_list, labels_list, x_labels):
            labels = np.unique(np.concatenate([np.array(sublist) for sublist in labels]))
            p_grid = compute_pval_grid(data)
            x, y = np.meshgrid(labels, labels)
            mask = p_grid.ravel() < 0.05
            ax.scatter(x.ravel()[mask], y.ravel()[mask], c='black', s=150, label='p < 0.05')
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
            ax.set_yticklabels(labels, fontsize=14)
            ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)
        plt.savefig(f'{fname_prefix}_mannwhitney_all.png')
        plt.close()


    # ==================== Run Wind Change Plots =====================
    plot_wind_change(results, [-24, -18, -12, -6], 'prev_intensity_change', 'Previous')
    plot_wind_change(results, [6, 12, 18, 24], 'future_intensity_change', 'Future')

    # ==================== Violin & Contour Plots for Basins =====================
    data_al, labels_al = group_func(results_al['diurnal_pixel'], results_al['diurnal_count'])
    data_ep, labels_ep = group_func(results_ep['diurnal_pixel'], results_ep['diurnal_count'])
    plot_violin_2panel(data_al, labels_al, data_ep, labels_ep,
                       suptitle='TCB Occurrences vs Diurnal Cycle Stage',
                       xlabel='Local Solar Time',
                       ylabel='Percentage of Storm Pixels with TCBs',
                       filename='diurnal_violin_ALEP.png')
    plot_contourf_2panel(data_al, labels_al, data_ep, labels_ep,
                         suptitle='Diurnal Cycle Mann-Whitney P-Values',
                         xlabel='Local Solar Time',
                         ylabel='Local Solar Time',
                         filename='diurnal_mannwhitney_ALEP.png')

    data_al, labels_al = group_func(results_al['intensity_pixel'], results_al['intensity_count'])
    data_ep, labels_ep = group_func(results_ep['intensity_pixel'], results_ep['intensity_count'])
    plot_violin_2panel(data_al, labels_al, data_ep, labels_ep,
                       suptitle='TCB Occurrences vs TC Intensity',
                       xlabel='TC Current Wind Speed (kts)',
                       ylabel='Percentage of Storm Pixels with TCBs',
                       filename='intensity_violin_ALEP.png')
    plot_contourf_2panel(data_al, labels_al, data_ep, labels_ep,
                         suptitle='TC Intensity Mann-Whitney P-Values',
                         xlabel='Intensity (kts)',
                         ylabel='Intensity (kts)',
                         filename='intensity_mannwhitney_ALEP.png')
