# coding=utf-8
"""
Last Edited: 10/02/2025
Author: John Mark Mayhall
Purpose: Identify transverse bands in TC quadrants based on shear vector,
         generate violin plots and Mann-Whitney contour plots for RH and SST.
"""
import glob
from collections import defaultdict
from multiprocessing import Pool

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from shear_multi import mp_running, init_worker


# ---------------- Utility Functions ---------------- #

def label_every_10(x, pos):
    """Tick formatter: show every 10th label."""
    _ = pos  # Included to remove variable not used warning.
    return f"{int(x)}" if x % 10 == 0 else ''


def adjacent_values(sorted_array, q1, q3):
    """Compute whisker values for box/violin plots."""
    iqr = q3 - q1
    upper_adj = q3 + 1.5 * iqr
    lower_adj = q1 - 1.5 * iqr
    upper = max([x for x in sorted_array if x <= upper_adj], default=q3)
    lower = min([x for x in sorted_array if x >= lower_adj], default=q1)
    return lower, upper


def group_func(data: list, group_labels: list):
    """
    Group data by labels, convert to percentage of total pixels.
    :param data: list of pixel counts
    :param group_labels: list of group identifiers
    :return: (grouped_data, grouped_labels)
    """
    grouped_l2 = defaultdict(list)
    for label, value in zip(group_labels, data):
        grouped_l2[label].append((value / (1024 * 1024)) * 100)

    keys = sorted(grouped_l2.keys())
    grouped_data = [grouped_l2[k] for k in keys]
    grouped_labels = [[k] * len(grouped_l2[k]) for k in keys]

    return grouped_data, grouped_labels


def compute_pval_grid(data: list):
    """Compute pairwise Mann-Whitney p-values between data groups."""
    n = len(data)
    mannwhitney_type = 'two-sided'
    p_grid = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            _, p = mannwhitneyu(data[i], data[j], alternative=mannwhitney_type)
            p_grid[i, j] = p
    return p_grid


# ---------------- Plotting Functions ---------------- #

def plot_violin(data, labels, title, xlabel, ylabel, filename, ylim=(0, 60), xlim=None, xticks=None):
    """Generic violin plot."""
    labels_flat = np.unique(np.concatenate([np.array(sublist) for sublist in labels]))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.tick_params(labelsize=16)
    ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    if xticks:
        ax.set_xticks(xticks)

    showmeans_bool, showmedians_bool = False, False
    parts = ax.violinplot(data, positions=labels_flat, showmeans=showmeans_bool, showmedians=showmedians_bool,
                          showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#FFA500')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    # Overlay quartiles and medians
    for group, label in zip(data, labels_flat):
        group_sorted = np.sort(group)
        q1, med, q3 = np.percentile(group_sorted, [25, 50, 75])
        whisk_min, whisk_max = adjacent_values(group_sorted, q1, q3)
        ax.vlines(label, q1, q3, color='k', lw=5)
        ax.vlines(label, whisk_min, whisk_max, color='k', lw=1)
        ax.scatter(label, med, color='blue', s=100, zorder=3)
        # Add count text above each violin
        ax.text(label, min(max(group_sorted) + 0.05 * (max(group_sorted) - min(group_sorted)), 100),
                f"{len(group)}", ha='center', fontsize=16, fontweight='bold', rotation=90,
                va='center', path_effects=[path_effects.Stroke(linewidth=2, foreground='white'),
                                           path_effects.Normal()])

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20, x=0.05)
    ax.set_title(title, fontsize=20)
    ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.savefig(filename)
    plt.close()


def plot_violin_2panel(data1, labels1, data2, labels2, suptitle, xlabel, ylabel, filename):
    """Two-panel violin plot for AL and EP data."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, data, labels, region in zip(axes, [data1, data2], [labels1, labels2], ['Atlantic', 'Eastern Pacific']):
        plot_violin(data, labels, title=region, xlabel=xlabel, ylabel=ylabel, filename=None)
        ax.set_title(region, fontsize=16)
    fig.suptitle(suptitle, fontsize=20)
    fig.supxlabel(xlabel, fontsize=20)
    fig.supylabel(ylabel, fontsize=20)
    plt.savefig(filename)
    plt.close()


def plot_mann_whitney_contour(data, labels, filename, xlabel, ylabel, title):
    """Plot Mann-Whitney p-values as black dots where p < 0.05."""
    unique_labels = sorted(set([item for sublist in labels for item in sublist]))
    n = len(unique_labels)
    grouped_data = {label[0]: d for label, d in zip(labels, data)}
    stat_matrix = np.zeros((n, n))
    mannwhitney_type = 'two-sided'

    for i in range(n):
        for j in range(n):
            group1 = grouped_data[unique_labels[i]]
            group2 = grouped_data[unique_labels[j]]
            _, p_value = mannwhitneyu(group1, group2, alternative=mannwhitney_type)
            stat_matrix[i, j] = p_value

    stat_matrix[stat_matrix < 0.05] = 0.049
    fig, ax = plt.subplots(figsize=(12, 8))
    x, y = np.meshgrid(unique_labels, unique_labels)
    mask = stat_matrix.ravel() < 0.05
    ax.scatter(x.ravel()[mask], y.ravel()[mask], c='black', s=200, label='p < 0.05')
    ax.set_xlim((17, 33))
    ax.set_ylim((17, 33))
    tick_marks = range(18, 33, 1)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xlabel(xlabel, fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17, x=0.05)
    ax.set_title(title, fontsize=17)
    ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)
    ax.legend(loc='upper right', fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.savefig(filename)
    plt.close()


def plot_contourf_2panel(data1, labels1, data2, labels2, suptitle, xlabel, ylabel, filename):
    """Two-panel Mann-Whitney plot for AL and EP."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, data, labels, region in zip(axes, [data1, data2], [labels1, labels2], ['Atlantic', 'Eastern Pacific']):
        pvals = compute_pval_grid(data)
        pvals[pvals < 0.05] = 0.049
        unique_labels = sorted(set([item for sublist in labels for item in sublist]))
        x, y = np.meshgrid(unique_labels, unique_labels)
        mask = pvals.ravel() < 0.05
        ax.scatter(x.ravel()[mask], y.ravel()[mask], c='black', s=200, label='p < 0.05')
        ax.set_xlim(min(unique_labels), max(unique_labels))
        ax.set_ylim(min(unique_labels), max(unique_labels))
        ax.set_xticks(unique_labels)
        ax.set_yticks(unique_labels)
        ax.set_title(region, fontsize=16)
        ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)
    fig.suptitle(suptitle, fontsize=20)
    fig.supxlabel(xlabel, fontsize=20)
    fig.supylabel(ylabel, fontsize=20)
    plt.savefig(filename)
    plt.close()


def prepare_rh_data(results_dict):
    """Prepare RH data for plotting."""
    data_list = [group_func(results_dict[f'rh{h}_pixel'], results_dict[f'rh{h}_count'])[0] for h in rh_ids]
    labels_list = [group_func(results_dict[f'rh{h}_pixel'], results_dict[f'rh{h}_count'])[1] for h in rh_ids]
    return data_list, labels_list


if __name__ == '__main__':
    # ---------------- Main Workflow ---------------- #
    # Config
    CUTOFF = 0.02
    base_dir = '/rstor/jmayhall/'
    paths = {
        'latlon': f'{base_dir}cataloging/nc_process/geojson_transform/completed_arrays/latlon_arrs/*.npz',
        'hurdat': f'{base_dir}cataloging/hurdat_update.txt',
        'c8': f'{base_dir}cataloging/nc_process/geojson_transform/completed_arrays/C08_scaled/*',
        'c13': f'{base_dir}cataloging/nc_process/geojson_transform/completed_arrays/C13_scaled/*',
        'ships': f'{base_dir}cataloging/nc_process/violin_plots/humidity_sst_all/ships_interp.txt'
    }

    # Load data
    hurdat_df = pd.read_csv(paths['hurdat'], sep='\t')
    latlon_arrays = glob.glob(paths['latlon'])
    c8_scaled = pd.DataFrame({'Name': glob.glob(paths['c8'])})
    c13_scaled = glob.glob(paths['c13'])
    wind_dict = pd.read_csv(paths['ships'], sep='\t', index_col=0)
    path_len = len(paths['c13']) - 1

    # Prepare multiprocessing arguments
    needed_args = []
    for file in c13_scaled:
        try:
            matching_c8 = c8_scaled.loc[c8_scaled['Name'].str.contains(file[path_len:path_len + 22]), 'Name'].values[0]
            needed_args.append({'file': file, 'path_len': path_len, 'paths': paths,
                                'c8_scaled': matching_c8, 'cut_off': CUTOFF})
        except IndexError:
            continue

    # Collect results in parallel
    results = defaultdict(list)
    with Pool(12, initializer=init_worker) as pool:
        for res in pool.map(mp_running, needed_args):
            if res is None:
                continue
            rhlo_count, rhmd_count, rhhi_count, sst_count, pixels, atcf_id = res
            results['rhhi_pixel'].append(pixels)
            results['rhhi_count'].append(rhhi_count)
            results['rhmd_pixel'].append(pixels)
            results['rhmd_count'].append(rhmd_count)
            results['rhlo_pixel'].append(pixels)
            results['rhlo_count'].append(rhlo_count)
            results['sst_pixel'].append(pixels)
            results['sst_count'].append(sst_count)
            results['id_list'].append(atcf_id)

    # Split results by basin
    results_AL = {k: [] for k in results}
    results_EP = {k: [] for k in results}
    for idx, storm_id in enumerate(results['id_list']):
        target = results_AL if 'AL' in storm_id else results_EP if 'EP' in storm_id else None
        if target:
            for k in results:
                target[k].append(results[k][idx])

    # ------------------ Plot SST ------------------ #
    data_all, labels_all = group_func(results['sst_pixel'], results['sst_count'])
    plot_violin(data_all, labels_all,
                title='TCB Occurrences vs SST for 2019-2023 Atlantic & Eastern Pacific TCs',
                xlabel='SST (C)',
                ylabel='Percentage of Storm Pixels with TCBs',
                filename='sst_violin_all.png')
    plot_mann_whitney_contour(data_all, labels_all, 'sst_mannwhitney_all.png', 'SST (C)', 'SST (C)',
                              'SST Mann-Whitney P-Values')

    data_AL, labels_AL = group_func(results_AL['sst_pixel'], results_AL['sst_count'])
    data_EP, labels_EP = group_func(results_EP['sst_pixel'], results_EP['sst_count'])
    plot_violin_2panel(data_AL, labels_AL, data_EP, labels_EP,
                       suptitle='TCB Occurrences vs SST',
                       xlabel='SST (C)',
                       ylabel='Percentage of Storm Pixels with TCBs',
                       filename='sst_violin_ALEP.png')
    plot_contourf_2panel(data_AL, labels_AL, data_EP, labels_EP,
                         suptitle='SST Mann-Whitney P-Values',
                         xlabel='SST (C)',
                         ylabel='SST (C)',
                         filename='sst_mannwhitney_ALEP.png')

    # ------------------ Plot RH ------------------ #
    rh_ids = ['lo', 'md', 'hi']
    rh_labels = ['850-700 hPa RH', '700-500 hPa RH', '500-300 hPa RH']

    data_AL_rh, labels_AL_rh = prepare_rh_data(results_AL)
    data_EP_rh, labels_EP_rh = prepare_rh_data(results_EP)

    # Combine for 2x3 plotting
    data_rh_all = data_AL_rh + data_EP_rh
    labels_rh_all = labels_AL_rh + labels_EP_rh
    labels_names = rh_labels + rh_labels

    # Plot violin and Mann-Whitney for RH
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('TCB Occurrences vs RH', fontsize=24)
    fig.supxlabel('RH (%)', fontsize=24)
    fig.supylabel('Percentage of Pixels with TCBs', fontsize=24)

    for idx, (ax, data, label, labels) in enumerate(zip(axes.flatten(), data_rh_all, labels_names, labels_rh_all)):
        plot_violin([data], [labels], title='', xlabel='', ylabel='', filename=None)  # reuse function for each subplot
        region = 'Atlantic' if idx < 3 else 'Eastern Pacific'
        ax.set_title(f"{region}: {label}", fontsize=20)

    plt.savefig('rh_ALEP.png')
    plt.close()

    # RH Mann-Whitney 2x3
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('RH Mann-Whitney P-Values', fontsize=24)
    fig.supxlabel('RH (%)', fontsize=24)
    fig.supylabel('RH (%)', fontsize=24)

    for idx, (ax, data, labels, label) in enumerate(zip(axes.flatten(), data_rh_all, labels_rh_all, labels_names)):
        pvals = compute_pval_grid(data)
        pvals[pvals < 0.05] = 0.049
        unique_labels = sorted(set([item for sublist in labels for item in sublist]))
        x, y = np.meshgrid(unique_labels, unique_labels)
        mask = pvals.ravel() < 0.05
        ax.scatter(x.ravel()[mask], y.ravel()[mask], c='black', s=200, label='p < 0.05')
        region = 'Atlantic' if idx < 3 else 'Eastern Pacific'
        ax.set_title(f"{region}: {rh_labels[idx % 3]}", fontsize=20)
    plt.savefig('rh_ALEP_mannwhitney.png')
    plt.close()
