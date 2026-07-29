# coding=utf-8
"""
Last Edited: 06/12/2026
@author: John Mark Mayhall
Purpose: Identify cirrus bands in TC quadrants based on shear vector.
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
from shear_multi import mp_running, init_worker
from statsmodels.stats.multitest import multipletests


def shear_color(vws):
    if vws < 5:
        return 'green'
    elif vws <= 10:
        return 'orange'
    else:
        return 'red'


def group_func(data: list, group_labels: list):
    df = pd.DataFrame({"label": group_labels, "data": data}).dropna()
    df["data"] = (df["data"] / (1024 * 1024)) * 100
    grouped = df.groupby("label")["data"].apply(list).sort_index()
    return grouped.tolist(), [[k] * len(v) for k, v in grouped.items()]


def adjacent_values(sorted_array, q1, q3):
    iqr = q3 - q1
    upper_adj = q3 + 1.5 * iqr
    lower_adj = q1 - 1.5 * iqr
    upper = max([x for x in sorted_array if x <= upper_adj], default=q3)
    lower = min([x for x in sorted_array if x >= lower_adj], default=q1)
    return lower, upper


def compute_pval_rbc_grid(data: list) -> tuple[np.ndarray, np.ndarray]:
    n = len(data)
    p_grid, rbc_grid = np.full((n, n), np.nan), np.full((n, n), np.nan)
    p_values_list, rbc_list, indices = [], [], []

    for i in range(n):
        for j in range(i + 1, n):
            if len(data[i]) == 0 or len(data[j]) == 0: continue
            try:
                u_stat, p = mannwhitneyu(data[i], data[j], alternative='two-sided')
                p_values_list.append(p)
                indices.append((i, j))
                n1, n2 = len(data[i]), len(data[j])
                rbc_list.append(abs(1 - (2 * u_stat) / (n1 * n2)))
            except ValueError:
                pass

    if len(p_values_list) > 0:
        _, corrected_p_values, _, _ = multipletests(pvals=p_values_list, alpha=0.05, method='fdr_bh')
        for (i, j), p_corr, rbc in zip(indices, corrected_p_values, rbc_list):
            p_grid[i, j] = p_grid[j, i] = p_corr
            rbc_grid[i, j] = rbc_grid[j, i] = rbc

    return p_grid, rbc_grid


def clean_func(list1: list, list2: list) -> list:
    none_indices = ([i for i, v in enumerate(list1) if v is None] + [i for i, v in enumerate(list2) if v is None])
    return [v for i, v in enumerate(list1) if i not in none_indices], [v for i, v in enumerate(list2) if
                                                                       i not in none_indices]


def split_basin(results: dict):
    results_al = {k: [] for k in results}
    results_ep = {k: [] for k in results}
    for idx, storm_id in enumerate(results['id_list']):
        target = results_al if 'AL' in storm_id else results_ep if 'EP' in storm_id else None
        if target:
            target['id_list'].append(storm_id)
            for key in ['shear_pixel', 'shear_count']:
                target[key].append(results[key][idx])
    return results_al, results_ep


def clean_all(results: dict) -> dict:
    results['shear_pixel'], results['shear_count'] = clean_func(results['shear_pixel'], results['shear_count'])
    return results


# ================= Single-Axis Plotting Functions =================

def plot_violin_single_shear(ax, data, labels, title, y_label=None):
    # Bin into 3-hour (5 m/s) intervals
    bins = defaultdict(list)
    for group, label_group in zip(data, labels):
        for val, lab in zip(group, label_group):
            bins[5 * np.round(lab / 5)].append(val)

    violin_positions = np.array(sorted(bins.keys()), dtype=float)
    violin_data = [np.asarray(bins[bc], dtype=float) for bc in violin_positions]

    quartile1, medians, quartile3, whiskers_min, whiskers_max = [], [], [], [], []
    for group in violin_data:
        group = group[np.isfinite(group)]
        if group.size == 0:
            quartile1.append(np.nan);
            medians.append(np.nan);
            quartile3.append(np.nan)
            whiskers_min.append(np.nan);
            whiskers_max.append(np.nan)
            continue
        q1, med, q3 = np.percentile(group, [25, 50, 75])
        whisk_min, whisk_max = adjacent_values(group, q1, q3)
        quartile1.append(q1);
        medians.append(med);
        quartile3.append(q3)
        whiskers_min.append(whisk_min);
        whiskers_max.append(whisk_max)

    parts = ax.violinplot(violin_data, positions=violin_positions, showmeans=False, showmedians=False,
                          showextrema=False, widths=3)
    for pc in parts['bodies']:
        pc.set_facecolor('#FFA500')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    ax.scatter(violin_positions, medians, color='blue', s=80, zorder=3)
    ax.vlines(violin_positions, quartile1, quartile3, color='k', lw=2)
    ax.vlines(violin_positions, whiskers_min, whiskers_max, color='k', lw=1)

    for pos, values in zip(violin_positions, violin_data):
        if len(values) == 0: continue
        ax.text(pos, min(max(values) + 2, 60), f'{len(values)}', ha='center', fontsize=16, fontweight='bold',
                rotation=90, va='center',
                path_effects=[path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

    ax.set_xlim((-5, 35))
    ax.set_ylim((0, 60))
    ax.set_xticks(range(0, 31, 5))
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)
    ax.set_title(title, fontsize=20)
    if y_label: ax.set_ylabel(y_label, fontsize=20)


def plot_pvals_single_shear(ax, data, labels, title, y_label=None):
    centers = np.unique(np.concatenate([np.array(sublist) for sublist in labels]))
    pvals, rbc = compute_pval_rbc_grid(data)

    x, y = np.meshgrid(centers, centers)
    mask = (np.array(pvals).ravel() < 0.05)
    x_masked, y_masked = x.ravel()[mask], y.ravel()[mask]

    for x_val, y_val in zip(x_masked, y_masked):
        ax.plot(x_val, y_val, marker='o', markersize=np.sqrt(250),
                markerfacecolor=shear_color(x_val), markerfacecoloralt=shear_color(y_val),
                markeredgecolor='black', markeredgewidth=1.5, fillstyle='left', linestyle='None', zorder=3)

    ax.set_xlim((-5, 35))
    ax.set_ylim((-5, 35))
    ax.set_xticks(range(0, 31, 5))
    ax.set_yticks(range(0, 31, 5))
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)
    ax.set_title(title, fontsize=20)
    if y_label: ax.set_ylabel(y_label, fontsize=20)


def plot_rbc_single_shear(ax, data, labels, title, y_label=None):
    centers = np.unique(np.concatenate([np.array(sublist) for sublist in labels]))
    pvals, rbc = compute_pval_rbc_grid(data)

    base = plt.get_cmap("coolwarm")
    cmap = LinearSegmentedColormap.from_list("coolwarm_trimmed", np.vstack([
        base(np.linspace(0.00, 0.30, 128)), base(np.linspace(0.70, 1.00, 128))
    ]))

    x, y = np.meshgrid(centers, centers)
    mask = (np.array(pvals).ravel() < 0.05)
    x_masked, y_masked, rbc_masked = x.ravel()[mask], y.ravel()[mask], np.array(rbc).ravel()[mask]

    for x_val, y_val, rbc_val in zip(x_masked, y_masked, rbc_masked):
        ax.text(x_val, y_val, f"{rbc_val * 100:.0f}", color=cmap(rbc_val),
                ha='center', va='center', fontweight='bold', fontsize=18)

    ax.set_xlim((-5, 35))
    ax.set_ylim((-5, 35))
    ax.set_xticks(range(0, 31, 5))
    ax.set_yticks(range(0, 31, 5))
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)
    ax.set_title(title, fontsize=20)
    if y_label: ax.set_ylabel(y_label, fontsize=20)


if __name__ == '__main__':
    # ======================== Config ========================
    cut_off = 0.02
    ri_thresh, rw_thresh = 30, -20
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

    needed_args = [{
        'i': i, 'file': file, 'path_len': path_len, 'paths': paths, 'c8_scaled': c8_scaled
    } for i, file in enumerate(c13_scaled)]

    results = {'shear_pixel': [], 'shear_count': [], 'id_list': []}

    with Pool(12, initializer=init_worker) as pool:
        for result in pool.map(mp_running, needed_args):
            if result is None: continue
            shear_count, pixels, atcf_id = result
            results['shear_pixel'].append(pixels)
            results['shear_count'].append(shear_count)
            results['id_list'].append(atcf_id)

    results_al, results_ep = split_basin(results)
    results = clean_all(results)
    results_al = clean_all(results_al)
    results_ep = clean_all(results_ep)

    data_al, labels_al = group_func(results_al['shear_pixel'], results_al['shear_count'])
    data_ep, labels_ep = group_func(results_ep['shear_pixel'], results_ep['shear_count'])

    # ==================== Generate 2x3 Plot =====================
    fig, axes = plt.subplots(2, 3, figsize=(28, 16), sharex=True)
    fig.suptitle('CB Occurrences vs Vertical Wind Shear', fontsize=32)

    ylab_violin = 'Percentage of Storm Pixels with CBs'
    ylab_stat = r'Vertical Wind Shear ($m s^{-1}$)'
    xlab_all = r'Vertical Wind Shear ($m s^{-1}$)'

    # --- ROW 0: Atlantic ---
    plot_violin_single_shear(axes[0, 0], data_al, labels_al, 'AL: Violin Plot', y_label=ylab_violin)
    plot_pvals_single_shear(axes[0, 1], data_al, labels_al, 'AL: Mann-Whitney P-Values', y_label=ylab_stat)
    plot_rbc_single_shear(axes[0, 2], data_al, labels_al, 'AL: Rank-Biserial Correlations', y_label=ylab_stat)

    # --- ROW 1: East Pacific ---
    plot_violin_single_shear(axes[1, 0], data_ep, labels_ep, 'EP: Violin Plot', y_label=ylab_violin)
    plot_pvals_single_shear(axes[1, 1], data_ep, labels_ep, 'EP: Mann-Whitney P-Values', y_label=ylab_stat)
    plot_rbc_single_shear(axes[1, 2], data_ep, labels_ep, 'EP: Rank-Biserial Correlations', y_label=ylab_stat)

    for j in range(3):
        axes[1, j].set_xlabel(xlab_all, fontsize=20)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    # Shear specific legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=14, label='Low Shear'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=14, label='Moderate Shear'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=14, label='High Shear')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=20, bbox_to_anchor=(0.5, 0.93))

    plt.savefig("shear_combined.png", dpi=300, bbox_inches="tight")
    plt.close('all')