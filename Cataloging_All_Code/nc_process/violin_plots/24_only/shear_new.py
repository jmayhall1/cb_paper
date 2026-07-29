# coding=utf-8
"""
Last Edited: 07/10/2025
@author: John Mark Mayhall
Purpose: Identify cirrus bands in TC quadrants based on intensity change.
"""
import glob
from multiprocessing import Pool

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge
from scipy.stats import mannwhitneyu
from shear_multi import mp_running, init_worker
from statsmodels.stats.multitest import multipletests


class HandlerHalfCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        center = (width / 2 - xdescent, height / 2 - ydescent)
        radius = min(width, height) / 1.45
        left = Wedge(center, radius, 90, 270, facecolor='red', edgecolor='black', lw=1, transform=trans)
        right = Wedge(center, radius, -90, 90, facecolor='blue', edgecolor='black', lw=1, transform=trans)
        return [left, right]


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
    p_grid = np.full((n, n), np.nan)
    rbc_grid = np.full((n, n), np.nan)
    p_values_list, rbc_list, indices = [], [], []

    for i in range(n):
        for j in range(i + 1, n):
            if len(data[i]) == 0 or len(data[j]) == 0:
                continue
            try:
                u_stat, p = mannwhitneyu(data[i], data[j], alternative='two-sided')
                p_values_list.append(p)
                indices.append((i, j))
                n1, n2 = len(data[i]), len(data[j])
                rbc = abs(1 - (2 * u_stat) / (n1 * n2))
                rbc_list.append(rbc)
            except ValueError:
                pass

    if len(p_values_list) > 0:
        reject_null, corrected_p_values, _, _ = multipletests(pvals=p_values_list, alpha=0.05, method='fdr_bh')
        for (i, j), p_corr, rbc in zip(indices, corrected_p_values, rbc_list):
            p_grid[i, j] = p_grid[j, i] = p_corr
            rbc_grid[i, j] = rbc_grid[j, i] = rbc

    return p_grid, rbc_grid


def clean_func(list1: list, list2: list) -> list:
    none_indices = ([i for i, v in enumerate(list1) if v is None] +
                    [i for i, v in enumerate(list2) if v is None])
    return [v for i, v in enumerate(list1) if i not in none_indices], [v for i, v in enumerate(list2) if
                                                                       i not in none_indices]


def split_basin_intensity(results: dict):
    results_al = {'wind_change': {f'{h:+}': {'pixel': [], 'count': []} for h in [-24, 24]}, 'id_list': []}
    results_ep = {'wind_change': {f'{h:+}': {'pixel': [], 'count': []} for h in [-24, 24]}, 'id_list': []}

    for idx, storm_id in enumerate(results['id_list']):
        target = results_al if 'AL' in storm_id else results_ep if 'EP' in storm_id else None
        if target:
            target['id_list'].append(storm_id)
            for h in target['wind_change']:
                target['wind_change'][h]['pixel'].append(results['wind_change'][h]['pixel'][idx])
                target['wind_change'][h]['count'].append(results['wind_change'][h]['count'][idx])
    return results_al, results_ep


def clean_all_intensity(results: dict) -> dict:
    for h in results['wind_change']:
        results['wind_change'][h]['pixel'], results['wind_change'][h]['count'] = clean_func(
            results['wind_change'][h]['pixel'], results['wind_change'][h]['count']
        )
    return results


# ================= Single-Axis Plotting Functions =================

def plot_violin_single(ax, data, labels, title, xlims, xticks, y_label=None):
    unique_labels = np.unique(np.concatenate([np.array(sublist) for sublist in labels]))
    parts = ax.violinplot(data, positions=unique_labels, showmeans=False, showmedians=False, showextrema=False,
                          widths=8)

    for pc in parts['bodies']:
        pc.set_facecolor('#FFA500')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1, medians, quartile3, whiskers_min, whiskers_max = [], [], [], [], []
    for group in data:
        group = np.sort(group)
        q1, med, q3 = np.percentile(group, [25, 50, 75])
        w_min, w_max = adjacent_values(group, q1, q3)
        quartile1.append(q1);
        medians.append(med);
        quartile3.append(q3)
        whiskers_min.append(w_min);
        whiskers_max.append(w_max)

    ax.scatter(unique_labels, medians, marker='o', color='blue', s=100, zorder=3)
    ax.vlines(unique_labels, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(unique_labels, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    ymin, ymax = min([np.min(d) for d in data]), max([np.max(d) for d in data])
    y_offset = 0.05 * (ymax - ymin)
    for i, d in enumerate(data):
        offset = y_offset if i % 2 == 0 else -1 * y_offset
        text_y = min(max(d) + offset, 100 - 0.05 * (ymax - ymin))
        ax.text(unique_labels[i], text_y, f'{len(d)}', ha='center', fontsize=18, fontweight='bold',
                rotation=90, va='center',
                path_effects=[path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

    ax.set_ylim((0, 60))
    ax.set_xlim(xlims)
    ax.set_xticks(xticks)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)

    line_rw = ax.axvline(x=-20, color='blue', linestyle='--', lw=2, label='RW Transition')
    line_ri = ax.axvline(x=30, color='magenta', linestyle='--', lw=2, label='RI Transition')

    ax.set_title(title, fontsize=20)
    if y_label: ax.set_ylabel(y_label, fontsize=20)

    return [line_rw, line_ri]


def plot_pvals_single(ax, data, labels, title, xlims, xticks, y_label=None):
    unique_labels = np.unique(np.concatenate([np.array(sublist) for sublist in labels]))
    p_grid, rbc_grid = compute_pval_rbc_grid(data)
    x, y = np.meshgrid(unique_labels, unique_labels)
    mask = (p_grid.ravel() < 0.05)
    x_masked, y_masked = x.ravel()[mask], y.ravel()[mask]

    high_mask = (x_masked > 30) | (y_masked > 30)
    low_mask = (x_masked < -20) | (y_masked < -20)
    both_mask = high_mask & low_mask
    red_mask = high_mask & ~low_mask
    blue_mask = low_mask & ~high_mask
    black_mask = ~(high_mask | low_mask)

    ax.scatter(x_masked[red_mask], y_masked[red_mask], c='red', s=150, label='> 30 kt bin')
    ax.scatter(x_masked[blue_mask], y_masked[blue_mask], c='blue', s=150, label='< -20 kt bin')
    ax.scatter(x_masked[black_mask], y_masked[black_mask], c='black', s=150, label='Other (p < 0.05)')

    for x_val, y_val in zip(x_masked[both_mask], y_masked[both_mask]):
        ax.plot(x_val, y_val, marker='o', markersize=np.sqrt(150), markerfacecolor='red',
                markerfacecoloralt='blue', markeredgecolor='black', markeredgewidth=1,
                fillstyle='left', linestyle='None', zorder=3)

    ax.set_xlim(xlims)
    ax.set_ylim(xlims)
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)
    ax.set_title(title, fontsize=20)
    if y_label: ax.set_ylabel(y_label, fontsize=20)


def plot_rbc_single(ax, data, labels, title, xlims, xticks, y_label=None):
    base = plt.get_cmap("coolwarm")
    cmap = LinearSegmentedColormap.from_list("coolwarm_trimmed", np.vstack([
        base(np.linspace(0.00, 0.30, 128)), base(np.linspace(0.70, 1.00, 128))
    ]))

    unique_labels = np.unique(np.concatenate([np.array(sublist) for sublist in labels]))
    p_grid, rbc_grid = compute_pval_rbc_grid(data)
    x, y = np.meshgrid(unique_labels, unique_labels)
    mask = (p_grid.ravel() < 0.05)
    x_masked, y_masked, rbc_masked = x.ravel()[mask], y.ravel()[mask], rbc_grid.ravel()[mask]

    for x_val, y_val, rbc_val in zip(x_masked, y_masked, rbc_masked):
        color = cmap(rbc_val)
        ax.text(x_val, y_val, f"{rbc_val * 100:.0f}", color=color, ha='center', va='center', fontweight='bold',
                fontsize=18)

    ax.set_xlim(xlims)
    ax.set_ylim(xlims)
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
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

    results = {
        'wind_change': {f'{h:+}': {'pixel': [], 'count': []} for h in [-24, 24]},
        'id_list': []
    }

    with Pool(12, initializer=init_worker) as pool:
        for result in pool.map(mp_running, needed_args):
            if result is None:
                continue
            wind_change_counts, pixels, atcf_id = result
            for h, wind_change in zip([-24, 24], wind_change_counts):
                rounded_wind = int(np.round(wind_change / 20) * 20) if wind_change is not None else None
                key = f'{h:+}'
                results['wind_change'][key]['pixel'].append(pixels)
                results['wind_change'][key]['count'].append(rounded_wind)
            results['id_list'].append(atcf_id)

    # Split & Clean
    results_al, results_ep = split_basin_intensity(results)
    results_al = clean_all_intensity(results_al)
    results_ep = clean_all_intensity(results_ep)

    # ================== Generate Plots =======================
    for h, main_title, x_lims, x_ticks, out_file in [
        ('-24', r'Previous TC Intensity Change, 20kt (20h)$^{-1}$', (-85, 85), range(-80, 81, 20), 'prev_intensity_combined.png'),
        ('+24', r'Future TC Intensity Change, 20kt (20h)$^{-1}$', (-105, 85), range(-100, 81, 20), 'future_intensity_combined.png')
    ]:
        data_al, labels_al = group_func(results_al['wind_change'][h]['pixel'], results_al['wind_change'][h]['count'])
        data_ep, labels_ep = group_func(results_ep['wind_change'][h]['pixel'], results_ep['wind_change'][h]['count'])

        fig, axes = plt.subplots(2, 3, figsize=(28, 16), sharex=True)
        fig.suptitle(f'CB Occurrences vs {main_title}', fontsize=32)

        # Labels for the axes
        ylab_violin = 'Percentage of Pixels with CBs'
        ylab_stat = r'TC Wind Speed Change ($\frac{dv}{dt}$)'
        xlab_all = r'TC Wind Speed Change ($\frac{dv}{dt}$)'

        # --- ROW 0: Atlantic (AL) ---
        title_al = f'AL: {main_title.split(" ")[0]} 24 Hours'
        violin_handles = plot_violin_single(axes[0, 0], data_al, labels_al, f'AL: Violin Plot', x_lims, x_ticks,
                                            y_label=ylab_violin)
        plot_pvals_single(axes[0, 1], data_al, labels_al, f'AL: Mann-Whitney P-Values', x_lims, x_ticks,
                          y_label=ylab_stat)
        plot_rbc_single(axes[0, 2], data_al, labels_al, f'AL: Rank-Biserial Correlations', x_lims, x_ticks,
                        y_label=ylab_stat)

        # --- ROW 1: East Pacific (EP) ---
        title_ep = f'EP: {main_title.split(" ")[0]} 24 Hours'
        plot_violin_single(axes[1, 0], data_ep, labels_ep, f'EP: Violin Plot', x_lims, x_ticks,
                           y_label=ylab_violin)
        plot_pvals_single(axes[1, 1], data_ep, labels_ep, f'EP: Mann-Whitney P-Values', x_lims, x_ticks,
                          y_label=ylab_stat)
        plot_rbc_single(axes[1, 2], data_ep, labels_ep, f'EP: Rank-Biserial Correlations', x_lims, x_ticks,
                        y_label=ylab_stat)

        # Add x-labels to the bottom row
        for j in range(3):
            axes[1, j].set_xlabel(xlab_all, fontsize=20)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])

        # Global Legends
        half_circle = Circle((0, 0), 1)
        pval_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=14, label='p<0.05 (RI)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=14, label='p<0.05 (RW)'),
            half_circle,
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=14, label='Other (p < 0.05)')
        ]
        pval_labels = ['p<0.05 (RI)', 'p<0.05 (RW)', 'p<0.05 (RI & RW)', 'p<0.05']

        # Add legend to top center
        fig.legend(handles=violin_handles + pval_handles,
                   labels=['RW Transition', 'RI Transition'] + pval_labels,
                   handler_map={half_circle: HandlerHalfCircle()},
                   loc='upper center', ncol=6, fontsize=20, bbox_to_anchor=(0.5, 0.93))

        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close('all')