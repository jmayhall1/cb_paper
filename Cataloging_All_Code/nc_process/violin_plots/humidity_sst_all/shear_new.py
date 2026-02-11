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


def fig_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)


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


def prepare_rh_data(results_dict):
    """Prepare RH data for plotting."""
    data_list = [group_func(results_dict[f'rh{h}_pixel'], results_dict[f'rh{h}_count'])[0] for h in rh_ids]
    labels_list = [group_func(results_dict[f'rh{h}_pixel'], results_dict[f'rh{h}_count'])[1] for h in rh_ids]
    return data_list, labels_list


# ---------------- Plotting Functions ---------------- #
def plot_violin_2panel(data1, labels1, data2, labels2, suptitle, xlabel, ylabel, filename):
    """
    Plot two side-by-side violin plots: violins + quartiles
    """

    def plot_single_violin(ax: plt.axis, data: list, labels: list, panel_title: str):
        bin_centers = list(range(18, 33, 1))
        bins = defaultdict(list)
        for group, label_group in zip(data, labels):
            for val, lab in zip(group, label_group):
                binned_lab = lab
                bins[binned_lab].append(val)

        # --- Prepare violin data ---
        violin_data = [bins[bc] for bc in bin_centers if bc in bins]
        violin_positions = [bc for bc in bin_centers if bc in bins]

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
            showmedians=False, showextrema=False, widths=0.6
        )
        for pc in parts['bodies']:
            pc.set_facecolor('#FFA500')
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        ax.scatter(violin_positions, medians, color='blue', s=80, zorder=3)
        ax.vlines(violin_positions, quartile1, quartile3, color='k', lw=5)
        ax.vlines(violin_positions, whiskers_min, whiskers_max, color='k', lw=2)

        # --- Formatting ---
        ax.set_xlim((17, 33))
        ax.set_xticks(range(18, 33, 1))
        ax.set_xticklabels(np.arange(18, 33, 1), rotation=45, ha='right')
        ax.set_ylim((0, 60))
        ax.set_title(panel_title, fontsize=28)
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)

        # --- Sample counts ---
        for pos, values in zip(violin_positions, violin_data):
            if len(values) == 0:
                continue
            y = min(max(values) + 2, 60)
            ax.text(
                pos, y, f'{len(values)}', ha='center', fontsize=20, fontweight='bold',
                rotation=90, va='center',
                path_effects=[path_effects.Stroke(linewidth=2, foreground='white'),
                              path_effects.Normal()]
            )

    # --- Create horizontal 2-panel plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plot_single_violin(axes[0], data1, labels1, 'Atlantic')
    plot_single_violin(axes[1], data2, labels2, 'Eastern Pacific')

    fig.suptitle(suptitle, fontsize=28)
    fig.supxlabel(xlabel, fontsize=28)
    fig.supylabel(ylabel, fontsize=28, x=0.05)
    plt.savefig(filename)
    return fig, axes


def plot_contourf_2panel(data1: list, labels1: list, data2: list, labels2: list, suptitle: str, xlabel: str,
                         ylabel: str, filename: str):
    """
    Plot p-values
    """

    def bin_data(data: list, labels: list):
        """
        Bin data
        """
        bins = defaultdict(list)  # key: (x_bin, y_bin) => list of values

        for group, label_group in zip(data, labels):
            for val, (x_lab, y_lab) in zip(group, zip(label_group, label_group)):
                x_bin = (int(x_lab))
                y_bin = (int(y_lab))
                bins[(x_bin, y_bin)].append(val)

        bin_centers = list(range(18, 33, 1))
        grid = [[bins.get((x, y), []) for x in bin_centers] for y in bin_centers]
        return grid, bin_centers

    # Bin and compute p-value grids
    binned1, centers1 = bin_data(data1, labels1)
    binned2, centers2 = bin_data(data2, labels2)

    pvals1 = compute_pval_grid(binned1)
    pvals2 = compute_pval_grid(binned2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Common tick marks
    tick_marks = range(18, 33, 1)

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
    axes[0].set_xlim((17, 33))
    axes[0].set_ylim((17, 33))
    axes[0].set_xticks(tick_marks)
    axes[0].set_yticks(tick_marks)
    axes[0].set_xticklabels(tick_marks, rotation=45, ha='right')
    axes[0].set_yticklabels(tick_marks)
    axes[0].tick_params(axis='both', labelsize=18)
    axes[0].set_title('Atlantic', fontsize=28)
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
    axes[1].set_xlim((17, 33))
    axes[1].set_ylim((17, 33))
    axes[1].set_xticks(tick_marks)
    axes[1].set_yticks(tick_marks)
    axes[1].set_xticklabels(tick_marks, rotation=45, ha='right')
    axes[1].set_yticklabels(tick_marks)
    axes[1].tick_params(axis='both', labelsize=18)
    axes[1].set_title('Eastern Pacific', fontsize=28)
    axes[1].grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)

    # Final layout
    fig.subplots_adjust(bottom=0.25)
    fig.suptitle(suptitle, fontsize=28, y=0.98)
    fig.supxlabel(xlabel, fontsize=28, y=0.13)
    fig.supylabel(ylabel, fontsize=28, x=0.05)

    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=20, framealpha=0)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_rh_violin_ax(ax, data, labels, title, rh_lims):
    start, end, step = rh_lims
    bin_centers = list(range(start, end, step))
    bins = defaultdict(list)

    for group, label_group in zip(data, labels):
        for val, lab in zip(group, label_group):
            binned_lab = (10 * (lab // 10))
            bins[binned_lab].append(val)

    violin_positions = np.array(sorted(bins.keys()), dtype=float)
    violin_data = [np.asarray(bins[bc], dtype=float) for bc in violin_positions]

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

        quartile1.append(q1)
        medians.append(med)
        quartile3.append(q3)
        whiskers_min.append(whisk_min)
        whiskers_max.append(whisk_max)

    parts = ax.violinplot(
        violin_data, positions=violin_positions,
        showmeans=False, showmedians=False, showextrema=False, widths=6
    )

    for pc in parts['bodies']:
        pc.set_facecolor('#FFA500')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    ax.scatter(violin_positions, medians, color='blue', s=80, zorder=3)
    ax.vlines(violin_positions, quartile1, quartile3, color='k', lw=6)
    ax.vlines(violin_positions, whiskers_min, whiskers_max, color='k', lw=2)
    ax.tick_params(axis='both', labelsize=18)

    # --- Sample size labels ---
    for pos, values in zip(violin_positions, violin_data):
        if len(values) == 0:
            continue

        y = min(np.nanmax(values) + 2, 58)

        ax.text(
            pos, y, f'{len(values)}',
            ha='center',
            va='center',
            fontsize=20,
            fontweight='bold',
            rotation=90,
            path_effects=[
                path_effects.Stroke(linewidth=2, foreground='white'),
                path_effects.Normal()
            ],
            zorder=5
        )

    ax.set_xlim((start - 5, end + 4))
    ax.set_xticks(range(start, end, step))
    ax.set_ylim((0, 60))
    ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)
    ax.set_title(title, fontsize=24)


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
    data_AL, labels_AL = group_func(results_AL['sst_pixel'], results_AL['sst_count'])
    data_EP, labels_EP = group_func(results_EP['sst_pixel'], results_EP['sst_count'])
    fig1, _ = plot_violin_2panel(data_AL, labels_AL, data_EP, labels_EP,
                       suptitle='TCB Occurrences vs SST',
                       xlabel='SST (C)',
                       ylabel='Percentage of Storm Pixels with TCBs',
                       filename='sst_violin_ALEP.png')
    fig2, _ = plot_contourf_2panel(data_AL, labels_AL, data_EP, labels_EP,
                         suptitle='SST Mann-Whitney P-Values',
                         xlabel='SST (C)',
                         ylabel='SST (C)',
                         filename='sst_mannwhitney_ALEP.png')

    img1 = fig_to_rgb(fig1)
    img2 = fig_to_rgb(fig2)

    fig, ax = plt.subplots(
        figsize=(img1.shape[1] / 100, (img1.shape[0] + img2.shape[0]) / 100)
    )

    ax.imshow(np.vstack([img1, img2]))
    ax.axis("off")

    plt.savefig("sst_combined.png", dpi=300, bbox_inches="tight")
    plt.close('all')

    # ------------------ Plot RH ------------------ #
    rh_ids = ['lo', 'md', 'hi']
    rh_labels = rh_titles = ['850-700 hPa RH', '700-500 hPa RH', '500-300 hPa RH']
    rh_lims = [(40, 91, 10), (30, 91, 10), (20, 91, 10)]
    data_AL_rh, labels_AL_rh = prepare_rh_data(results_AL)
    data_EP_rh, labels_EP_rh = prepare_rh_data(results_EP)
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    for col in range(3):
        # Atlantic (top row)
        plot_rh_violin_ax(
            axes[0, col],
            data_AL_rh[col],
            labels_AL_rh[col],
            f'Atlantic: {rh_titles[col]}',
            rh_lims[col]
        )

        # Eastern Pacific (bottom row)
        plot_rh_violin_ax(
            axes[1, col],
            data_EP_rh[col],
            labels_EP_rh[col],
            f'Eastern Pacific: {rh_titles[col]}',
            rh_lims[col]
        )

    fig.suptitle('TCB Occurrences vs RH', fontsize=28)
    fig.supxlabel('RH (%)', fontsize=28)
    fig.supylabel('Percentage of Storm Pixels with TCBs', fontsize=28)

    plt.savefig('rh_ALEP.png', dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    for col in range(3):
        for row, (data, labels, basin) in enumerate([
            (data_AL_rh[col], labels_AL_rh[col], 'Atlantic'),
            (data_EP_rh[col], labels_EP_rh[col], 'Eastern Pacific')
        ]):
            pvals = compute_pval_grid(data)
            x = y = sorted(set(v for sub in labels for v in sub))
            X, Y = np.meshgrid(x, y)

            mask = pvals < 0.05
            axes[row, col].scatter(X[mask], Y[mask], c='black', s=200, label='p < 0.05')
            axes[row, col].set_title(f'{basin}: {rh_titles[col]}', fontsize=24)
            axes[row, col].grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)
            axes[row, col].set_xlim(rh_lims[col][0] - 5, rh_lims[col][1] + 4)
            axes[row, col].set_ylim(rh_lims[col][0] - 5, rh_lims[col][1] + 4)
            axes[row, col].tick_params(axis='both', labelsize=18)

    fig.suptitle('RH Mann–Whitney P-Values', fontsize=28)
    fig.supxlabel('RH (%)', fontsize=28)
    fig.supylabel('RH (%)', fontsize=28)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=28, framealpha=0)

    plt.savefig('rh_ALEP_mannwhitney.png', dpi=300, bbox_inches="tight")
    plt.close()
