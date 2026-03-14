# coding=utf-8
"""
Last Edited: 07/10/2025
@author: John Mark Mayhall
Purpose: Identify transverse bands in TC quadrants based on shear vector.
"""
import glob
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


def label_every_20(x, pos):
    """
    Function for creating labels every 20 points.
    :param x: Tick marks
    :return: Tick labels
    """
    return f"{int(x)}" if x % 20 == 0 else ''


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



def adjacent_values(sorted_array, q1, q3):
    """Calculate adjacent values for whiskers in box/violin plots."""
    iqr = q3 - q1
    upper_adj = q3 + 1.5 * iqr
    lower_adj = q1 - 1.5 * iqr
    upper = max([x for x in sorted_array if x <= upper_adj], default=q3)
    lower = min([x for x in sorted_array if x >= lower_adj], default=q1)
    return lower, upper

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
    'wind_change': {f'{h:+}': {'pixel': [], 'count': []} for h in [-24, 24]},
    'id_list': []
}

with Pool(12, initializer=init_worker) as pool:
    for result in pool.map(mp_running, needed_args):
        if result is None:
            continue

        wind_change_counts, pixels, atcf_id = result

        for h, wind_change in zip([-24, 24], wind_change_counts):
            if wind_change is not None:
                rounded_wind = int(np.round(wind_change / 20) * 20)
            else:
                rounded_wind = None
            key = f'{h:+}'  # "+6", "-24", etc.
            results['wind_change'][key]['pixel'].append(pixels)
            results['wind_change'][key]['count'].append(rounded_wind)
        results['id_list'].append(atcf_id)

# Initialize new dictionaries for AL and EP
results_AL = {'wind_change': {f'{h:+}': {'pixel': [], 'count': []} for h in [-24, 24]}, 'id_list': []}

results_EP = {'wind_change': {f'{h:+}': {'pixel': [], 'count': []} for h in [-24, 24]}, 'id_list': []}

for idx, storm_id in enumerate(results['id_list']):
    if 'AL' in storm_id:
        results_AL['id_list'].append(storm_id)
        for h in results_AL['wind_change']:
            results_AL['wind_change'][h]['pixel'].append(results['wind_change'][h]['pixel'][idx])
            results_AL['wind_change'][h]['count'].append(results['wind_change'][h]['count'][idx])
    elif 'EP' in storm_id:
        results_EP['id_list'].append(storm_id)
        for h in results_EP['wind_change']:
            results_EP['wind_change'][h]['pixel'].append(results['wind_change'][h]['pixel'][idx])
            results_EP['wind_change'][h]['count'].append(results['wind_change'][h]['count'][idx])
for h in results_AL['wind_change']:
    results_AL['wind_change'][h]['pixel'], results_AL['wind_change'][h]['count'] = (
        clean_func(results_AL['wind_change'][h]['pixel'], results_AL['wind_change'][h]['count']))
for h in results['wind_change']:
    results_EP['wind_change'][h]['pixel'], results_EP['wind_change'][h]['count'] = (
        clean_func(results_EP['wind_change'][h]['pixel'], results_EP['wind_change'][h]['count']))

# =========== Plot: Wind Change (Past) ============
past_hours = [-24]
data_list = ([group_func(results_AL['wind_change'][f'{h:+}']['pixel'],
                         results_AL['wind_change'][f'{h:+}']['count'])[0] for h in past_hours] +
             [group_func(results_EP['wind_change'][f'{h:+}']['pixel'],
                         results_EP['wind_change'][f'{h:+}']['count'])[0] for h in past_hours])
x_labels = ([group_func(results_AL['wind_change'][f'{h:+}']['pixel'],
                        results_AL['wind_change'][f'{h:+}']['count'])[1] for h in past_hours] +
            [group_func(results_EP['wind_change'][f'{h:+}']['pixel'],
                        results_EP['wind_change'][f'{h:+}']['count'])[1] for h in past_hours])
labels_list = [f"{h}hr" for h in past_hours] + [f"{h}hr" for h in past_hours]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('TCB Occurrences vs Previous TC Intensity Change',
             fontsize=24, y=1)
fig.supxlabel(r'TC Wind Speed Change ($\frac{{dv}}{{dt}}$)', fontsize=24)
fig.supylabel('Percentage of Pixels with TCBs', fontsize=24)
for z, (ax, data, label, labels) in enumerate(zip(axes.flatten(), data_list, labels_list, x_labels)):
    labels = np.unique(np.concatenate([np.array(sublist) for sublist in labels]))
    ax.tick_params(axis='both', labelsize=16)
    parts = ax.violinplot(data, positions=labels, showmeans=False, showmedians=False, showextrema=False, widths=8)

    for pc in parts['bodies']:
        pc.set_facecolor('#FFA500')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    # Compute quartiles and whiskers for each group manually
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

    inds = labels
    ax.scatter(inds, medians, marker='o', color='blue', s=100, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
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
        ax.set_xlim((-85, 85))
        ax.set_xticks(range(-80, 81, 20))
    ax.xaxis.set_major_formatter(FuncFormatter(label_every_20))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    # Ensure there's enough space at the top
    ymin = min([np.min(d) for d in data])
    ymax = max([np.max(d) for d in data])
    y_offset = 0.05 * (ymax - ymin)
    fig_ylim_top = 100

    # Now safely add text without going above the y-limit
    for i, d in enumerate(data):
        offset = y_offset if i % 2 == 0 else -1 * y_offset
        text_y = min(max(d) + offset, fig_ylim_top - 0.05 * (ymax - ymin))  # small buffer
        ax.text(labels[i], text_y, f'{len(d)}', ha='center', fontsize=16, fontweight='bold', rotation=90, va='center',
                path_effects=[path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
    ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)
    ax.axvline(x=-20, color='blue', linestyle='--', lw=2, label='RW Transition')
    ax.axvline(x=30, color='magenta', linestyle='--', lw=2, label='RI Transition')
    if z == 0:
        ax.set_title(f'Intensity Change over\n the Previous {label[1:-2]} Hours ' + rf'(Atlantic, kt {label[1:]}$^{{-1}}$)', fontsize=16)
    else:
        ax.set_title(f'Intensity Change over\n the Previous {label[1:-2]} Hours ' + rf'(Eastern Pacific, kt {label[1:]}$^{{-1}}$)', fontsize=16)

    # Add legend after the lines are drawn
fig.subplots_adjust(
    left=0.07,  # space for ylabel
    right=0.92,  # space for legend or vertical colorbar
    bottom=0.15,  # space for supxlabel
    top=0.88,  # space for suptitle
    hspace=0.4,  # vertical spacing between rows
    wspace=0.3  # horizontal spacing between columns
)
handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper right', fontsize=12)
legend.set_alpha(0.4)
plt.savefig('prev_intensity_change_ALEP.png')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Previous TC Intensity Change Mann-Whitney P-Values', fontsize=24, y=1)
fig.supxlabel(r'TC Wind Speed Change ($\frac{dv}{dt}$)', fontsize=24, y=0.07)
fig.supylabel(r'TC Wind Speed Change ($\frac{dv}{dt}$)', fontsize=24, x=0.05)

# Loop through each subplot
for z, (ax, data, label, labels) in enumerate(zip(axes.flatten(), data_list, labels_list, x_labels)):
    labels = np.unique(np.concatenate([np.array(sublist) for sublist in labels]))

    # Compute p-value matrix
    p_grid = compute_pval_grid(data)
    # Create contourf plot
    x, y = np.meshgrid(labels, labels)
    z_pgrid = p_grid
    # Flatten everything
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = z_pgrid.ravel()
    mask = z_flat < 0.05
    x_masked = x_flat[mask]
    y_masked = y_flat[mask]
    # Plot only black dots for z < 0.05
    # --- classification masks ---
    high_mask = (x_masked > 30) | (y_masked > 30)
    low_mask = (x_masked < -20) | (y_masked < -20)

    both_mask = high_mask & low_mask
    red_mask = high_mask & ~low_mask
    blue_mask = low_mask & ~high_mask
    black_mask = ~(high_mask | low_mask)

    # --- normal markers ---
    ax.scatter(x_masked[red_mask], y_masked[red_mask], c='red', s=150, label='> 30 kt bin')
    ax.scatter(x_masked[blue_mask], y_masked[blue_mask], c='blue', s=150, label='< -20 kt bin')
    ax.scatter(x_masked[black_mask], y_masked[black_mask], c='black', s=150, label='Other (p < 0.05)')

    for x_val, y_val in zip(x_masked[both_mask], y_masked[both_mask]):
        marker_size = np.sqrt(150)

        ax.plot(
            x_val, y_val,
            marker='o',
            markersize=marker_size,
            markerfacecolor='red',  # left half
            markerfacecoloralt='blue',  # right half
            markeredgecolor='black',
            markeredgewidth=1,
            fillstyle='left',
            linestyle='None',
            zorder=3
        )
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels(labels, fontsize=14)
    if '6' in label:
        ax.set_xlim((-65, 45))
        ax.set_ylim((-65, 45))
        tick_marks = range(-60, 41, 20)
    elif '12' in label:
        ax.set_xlim((-105, 65))
        ax.set_ylim((-105, 65))
        tick_marks = range(-100, 61, 20)
    elif '18' in label:
        ax.set_xlim((-85, 85))
        ax.set_ylim((-85, 85))
        tick_marks = range(-80, 81, 20)
    elif '24' in label:
        ax.set_xlim((-85, 85))
        ax.set_ylim((-85, 85))
        tick_marks = range(-80, 81, 20)
    ax.set_xticks(tick_marks)
    ax.xaxis.set_major_formatter(FuncFormatter(label_every_20))
    ax.set_yticks(tick_marks)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.yaxis.set_major_formatter(FuncFormatter(label_every_20))
    plt.setp(ax.get_yticklabels())
    ax.set_title(f'{label}', fontsize=16)
    ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)
    if z == 0:
        ax.set_title(f'Intensity Change over\n the Previous {label[1:-2]} Hours ' + rf'(Atlantic, kt {label[1:]}$^{{-1}}$)', fontsize=16)
    else:
        ax.set_title(f'Intensity Change over\n the Previous {label[1:-2]} Hours ' + rf'(Eastern Pacific, kt {label[1:]}$^{{-1}}$)', fontsize=16)

# Add colorbar
fig.subplots_adjust(
    left=0.12,  # space for ylabel
    right=0.92,  # space for legend or vertical colorbar
    bottom=0.25,  # space for supxlabel
    top=0.85,  # space for suptitle
)
half_circle = Circle((0,0), 1)

legend_handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='red',
           markersize=12, label='p<0.05 (RI)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='blue',
           markersize=12, label='p<0.05 (RW)'),
    half_circle,
    Line2D([0],[0], marker='o', color='w', markerfacecolor='black',
           markersize=12, label='Other (p < 0.05)')
]

legend_labels = [
    'p<0.05 (RI)',
    'p<0.05 (RW)',
    'p<0.05 (RI & RW)',
    'p<0.05'
]

fig.legend(
    legend_handles,
    legend_labels,
    handler_map={half_circle: HandlerHalfCircle()},
    loc='upper right',
    fontsize=12
)
# Adjust spacing to make room for colorbar at the bottom

# Create a new axis for the horizontal colorbar that spans the full width
plt.savefig('prev_intensity_change_ALEP_mannwhitney.png')
plt.close()

# =========== Plot: Wind Change (Future) ==========
future_hours = [24]
data_list = ([group_func(results_AL['wind_change'][f'{h:+}']['pixel'],
                         results_AL['wind_change'][f'{h:+}']['count'])[0] for h in future_hours] +
             [group_func(results_EP['wind_change'][f'{h:+}']['pixel'],
                         results_EP['wind_change'][f'{h:+}']['count'])[0] for h in future_hours])
x_labels = ([group_func(results_AL['wind_change'][f'{h:+}']['pixel'],
                        results_AL['wind_change'][f'{h:+}']['count'])[1] for h in future_hours] +
            [group_func(results_EP['wind_change'][f'{h:+}']['pixel'],
                        results_EP['wind_change'][f'{h:+}']['count'])[1] for h in future_hours])
labels_list = [f"{h}hr" for h in future_hours] + [f"{h}hr" for h in future_hours]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('TCB Occurrences vs Future TC Intensity Change',
             fontsize=24, y=1)
fig.supxlabel(r'TC Wind Speed Change ($\frac{{dv}}{{dt}}$)', fontsize=24)
fig.supylabel('Percentage of Pixels with TCBs', fontsize=24)
for z, (ax, data, label, labels) in enumerate(zip(axes.flatten(), data_list, labels_list, x_labels)):
    labels = np.unique(np.concatenate([np.array(sublist) for sublist in labels]))
    ax.tick_params(axis='both', labelsize=16)
    parts = ax.violinplot(data, positions=labels, showmeans=False, showmedians=False, showextrema=False, widths=8)

    for pc in parts['bodies']:
        pc.set_facecolor('#FFA500')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    # Compute quartiles and whiskers for each group manually
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

    inds = labels
    ax.scatter(inds, medians, marker='o', color='blue', s=100, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
    ax.set_ylim((0, 60))
    if '6' in label:
        ax.set_xlim((-65, 45))
        ax.set_xticks(range(-60, 41, 20))
    elif '12' in label:
        ax.set_xlim((-105, 65))
        ax.set_xticks(range(-100, 61, 20))
    elif '18' in label:
        ax.set_xlim((-105, 85))
        ax.set_xticks(range(-100, 81, 20))
    elif '24' in label:
        ax.set_xlim((-105, 85))
        ax.set_xticks(range(-100, 81, 20))
    ax.xaxis.set_major_formatter(FuncFormatter(label_every_20))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    # Ensure there's enough space at the top
    ymin = min([np.min(d) for d in data])
    ymax = max([np.max(d) for d in data])
    y_offset = 0.05 * (ymax - ymin)
    fig_ylim_top = 100

    # Now safely add text without going above the y-limit
    for i, d in enumerate(data):
        offset = y_offset if i % 2 == 0 else -1 * y_offset
        text_y = min(max(d) + offset, fig_ylim_top - 0.05 * (ymax - ymin))  # small buffer
        ax.text(labels[i], text_y, f'{len(d)}', ha='center', fontsize=16, fontweight='bold', rotation=90, va='center',
                path_effects=[path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
    ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)
    ax.axvline(x=-20, color='blue', linestyle='--', lw=2, label='RW Transition')
    ax.axvline(x=30, color='magenta', linestyle='--', lw=2, label='RI Transition')
    if z == 0:
        ax.set_title(f'Intensity Change over\n the Next {label[:-2]} Hours ' + rf'(Atlantic, kt {label}$^{{-1}}$)', fontsize=16)
    else:
        ax.set_title(f'Intensity Change over\n the Next {label[:-2]} Hours ' + rf'(Eastern Pacific, kt {label}$^{{-1}}$)', fontsize=16)

    # Add legend after the lines are drawn
fig.subplots_adjust(
    left=0.07,  # space for ylabel
    right=0.92,  # space for legend or vertical colorbar
    bottom=0.15,  # space for supxlabel
    top=0.88,  # space for suptitle
    hspace=0.4,  # vertical spacing between rows
    wspace=0.3  # horizontal spacing between columns
)
handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper right', fontsize=12)
legend.set_alpha(0.4)
plt.savefig('future_intensity_change_ALEP.png')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Future TC Intensity Change Mann-Whitney P-Values', fontsize=24, y=1)
fig.supxlabel(r'TC Wind Speed Change ($\frac{dv}{dt}$)', fontsize=24, y=0.12)
fig.supylabel(r'TC Wind Speed Change ($\frac{dv}{dt}$)', fontsize=24, x=0.05)

# Loop through each subplot
for z, (ax, data, label, labels) in enumerate(zip(axes.flatten(), data_list, labels_list, x_labels)):
    labels = np.unique(np.concatenate([np.array(sublist) for sublist in labels]))

    # Compute p-value matrix
    p_grid = compute_pval_grid(data)
    # Create contourf plot
    x, y = np.meshgrid(labels, labels)
    z_pgrid = p_grid
    # Flatten everything
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = z_pgrid.ravel()
    mask = z_flat < 0.05
    x_masked = x_flat[mask]
    y_masked = y_flat[mask]
    # Plot only black dots for z < 0.05
    # --- classification masks ---
    high_mask = (x_masked > 30) | (y_masked > 30)
    low_mask = (x_masked < -20) | (y_masked < -20)

    both_mask = high_mask & low_mask
    red_mask = high_mask & ~low_mask
    blue_mask = low_mask & ~high_mask
    black_mask = ~(high_mask | low_mask)

    # --- normal markers ---
    ax.scatter(x_masked[red_mask], y_masked[red_mask], c='red', s=150, label='> 30 kt bin')
    ax.scatter(x_masked[blue_mask], y_masked[blue_mask], c='blue', s=150, label='< -20 kt bin')
    ax.scatter(x_masked[black_mask], y_masked[black_mask], c='black', s=150, label='Other (p < 0.05)')

    for x_val, y_val in zip(x_masked[both_mask], y_masked[both_mask]):
        marker_size = np.sqrt(150)

        ax.plot(
            x_val, y_val,
            marker='o',
            markersize=marker_size,
            markerfacecolor='red',  # left half
            markerfacecoloralt='blue',  # right half
            markeredgecolor='black',
            markeredgewidth=1,
            fillstyle='left',
            linestyle='None',
            zorder=3
        )
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels(labels, fontsize=14)
    if '6' in label:
        ax.set_xlim((-65, 45))
        ax.set_ylim((-65, 45))
        tick_marks = range(-60, 41, 20)
    elif '12' in label:
        ax.set_xlim((-105, 65))
        ax.set_ylim((-105, 65))
        tick_marks = range(-100, 61, 20)
    elif '18' in label:
        ax.set_xlim((-105, 85))
        ax.set_ylim((-105, 85))
        tick_marks = range(-100, 81, 20)
    elif '24' in label:
        ax.set_xlim((-105, 85))
        ax.set_ylim((-105, 85))
        tick_marks = range(-100, 81, 20)
    ax.set_xticks(tick_marks)
    ax.xaxis.set_major_formatter(FuncFormatter(label_every_20))
    ax.set_yticks(tick_marks)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.yaxis.set_major_formatter(FuncFormatter(label_every_20))
    plt.setp(ax.get_yticklabels())
    ax.set_title(f'{label}', fontsize=16)
    ax.grid(True, color='black', linestyle='--', linewidth=1.0, alpha=1)
    if z == 0:
        ax.set_title(f'Intensity Change over\n the Next {label[:-2]} Hours ' + rf'(Atlantic, kt {label}$^{{-1}}$)', fontsize=16)
    else:
        ax.set_title(f'Intensity Change over\n the Next {label[:-2]} Hours ' + rf'(Eastern Pacific, kt {label}$^{{-1}}$)', fontsize=16)

# Add colorbar
# Adjust spacing to make room for colorbar at the bottom
fig.subplots_adjust(
    left=0.12,  # space for ylabel
    right=0.92,  # space for legend or vertical colorbar
    bottom=0.25,  # space for supxlabel
    top=0.85,  # space for suptitle
    hspace=0.4,  # vertical spacing between rows
    wspace=0.3  # horizontal spacing between columns
)
half_circle = Circle((0,0), 1)

legend_handles = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='red',
           markersize=12, label='p<0.05 (RI)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='blue',
           markersize=12, label='p<0.05 (RW)'),
    half_circle,
    Line2D([0],[0], marker='o', color='w', markerfacecolor='black',
           markersize=12, label='Other (p < 0.05)')
]

legend_labels = [
    'p<0.05 (RI)',
    'p<0.05 (RW)',
    'p<0.05 (RI & RW)',
    'p<0.05'
]

fig.legend(
    legend_handles,
    legend_labels,
    handler_map={half_circle: HandlerHalfCircle()},
    loc='upper right',
    fontsize=12
)
# Adjust spacing to make room for colorbar at the bottom

# Create a new axis for the horizontal colorbar that spans the full width
plt.savefig('future_intensity_change_ALEP_mannwhitney.png')
plt.close()
