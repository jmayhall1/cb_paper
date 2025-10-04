# coding=utf-8
"""
Optimized error statistics calculator for CNN-based tropical cyclone classification.
@author: John Mark Mayhall
Optimized: 10/01/2025

Features:
- Multiprocessing with worker-level model loading
- Efficient threshold-based confusion matrix calculation
- Optional error plotting with overlayed truth contours
- Basin-wise aggregation and metric calculation
"""

import glob
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from error_testing_setup import Setup
from keras.models import load_model


# ------------------- Initialization -------------------
def init_worker(model_path, c13_map, c8_map, hurdat, grid_dict):
    """
    Initializes each worker process with shared resources (CNN model, file maps, HURDAT, grid).
    Model loading is retried if it fails due to multiprocessing race conditions.
    """
    global model, c13_files, c8_files, hurdat_df, lonlat
    while True:
        try:
            model = load_model(model_path, compile=False)
            break
        except Exception as e:
            print(f"Model load failed in worker: {e}. Retrying...")
    c13_files = c13_map
    c8_files = c8_map
    hurdat_df = hurdat
    lonlat = grid_dict


def load_file_dict(path: str) -> dict:
    """Return dictionary mapping filename → full path."""
    return {Path(f).name: f for f in glob.glob(path)}


def parse_coordinate_series(series: pd.Series) -> pd.Series:
    """
    Convert lat/lon coordinate strings (e.g., '25.0N', '80.5W') into signed floats.
    """
    numeric = series.str[:-1].astype(float)
    direction = series.str[-1]
    sign = direction.map({'W': -1, 'S': -1, 'E': 1, 'N': 1})
    return numeric * sign


# ------------------- Worker Function -------------------
def process_file(file_info):
    """
    Worker function:
    - Loads prediction and truth for a given storm-time
    - Runs CNN forward pass
    - Compares prediction to truth across probability thresholds
    - Returns confusion matrix counts for all cutoffs
    """
    path = Path(file_info['file_path'])
    i, total, make_error_plot = file_info['i'], file_info['file_len'], file_info['error_plot']
    print(f"Processing file {i + 1} of {total}: {path.name}")

    # Parse filename
    parts = path.stem.split('_')
    s_id = parts[-1]
    date, time = parts[1][:8], parts[1][9:13]
    basin = 'AL' if 'AL' in s_id else 'EP'

    # Load scaled inputs
    x13 = np.load(c13_files[f'{s_id}_{date}_{time}_C13_scaled_cut.npz'])['brightness']
    x8 = np.load(c8_files[f'{s_id}_{date}_{time}_C08_scaled_cut.npz'])['brightness']

    # CNN forward pass
    pred = model([x13[None], x8[None]], training=False)[0, :, :, 0].numpy()

    # Ground truth mask
    storm = hurdat_df.query(f"ID == '{s_id}' and Date == {int(date)} and Time == {int(time)}")
    setup = Setup(str(path), *lonlat[basin], storm)
    truth = setup.main()

    # Optional error plot overlay
    if make_error_plot:
        latlon_files = load_file_dict("/rstor/jmayhall/cataloging/nc_process/geojson_transform/completed_arrays/"
                                      "latlon_arrs/*.npz")
        c13_unscaled_files = load_file_dict("/rstor/jmayhall/cataloging/nc_process/geojson_transform/completed_arrays/"
                                            "C13_unscaled/*.npz")

        lat = np.load(latlon_files[f'{s_id}_{date}_{time}_latlon.npz'])["lat"].astype(np.float32)
        lon = np.load(latlon_files[f'{s_id}_{date}_{time}_latlon.npz'])["lon"].astype(np.float32)
        x_unscaled = np.load(c13_unscaled_files[f'{s_id}_{date}_{time}_C13_unscaled_cut.npz'])["brightness"].astype(
            np.float32)

        fig, ax = plt.subplots(figsize=(12, 8))
        extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
        ax.imshow(x_unscaled, cmap='Greys', extent=extent, aspect='auto')
        ax.contour(np.flip(truth, axis=0), levels=1, colors='green', extent=extent, linewidths=5.0)
        plt.savefig(f'/rstor/jmayhall/cataloging/nc_process/error_test/error_images/{s_id}_{date}_{time}.png')
        plt.close(fig)

    # Flatten arrays for thresholding
    truth = truth.ravel()
    pred = pred.ravel()

    # Confusion matrix tallies for each cutoff
    out = {}
    for c in cutoffs:
        binary = (pred >= c).astype(np.uint8)
        tp = np.sum((binary == 1) & (truth == 1))
        tn = np.sum((binary == 0) & (truth == 0))
        fp = np.sum((binary == 1) & (truth == 0))
        fn = np.sum((binary == 0) & (truth == 1))
        out[c] = {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    return basin, out


# ------------------- Metrics -------------------
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Jaccard, Accuracy, POD, FAR from confusion matrix values."""
    df['Jaccard'] = df['TP'] / (df['TP'] + df['FP'] + df['FN'])
    df['Accuracy'] = (df['TP'] + df['TN']) / df[['TP', 'TN', 'FP', 'FN']].sum(axis=1)
    df['POD'] = df['TP'] / (df['TP'] + df['FN'])
    df['FAR'] = df['FP'] / (df['FP'] + df['TN'])
    return df.fillna(0)


# ------------------- Main -------------------
if __name__ == '__main__':
    # ------------------- Config -------------------
    MODEL_PATH = '/rstor/jmayhall/Model_Training_Code/cnn_creation/model.keras'
    GEOJSON_PATH = '/rstor/jmayhall/cataloging/nc_process/error_test/*.geojson'
    HURDAT_PATH = '/rstor/jmayhall/cataloging/hurdat_update_interp.txt'
    C8_PATH = '/rstor/jmayhall/cataloging/nc_process/geojson_transform/completed_arrays/C08_scaled/*'
    C13_PATH = '/rstor/jmayhall/cataloging/nc_process/geojson_transform/completed_arrays/C13_scaled/*'
    LONLAT_PATH_AL = glob.glob('/rstor/jmayhall/cataloging/nc_process/geojson_transform/latlon_arrays/AL*')[0]
    LONLAT_PATH_EP = \
        glob.glob('/rstor/jmayhall/cataloging/nc_process/geojson_transform/latlon_arrays/EP_latlon_17.npz*')[0]

    NUM_WORKERS = 24
    PLOT_RESULTS = True
    SAVE_ERROR_PLOTS = True
    PLOT_PATH = '/rstor/jmayhall/cataloging/nc_process/error_test/error_stats.png'

    # Probability cutoffs for binary classification
    cutoffs = np.arange(0, 1.01, 0.01)

    # ------------------- Shared Globals -------------------
    # These will be initialized once per worker
    model = None
    # Load HURDAT dataset
    hurdat_df = pd.read_csv(HURDAT_PATH, sep='\t')
    hurdat_df['Lon'] = parse_coordinate_series(hurdat_df['Lon'])
    hurdat_df['Lat'] = parse_coordinate_series(hurdat_df['Lat'])

    # File maps
    c8_files = load_file_dict(C8_PATH)
    c13_files = load_file_dict(C13_PATH)
    lonlat = {
        'AL': (np.load(LONLAT_PATH_AL)['lon'], np.load(LONLAT_PATH_AL)['lat']),
        'EP': (np.load(LONLAT_PATH_EP)['lon'], np.load(LONLAT_PATH_EP)['lat'])
    }

    # GeoJSON files to process
    files = glob.glob(GEOJSON_PATH)
    args = [{'file_path': f, 'i': i, 'file_len': len(files), 'error_plot': SAVE_ERROR_PLOTS}
            for i, f in enumerate(files)]

    # Parallel processing
    with Pool(NUM_WORKERS, initializer=init_worker,
              initargs=(MODEL_PATH, c13_files, c8_files, hurdat_df, lonlat)) as pool:
        results = pool.map(process_file, args)

    # Aggregate results
    value_df = {b: pd.DataFrame({'Cutoff': cutoffs, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}) for b in ['AL', 'EP']}
    for basin, stats in results:
        for c, row in stats.items():
            mask = value_df[basin]['Cutoff'] == c
            value_df[basin].loc[mask, ['TP', 'TN', 'FP', 'FN']] += [row['TP'], row['TN'], row['FP'], row['FN']]

    # Compute metrics and optionally plot
    for basin in ['AL', 'EP']:
        df = compute_metrics(value_df[basin])
        print(df)
        best = df.loc[df['Jaccard'].idxmax()]
        print(f"\nBest Cutoff {basin}: {best.Cutoff:.2f}, "
              f"Jaccard: {best.Jaccard:.3f}, Accuracy: {best.Accuracy:.3f}, "
              f"POD: {best.POD:.3f}, FAR: {best.FAR:.3f}")

        if PLOT_RESULTS:
            plt.plot(df['Cutoff'], df['Jaccard'], label=f'Jaccard ({basin})',
                     linestyle='-' if basin == 'AL' else 'dotted', color='blue')
            plt.plot(df['Cutoff'], df['POD'], label=f'POD ({basin})',
                     linestyle='-' if basin == 'AL' else 'dotted', color='orange')
            plt.plot(df['Cutoff'], df['FAR'], label=f'FAR ({basin})',
                     linestyle='-' if basin == 'AL' else 'dotted', color='green')

    if PLOT_RESULTS:
        plt.xlabel("Probability Threshold", fontsize=14)
        plt.ylabel("Score", fontsize=14)
        plt.title("Model Performance for 2019 TCs\n(Atlantic and Eastern Pacific)", fontsize=16)
        plt.legend()
        plt.savefig(PLOT_PATH)
        plt.close()
