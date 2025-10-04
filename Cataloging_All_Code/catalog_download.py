# coding=utf-8
"""
Optimized: 10/01/2025
@author: John Mark Mayhall

Description:
------------
This script downloads satellite files listed in HURDAT2 records
for the AL (Atlantic) and EP (Eastern Pacific) basins. It:
    1. Loads and processes HURDAT2 track data.
    2. Interpolates missing synoptic (6-hourly) records.
    3. Generates multiprocessing arguments for satellite downloading.
    4. Runs parallel downloading for multiple channels and satellites.

Requirements:
-------------
- pandas
- multiprocessing
- local module: downloading_setup_folder (contains mp_downloading, init_worker)
"""

from multiprocessing import Pool

import pandas as pd
from downloading_setup_folder import mp_downloading, init_worker

# Configuration
CHANNELS = ['13', '08']  # GOES IR bands to download
NUM_WORKERS = 192  # Parallel worker count
COLUMN_NAMES = [
    'Date', 'Time', 'Extra', 'Type', 'Lat', 'Lon', 'Extra2', 'Pressure',
    'Extra3', 'Extra4', 'Extra5', 'Extra6', 'Extra7', 'Extra8', 'Extra9',
    'Extra10', 'Extra11', 'Extra12', 'Extra13', 'Extra14', 'Extra15'
]


def load_and_prepare_df(path: str) -> pd.DataFrame:
    """
    Load a HURDAT2 file and prepare it for analysis:
      - Extract storm IDs and names from header rows.
      - Filter to synoptic times (00, 06, 12, 18 UTC).

    Parameters
    ----------
    path : str
        Path to HURDAT2 file.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with columns ['ID', 'Name', ...].
    """
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)

    ids, names = [], []
    for _, row in df.iterrows():
        # Header row: contains ID and storm name, not track info
        if pd.isna(row.Extra2):
            count = int(row.Extra)  # number of records for this storm
            ids.extend([row.Date] * count)
            names.extend([row.Time.strip()] * count)

    # Keep only track rows (Lat non-null) and attach ID/Name
    df = df[df.Lat.notnull()].reset_index(drop=True)
    df.insert(0, 'Name', names)
    df.insert(0, 'ID', ids)

    # Keep only synoptic hours (00, 06, 12, 18 UTC)
    df['Time'] = df['Time'].astype(str).str.zfill(4)
    df = df[df['Time'].str[-2:] == '00'].reset_index(drop=True)

    return df


def interpolate_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert placeholder rows for missing 6-hour intervals in a storm's track.

    Parameters
    ----------
    df : pd.DataFrame
        Track DataFrame with 6-hourly (synoptic) data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with interpolated empty rows for missing synoptic times.
    """
    output_rows = []

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        output_rows.append(row)

        if row.ID == next_row.ID:
            t1, t2 = int(row.Time), int(next_row.Time or 2400)
            t1_hr, t2_hr = t1 // 100, t2 // 100
            gap = min(max(t2_hr - t1_hr, 1), 6)  # Fill gaps up to 6 hours

            for j in range(1, gap):
                new_time = (t1_hr + j) * 100
                if new_time < t2:
                    output_rows.append(pd.Series({
                        'ID': row.ID,
                        'Name': row.Name,
                        'Date': row.Date,
                        'Time': str(new_time).zfill(4),
                        **{col: '' for col in COLUMN_NAMES[3:]}  # empty placeholders
                    }))

    output_rows.append(df.iloc[-1])  # Add last row
    return pd.DataFrame(output_rows)


def generate_args(df: pd.DataFrame, sat_map: dict) -> list:
    """
    Create multiprocessing job arguments for downloading.

    Parameters
    ----------
    df : pd.DataFrame
        Track DataFrame.
    sat_map : dict
        Mapping of year -> satellite number.

    Returns
    -------
    list
        A list of argument dictionaries for mp_downloading.
    """
    args = []
    length = len(df)

    for index, row in df.reset_index(drop=True).iterrows():
        year = str(row.ID)[4:]
        sat = sat_map.get(year, '17')  # Default to GOES-17 if year missing

        for channel in CHANNELS:
            args.append({
                'index': index,
                'row': row,
                'length': length,
                'channel': channel,
                'sat': sat
            })
    return args


def process_basin(path: str, output_file: str, sat_map: dict):
    """
    Process a basin HURDAT2 file:
      - Load and interpolate.
      - Generate multiprocessing download args.
      - Run parallel download.
      - Save filled DataFrame.

    Parameters
    ----------
    path : str
        HURDAT2 input file path.
    output_file : str
        Output file path for processed track data.
    sat_map : dict
        Satellite assignment by year.
    """
    df = load_and_prepare_df(path)
    df_filled = interpolate_df(df)

    args = generate_args(df_filled, sat_map)
    with Pool(NUM_WORKERS, initializer=init_worker) as pool:
        pool.map(mp_downloading, args)

    df_filled.to_csv(output_file, sep='\t', index=False)


if __name__ == "__main__":
    # Input HURDAT2 paths
    hurdat_paths = {
        'AL': '/rstor/jmayhall/cataloging/hurdat2-2019-2023-AL.txt',
        'EP': '/rstor/jmayhall/cataloging/hurdat2-2019-2023-EP.txt',
    }

    # Satellite selection rules
    sat_selection = {
        'AL': lambda year: '16',  # GOES-16 for Atlantic
        'EP': lambda year: '18' if year == '2023' else '17',  # GOES-18 (2023), else GOES-17
    }

    # Loop through basins
    for basin, path in hurdat_paths.items():
        output_file = f"hurdat_update_{basin}.txt"
        sat_map = {str(y): sat_selection[basin](str(y)) for y in range(2019, 2024)}

        print(f"Processing {basin} basin with satellite mapping: {sat_map}")  # Info Statement
        process_basin(path, output_file, sat_map)
