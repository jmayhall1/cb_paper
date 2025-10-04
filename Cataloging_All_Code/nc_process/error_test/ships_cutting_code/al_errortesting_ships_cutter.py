# coding=utf-8
"""
Optimized cleanup script for SHIPS storm metadata and GeoJSON consistency checks.
Last optimized: 10/01/2025
@author: John Mark Mayhall
"""
import glob
import os

import pandas as pd

# ----------------------------------------------------------------------
# Load SHIPS Data
# ----------------------------------------------------------------------
ships_df = pd.read_csv(
    '//uahdata/rstor/cataloging/nc_process/shear_process/ships_interp_AL.txt',
    sep='\t',
    index_col=0
)

# ----------------------------------------------------------------------
# Filter SHIPS rows by year, location, and valid synoptic hours
# ----------------------------------------------------------------------
valid_hours = {0, 3, 6, 9, 12, 15, 18, 21}
rows, id_list = [], []

for timestamp, row in ships_df.iterrows():
    ts = pd.Timestamp(timestamp)
    if (
            ts.year == 2019
            and 5 < row.center_lat < 30
            and -105 < row.center_lon < -20
            and ts.hour in valid_hours
    ):
        rows.append(timestamp)
        id_list.append(row.atcf_id)

# Construct filtered SHIPS DataFrame
ships_df = pd.DataFrame(
    {"Index": rows, "atcf_id": id_list},
    index=rows
).sort_values(by=['atcf_id', 'Index'])


# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------
def clean_geojson_files(file_list, ships_df, id_slice, time_slice, base_path=""):
    """
    Verify and clean GeoJSON files against SHIPS data.

    Parameters
    ----------
    file_list : list[str]
        List of file paths to check.
    ships_df : pd.DataFrame
        Filtered SHIPS DataFrame with valid IDs/times.
    id_slice : tuple[int, int]
        Slice (start, end) for extracting ATCF ID from filename.
    time_slice : tuple[int, int]
        Slice (start, end) for extracting timestamp from filename.
    base_path : str
        Optional base directory for constructing full file paths.
    """
    remaining = ships_df.copy()

    for file in file_list:
        if "AL" not in file:
            continue

        atcf_id = file[id_slice[0]:id_slice[1]]
        file_time = pd.Timestamp(file[time_slice[0]:time_slice[1]])

        # Check if ID and timestamp exist in SHIPS dataset
        match = (
                (remaining['atcf_id'] == atcf_id) &
                (remaining['Index'] == str(file_time))
        ).any()

        if not match:
            os.remove(file if base_path == "" else os.path.join(base_path, file))
            print(file)

        # Drop this entry from SHIPS dataframe (avoid duplicate checking)
        remaining = remaining.loc[
            (remaining['atcf_id'] != atcf_id) | (remaining['Index'] != str(file_time))
            ]

    return remaining


# ----------------------------------------------------------------------
# Remove unmatched GeoJSON files
# ----------------------------------------------------------------------
done1 = glob.glob('C:/Users/jmayhall/Documents/Model_Training_Full/*.geojson')
done2 = glob.glob('//uahdata/rstor/cataloging/nc_process/error_test/*.geojson')

total_count = 0
total_count += len(done1) + len(done2)

ships_df = clean_geojson_files(done1, ships_df, (83, 91), (54, 69))
ships_df = clean_geojson_files(done2, ships_df, (84, 92), (55, 70))

# ----------------------------------------------------------------------
# Remove known bad files
# ----------------------------------------------------------------------
bad_files = [
    'SPoRT_20190901T120000_labels_WGS84_AL052019.geojson',
    'SPoRT_20190925T030000_labels_WGS84_AL122019.geojson',
    'SPoRT_20190925T030000_labels_WGS84_AL132019.geojson',
    'SPoRT_20191015T060000_labels_WGS84_AL152019.geojson',
    'SPoRT_20191015T090000_labels_WGS84_AL152019.geojson',
    'SPoRT_20191015T180000_labels_WGS84_AL152019.geojson',
    'SPoRT_20190923T120000_labels_WGS84_AL132019.geojson',
    'SPoRT_20190924T000000_labels_WGS84_AL132019.geojson',
    'SPoRT_20191015T030000_labels_WGS84_AL152019.geojson',
    'SPoRT_20191015T000000_labels_WGS84_AL152019.geojson',
    'SPoRT_20191015T120000_labels_WGS84_AL152019.geojson',
    'SPoRT_20191015T210000_labels_WGS84_AL152019.geojson',
    'SPoRT_20191016T000000_labels_WGS84_AL152019.geojson',
    'SPoRT_20190923T150000_labels_WGS84_AL132019.geojson',
    'SPoRT_20191014T210000_labels_WGS84_AL152019.geojson',
    'SPoRT_20191015T150000_labels_WGS84_AL152019.geojson',
    'SPoRT_20190923T210000_labels_WGS84_AL132019.geojson',
    'SPoRT_20190923T180000_labels_WGS84_AL132019.geojson',
    'SPoRT_20190923T030000_labels_WGS84_AL132019.geojson',
    'SPoRT_20190923T000000_labels_WGS84_AL132019.geojson',
    'SPoRT_20190923T060000_labels_WGS84_AL132019.geojson',
    'SPoRT_20190923T090000_labels_WGS84_AL132019.geojson',
]

for file in bad_files:
    atcf_id = file[-16:-8]
    file_time = pd.Timestamp(file[-45:-30])

    match = (
            (ships_df['atcf_id'] == atcf_id) &
            (ships_df['Index'] == str(file_time))
    ).any()

    if not match:
        full_path = f'//uahdata/rstor/cataloging/nc_process/error_test/{file}'
        os.remove(full_path)
        print(full_path)

    ships_df = ships_df.loc[
        (ships_df['atcf_id'] != atcf_id) | (ships_df['Index'] != str(file_time))
        ]

# ----------------------------------------------------------------------
# Final Output
# ----------------------------------------------------------------------
print("Remaining SHIPS records:", ships_df.shape[0])
print("Total GeoJSON files checked:", total_count)

ships_df.to_csv('error_ships_AL.txt', sep='\t')
