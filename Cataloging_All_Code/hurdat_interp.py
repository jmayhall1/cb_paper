# coding=utf-8
"""
Last Edited: 10/01/2025
Author: John Mark Mayhall

Description:
------------
Cleans and interpolates latitude/longitude coordinates in HURDAT2-derived
files for AL (Atlantic) and EP (Eastern Pacific) basins.

Workflow:
---------
1. Convert lat/lon strings (e.g., '25N', '80W') to signed floats.
2. Interpolate missing positions within each storm's track.
3. Reformat back to HURDAT2-style coordinates (e.g., '25.0N', '80.0W').
4. Save basin-specific and combined interpolated files.

Notes:
------
- Assumes input files are already synoptic (00, 06, 12, 18 UTC).
- Handles missing values with linear interpolation.
"""

import numpy as np
import pandas as pd


def clean_coord(val: str) -> float:
    """
    Convert a HURDAT-style coordinate string into a signed float.

    Examples:
        '25N' -> 25.0
        '80W' -> -80.0

    Parameters
    ----------
    val : str
        Coordinate string (e.g., '25N', '80W').

    Returns
    -------
    float
        Signed float coordinate. Returns NaN if invalid.
    """
    val = str(val).strip()
    if not val or val[-1] not in ['N', 'S', 'E', 'W']:
        return np.nan
    try:
        num = float(val[:-1])
        if val[-1] in ['S', 'W']:
            num = -num
        return num
    except ValueError:
        return np.nan


def format_coord(val: float, is_lat: bool) -> str:
    """
    Format a signed float into HURDAT-style coordinate string.

    Examples:
        25.3, is_lat=True  -> '25.3N'
        -80.2, is_lat=False -> '80.2W'

    Parameters
    ----------
    val : float
        Signed coordinate.
    is_lat : bool
        Whether the value is latitude (True) or longitude (False).

    Returns
    -------
    str
        Formatted coordinate string with direction (N/S/E/W).
    """
    if pd.isna(val):
        return ""
    direction = 'N' if is_lat else 'E'
    if val < 0:
        direction = 'S' if is_lat else 'W'
    return f"{abs(round(val, 1))}{direction}"


def interpolate_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate latitude/longitude for each storm and reformat to HURDAT style.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'ID', 'Lat', 'Lon' columns.

    Returns
    -------
    pd.DataFrame
        A DataFrame with interpolated 'Lat' and 'Lon' columns.
    """
    # Convert to numeric lat/lon
    df['Lat_num'] = df['Lat'].apply(clean_coord)
    df['Lon_num'] = df['Lon'].apply(clean_coord)

    # Interpolate per storm ID
    for storm_id in df['ID'].unique():
        mask = df['ID'] == storm_id
        df.loc[mask, 'Lat_num'] = df.loc[mask, 'Lat_num'].interpolate(
            method='linear', limit_direction='both'
        )
        df.loc[mask, 'Lon_num'] = df.loc[mask, 'Lon_num'].interpolate(
            method='linear', limit_direction='both'
        )

    # Convert back to HURDAT-style coordinates
    df['Lat'] = df['Lat_num'].apply(lambda x: format_coord(x, is_lat=True))
    df['Lon'] = df['Lon_num'].apply(lambda x: format_coord(x, is_lat=False))

    return df.drop(columns=['Lat_num', 'Lon_num'])


def process_basin(basin: str, input_path: str, output_path: str, names: list) -> pd.DataFrame:
    """
    Load, interpolate, and save HURDAT basin file.

    Parameters
    ----------
    basin : str
        Basin identifier ('AL' or 'EP').
    input_path : str
        Path to input HURDAT file.
    output_path : str
        Path to save interpolated file.
    names : list
        Column names for DataFrame.

    Returns
    -------
    pd.DataFrame
        Interpolated basin DataFrame.
    """
    df = pd.read_csv(input_path, header=0, names=names, sep='\t')
    df_interp = interpolate_latlon(df)
    df_interp.to_csv(output_path, sep='\t', index=False)
    print(f"Interpolated the {basin} basin and saved it to {output_path}")
    return df_interp


if __name__ == "__main__":
    # Common column names
    names = [
        'ID', 'Name', 'Date', 'Time', 'Extra', 'Type', 'Lat', 'Lon', 'Extra2',
        'Pressure', 'Extra3', 'Extra4', 'Extra5', 'Extra6', 'Extra7', 'Extra8',
        'Extra9', 'Extra10', 'Extra11', 'Extra12', 'Extra13', 'Extra14', 'Extra15'
    ]

    # Define input/output paths
    paths = {
        'AL': {
            'in': '//uahdata/rstor/cataloging/hurdat_update_AL.txt',
            'out': 'hurdat_update_interp_AL.txt'
        },
        'EP': {
            'in': '//uahdata/rstor/cataloging/hurdat_update_EP.txt',
            'out': 'hurdat_update_interp_EP.txt'
        },
    }

    # Process both basins
    df_interp_all = []
    for basin, p in paths.items():
        df_interp = process_basin(basin, p['in'], p['out'], names)
        df_interp_all.append(df_interp)

    # Save combined file
    combined = pd.concat(df_interp_all, ignore_index=True)
    combined.to_csv('hurdat_update_interp.txt', sep='\t', index=False)
