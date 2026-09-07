# coding=utf-8
"""
Optimized: 10/01/2025
@author: John Mark Mayhall

Purpose:
--------
Map tropical cyclones (TCs) occurring over ~25°C SST currently, in the past 24 hours, or in the next 24 hours.
TD, TS, CAT1-2, CAT3-5 categories are plotted.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_ships_data(file_path: Path) -> pd.DataFrame:
    """
    Load SHIPS data with datetime index for fast lookup.
    """
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    df.index = pd.to_datetime(df.index)
    df.atcf_id = df.atcf_id.astype(str)
    return df

if __name__ == '__main__':
    # Paths to SHIPS data
    al_file = Path('//uahdata/rstor/cataloging/nc_process/shear_process/shear_process_all/ships_interp_AL.txt')
    al_df = load_ships_data(al_file)
    id_list = []
    for row, column in al_df.iterrows():
        if (-105 < column.center_lon < -20) and (5 < column.center_lat < 30) and (column.rhmd < 35):
            id_list.append(column.atcf_id)
    print(len(id_list))
    print(set(id_list))
