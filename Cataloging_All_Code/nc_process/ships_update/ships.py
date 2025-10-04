# coding=utf-8
"""
Last Edited: 10/02/2025
@author: John Mark Mayhall
Code to update SHIPS data and interpolate for ATCF IDs.
"""
from pathlib import Path

import pandas as pd
from pickle_class import Pickler
from ships_class import StormInfo


def process_ships_region(hurdat_file: str, txt_file: str, pkl_file: str, interp_interval: int = 60) -> pd.DataFrame:
    """
    Process a SHIPS region: create pickle file, read ATCF IDs, interpolate storm data.

    :param hurdat_file: Path to the HURDAT update text file
    :param txt_file: Path to the SHIPS dat file
    :param pkl_file: Path to the SHIPS output pickle file
    :param interp_interval: Interpolation interval in minutes
    :return: Interpolated storm dataframe
    """
    hurdat_df = pd.read_csv(
        hurdat_file,
        header=None,
        sep='\t',
        names=[
            'ID', 'Name', 'Date', 'Time', 'Extra', 'Type', 'Lat', 'Lon', 'Extra2', 'Pressure',
            'Extra3', 'Extra4', 'Extra5', 'Extra6', 'Extra7', 'Extra8', 'Extra9', 'Extra10',
            'Extra11', 'Extra12', 'Extra13', 'Extra14', 'Extra15'
        ]
    )

    atcf_set = set(hurdat_df['ID']) - {'ID'}

    # Create SHIPS pickle file
    Pickler(txt_file, pkl_file).stock_pickle()

    # Initialize empty dataframe
    columns = [
        'max_winds', 'center_lat', 'center_lon', 'mslp', 'atcf_id', 'shear_u', 'shear_v',
        'shear_dir', 'cardinal_dir', 'rhhi', 'rhmd', 'rhlo', 'dsst', 'dtl'
    ]
    region_df = pd.DataFrame(columns=columns)

    # Iterate over ATCF IDs and collect interpolated storm data
    for storm_id in atcf_set:
        interface = StormInfo(storm_id, pkl_file, 'SHIPS_archive', interp_interval)
        region_df = pd.concat([region_df, interface.stormdf])

    # Sort dataframe
    region_df = region_df.sort_index(kind='mergesort').sort_values('atcf_id', kind='mergesort')
    return region_df


if __name__ == "__main__":
    indir = Path("//uahdata/rstor/cataloging/nc_process/ships_update/")

    # Process Atlantic region
    atlantic_df = process_ships_region(
        hurdat_file=str(indir / "hurdat_update_AL.txt"),
        txt_file=str(indir / "lsdiaga_2019_2023_sat_ts_7day.txt"),
        pkl_file=str(indir / "lsdiaga_2019_2023_sat_ts_7day.pkl"))
    atlantic_df.to_csv('ships_interp_AL.txt', sep='\t')

    # Process East Pacific region
    epacific_df = process_ships_region(
        hurdat_file=str(indir / "hurdat_update_EP.txt"),
        txt_file=str(indir / "lsdiage_2019_2023_sat_ts_7day.txt"),
        pkl_file=str(indir / "lsdiage_2019_2023_sat_ts_7day.pkl"))
    epacific_df.to_csv('ships_interp_EP.txt', sep='\t')

    # Combine both regions
    complete_df = pd.concat([atlantic_df, epacific_df])
    complete_df.to_csv('ships_interp.txt', sep='\t')

