# coding=utf-8
"""
Last Edited: 07/10/2025
@author: John Mark Mayhall
Purpose: Identify transverse bands in TC quadrants based on shear vector.
"""
import glob
import pandas as pd


# ======================== Config ========================
online, save_state = False, False
base_dir = '/rstor/jmayhall/' if online else '//uahdata/rstor/'

paths = {
    'latlon': f'{base_dir}cataloging/nc_process/geojson_transform/completed_arrays/latlon_arrs/*.npz',
    'hurdat': f'{base_dir}cataloging/hurdat_update.txt',
    'model': f'{base_dir}Model_Training_Code/cnn_creation',
    'c8': f'{base_dir}cataloging/nc_process/geojson_transform/completed_arrays/C08_scaled/*',
    'c13': f'{base_dir}cataloging/nc_process/geojson_transform/completed_arrays/C13_scaled/*',
    'ships': 'C:/Users/jmayhall/PycharmProjects/PythonProject/ships_interp.txt'
}

# ===================== Load Data ========================
hurdat_df = pd.read_csv(paths['hurdat'], sep='\t')
latlon_arrays = pd.DataFrame({'Name': glob.glob(paths['latlon'])})
c8_scaled = pd.DataFrame({'Name': glob.glob(paths['c8'])})
c13_scaled = glob.glob(paths['c13'])
path_len = len(paths['c13']) - 1
wind_dict = pd.read_csv(paths['ships'], sep='\t', index_col=0)
count, land_count_100, land_count_250, land_count_500 = 0, 0, 0, 0
al_count, ep_count, empty_count = 0, 0, 0

for file in c13_scaled:
    identifier = file[path_len: path_len + 22]
    df_storm_id = identifier[:8]
    date = identifier[9:17]
    time = identifier[18:22]
    real_time = pd.to_datetime(f'{date}{time}', format='%Y%m%d%H%M')

    storm = wind_dict.query("index == @real_time.strftime('%Y-%m-%d %H:%M:%S') and atcf_id == @df_storm_id")
    if storm.empty:
        empty_count += 1

    else:
        dist_land = storm.iloc[0].dtl
        if dist_land < 100:
            land_count_100 += 1
        if dist_land < 250:
            land_count_250 += 1
        if dist_land < 500:
            land_count_500 += 1
        if 'AL' in df_storm_id:
            al_count += 1
        if 'EP' in df_storm_id:
            ep_count += 1
        count += 1
print(land_count_100)
print(count)
print(land_count_100 / count)

print(land_count_250)
print(count)
print(land_count_250 / count)

print(land_count_500)
print(count)
print(land_count_500 / count)

print(al_count)
print(ep_count)