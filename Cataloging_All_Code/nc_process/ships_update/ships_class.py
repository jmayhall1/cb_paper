# coding=utf-8
"""
Last Edited: 10/02/2025
@author: John Mark Mayhall
Code to read and interpolate SHIPS, Best Track, B-deck, and IMPACT storm files
"""
import datetime
import pickle

import metpy.calc as calc
import numpy as np
import pandas as pd
from geographiclib.geodesic import Geodesic
from metpy.units import units
from tropycal import tracks


def get_archive_report(file_name: str, key: str) -> dict:
    """Retrieve a specific report from a SHIPS pickle file by key."""
    with open(file_name, "rb") as f:
        return pickle.load(f).get(key)


def get_storm_reports(file_name: str, storm_id: str) -> dict:
    """Retrieve all reports for a given ATCF ID from a SHIPS pickle file."""
    reports = {}
    with open(file_name, "rb") as f:
        pkl = pickle.load(f)
        for key, val in pkl.items():
            if storm_id in key:
                reports[key] = val
    return reports


def get_yearly_reports(file_name: str, year: int) -> dict:
    """Retrieve all reports for a given year from a SHIPS pickle file."""
    selected_year = datetime.datetime(year, 1, 1)
    reports = {}
    with open(file_name, "rb") as f:
        pkl = pickle.load(f)
        for key, val in pkl.items():
            atcf_id = key.split("_")[0]
            key_year = datetime.datetime.strptime(atcf_id[-4:], "%Y")
            if key_year == selected_year:
                reports[key] = val
    return reports


class StormInfo:
    """Class to retrieve and interpolate storm information from various sources for a given ATCF ID."""

    def __init__(self, atcfid: str, storminfofile: str, locsource: str, interpolate_interval: int = None):
        """
        Initialize StormInfo with the ATCF ID, data source, and optional interpolation interval.
        :param atcfid: ATCF storm ID.
        :param storminfofile: File path for the storm dataset.
        :param locsource: Source of data: SHIPS_archive, SHIPS_realtime, IMPACT, BEST_TRACK, or b-deck.
        :param interpolate_interval: Optional time interval in minutes for interpolation.
        """
        self.atcfid = atcfid
        self.storminfofile = storminfofile
        self.locsource = locsource
        self.interpolate_interval = interpolate_interval
        self.storm_data_interp = None

        if locsource == "SHIPS_archive":
            self.stormdf = self._interpolate_ships(self._read_ships_archive(), interpolate_interval)
        elif locsource == "SHIPS_realtime":
            self.stormdf = self._interpolate_ships(self._read_ships_realtime(), interpolate_interval)
        elif locsource == "IMPACT":
            self.stormdf = self._interpolate_impact(self._read_impact(), interpolate_interval)
        elif locsource == "BEST_TRACK":
            self.storm_obj = self._read_best_track()
            self.stormdf = self._interpolate_bt(self.storm_obj, interpolate_interval)
        elif locsource == "b-deck":
            bdeck_df = self._read_bdeck()
            self.stormdf = self._interpolate_bdeck(bdeck_df, interpolate_interval) if interpolate_interval else bdeck_df

    # ---------------------- Reading Functions ---------------------- #

    def _read_ships_archive(self) -> pd.DataFrame:
        """Read SHIPS archive pickle file and return as a dataframe with u,v shear components and cardinal direction."""
        shipsdict = self.get_storm_reports(self.storminfofile, self.atcfid)
        df = pd.DataFrame.from_dict(shipsdict, orient='index')

        # Extract SHIPS data
        for ts in df.index:
            df.loc[ts, 'rhhi'] = df.loc[ts].data.get('RHHI', [np.nan])[0]
            df.loc[ts, 'rhmd'] = df.loc[ts].data.get('RHMD', [np.nan])[0]
            df.loc[ts, 'rhlo'] = df.loc[ts].data.get('RHLO', [np.nan])[0]
            df.loc[ts, 'dsst'] = df.loc[ts].data.get('DSST', [np.nan, np.nan, np.nan])[2] / 10
            df.loc[ts, 'dtl'] = df.loc[ts].data.get('DTL', [np.nan, np.nan, np.nan])[2]

        # Convert numeric columns
        numeric_cols = ['max_winds', 'center_lat', 'center_lon', 'mslp', 'rhhi', 'rhmd', 'rhlo', 'dsst']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Convert lon from degrees west to east
        df['center_lon'] *= -1

        # Create timestamp index
        df['Timestamp'] = pd.to_datetime(df['report_date'].astype(str) + df['report_time'].astype(str) + "00",
                                         format='%y%m%d%H%M')
        df.set_index('Timestamp', inplace=True)

        # Convert shear magnitude/direction to u,v components
        shearmag = []
        sheardir = []
        for ts, row in df.iterrows():
            shdc = row.data.get('SHDC', [999.9])[0]
            sddc = row.data.get('SDDC', [9999])[0]
            shearmag.append(np.nan if shdc == 999.9 else shdc / 10.)
            sheardir.append(np.nan if sddc == 9999 else sddc)
        u, v = calc.wind_components(np.array(shearmag) * units('kt'), np.array(sheardir) * units('deg'))
        df['shear_u'], df['shear_v'], df['shear_dir'] = u.magnitude, v.magnitude, sheardir

        # Calculate cardinal direction between consecutive points
        brng_lst = []
        for i in range(len(df)):
            try:
                lat1, lon1 = df.iloc[i][['center_lat', 'center_lon']]
                lat2, lon2 = df.iloc[i + 1][['center_lat', 'center_lon']]
                brng = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)['azi1'] % 360
            except IndexError:
                brng = brng_lst[-1] if brng_lst else 0.0
            brng_lst.append(brng)
        df['cardinal_dir'] = brng_lst

        df.drop(columns=['report_date', 'report_time', 'storm_name', 'data'], inplace=True)
        return df

    def _read_ships_realtime(self) -> pd.DataFrame:
        """Read SHIPS realtime pickle file."""
        df = pd.read_pickle(self.storminfofile)
        df[['max_winds', 'center_lat', 'center_lon']] = df[['max_winds', 'center_lat',
                                                            'center_lon']].apply(pd.to_numeric)
        return df

    def _read_impact(self) -> pd.DataFrame:
        """Read IMPACT CSV file."""
        df = pd.read_csv(self.storminfofile)
        df[['prediction', 'lat', 'lng']] = df[['prediction', 'lat', 'lng']].astype(float)
        df['Timestamp'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
        df.set_index('Timestamp', inplace=True)
        df.rename(columns={'lat': 'center_lat', 'lng': 'center_lon'}, inplace=True)
        return df.iloc[::-1].copy()  # Reverse index

    def _read_best_track(self):
        """Read Best Track data for the Atlantic or East Pacific basins."""
        basin_name = 'north_atlantic' if self.atcfid[:2] == 'AL' else 'east_pacific'
        url_arg = 'atlantic_url' if basin_name == 'north_atlantic' else 'pacific_url'
        ds = tracks.TrackDataset(basin=basin_name, **{url_arg: self.storminfofile})
        storm = ds.get_storm(self.atcfid)
        df = pd.DataFrame({
            'Timestamp': storm.date,
            'center_lat': storm.lat,
            'center_lon': storm.lon,
            'max_winds': storm.vmax,
            'mslp': storm.mslp,
            'rhhi': storm.rhhi
        }).set_index('Timestamp')
        storm.df = df
        return storm

    def _read_bdeck(self) -> pd.DataFrame:
        """Read ATCF b-deck data."""
        cols = ['basin', 'cyclone number', 'YYYYMMDDHH', 'technetium', 'technique', 'forecast period',
                'center_lat', 'center_lon', 'max_winds', 'mslp', 'type', 'windrad', 'quadrant', 'r1', 'r2', 'r3', 'r4',
                'pouter', 'router', 'rmw', 'gusts', 'eye diameter', 'subregion', 'maxseas', 'initials',
                'direction', 'speed', 'stormname', 'depth', 'seas', 'seascode', 'seas1', 'seas2', 'seas3', 'seas4',
                'userdefine1', 'userdata1']
        df = pd.read_csv(self.storminfofile, header=None, usecols=range(37), names=cols)
        df['Timestamp'] = pd.to_datetime(df['YYYYMMDDHH'], format='%Y%m%d%H')
        df.set_index('Timestamp', inplace=True)
        return df

    # ---------------------- Interpolation Functions ---------------------- #

    @staticmethod
    def _interpolate_impact(df: pd.DataFrame, dt: int) -> pd.DataFrame:
        return df.resample(f'{dt}min').asfreq().interpolate()

    @staticmethod
    def _interpolate_ships(df: pd.DataFrame, dt: int) -> pd.DataFrame:
        df_resampled = df.resample(f'{dt}min').asfreq()
        df_resampled['atcf_id'] = df['atcf_id'].iloc[0] if 'atcf_id' in df.columns else None
        return df_resampled.interpolate()

    @staticmethod
    def _interpolate_bt(bt, dt: int) -> pd.DataFrame:
        df_resampled = bt.df.resample(f'{dt}min').asfreq()
        return df_resampled.interpolate()

    @staticmethod
    def _interpolate_bdeck(df: pd.DataFrame, dt: int) -> pd.DataFrame:
        df = df[['max_winds', 'center_lat', 'center_lon', 'mslp']].copy()
        df['center_lat'] = df['center_lat'].apply(lambda x: float("".join(list(str(x))[:-1])) * 0.1)
        df['center_lon'] = df['center_lon'].apply(lambda x: float("".join(list(str(x))[:-1])) * -0.1)
        df = df[~df.index.duplicated()]
        return df.resample(f'{dt}min').asfreq().interpolate().astype(float)

    # ---------------------- Accessor Functions ---------------------- #

    def get_time(self, timestamp: pd.Timestamp) -> pd.Series:
        """Return interpolated storm data at a specific timestamp."""
        return self.storm_data_interp.loc[timestamp]
