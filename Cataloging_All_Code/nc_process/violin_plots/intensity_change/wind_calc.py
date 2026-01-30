# coding=utf-8
"""
Last Edited: 07/10/2025
@author: John Mark Mayhall
"""
import numpy as np
import pandas as pd


class Wind:
    """
    Class for calculating wind speed changes across time intervals
    for a given tropical cyclone based on ATCF ID and timestamp.
    """

    def __init__(self, df_storm_id: str, real_time: pd.Timestamp, wind_dict: pd.DataFrame):
        """
        Initialize Wind instance.

        :param df_storm_id: Storm ID (e.g., AL012019).
        :param real_time: Timestamp (datetime or string format).
        :param wind_dict: DataFrame containing wind speed information.
        """
        self.df_storm_id = df_storm_id
        self.real_time = pd.to_datetime(real_time)
        self.wind_dict = wind_dict
        self.real_time_str = self.real_time.strftime('%Y-%m-%d %H:%M:%S')

    def get_wind_at_time(self, timestamp: pd.Timestamp) -> float:
        """
        Retrieves the max wind at a given time for the storm.

        :param timestamp: The timestamp to query.
        :return: Wind speed at the given time or None.
        """
        try:
            df = self.wind_dict[
                (self.wind_dict.index == str(timestamp)) &
                (self.wind_dict.atcf_id == self.df_storm_id)
                ]
            return df.max_winds.values[0]
        except (IndexError, AttributeError) as e:
            print(f'Wind Diff error: {e}')
            return None

    def get_wind_change(self, time_diff: pd.Timedelta, time: int) -> float:
        """
        Calculate wind change from the real time to a time offset.

        :param time: Time of displacement
        :param time_diff: Time offset (positive/negative).
        :return: Wind speed difference or None if unavailable.
        """
        base_wind = self.get_wind_at_time(self.real_time)
        target_wind = self.get_wind_at_time(self.real_time + time_diff)
        if base_wind is None or target_wind is None:
            return None
        wind_diff = None
        if time < 0:
            wind_diff = base_wind - target_wind
        elif time > 0:
            wind_diff = target_wind - base_wind
        return wind_diff

    def wind_change_calc(self) -> list:
        """
        Compute wind speed change over 8 time intervals centered on `real_time`.

        :return: List of wind differences at intervals [-24, -18, -12, -6, 6, 12, 18, 24] hours.
        """
        time_offsets = [pd.Timedelta(hours=h) for h in [-24, -18, -12, -6, 6, 12, 18, 24]]
        return [self.get_wind_change(td, time) for td, time in zip(time_offsets, [-24, -18, -12, -6, 6, 12, 18, 24])]