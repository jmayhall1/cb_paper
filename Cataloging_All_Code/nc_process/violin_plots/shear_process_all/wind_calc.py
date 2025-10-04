# coding=utf-8
"""
Last Edited: 10/02/2025
@author: John Mark Mayhall
"""
import pandas as pd


class Wind:
    """
    Class for calculating wind speed changes over time for a specific tropical cyclone.
    """

    def __init__(self, df_storm_id: str, real_time: pd.Timestamp, wind_df: pd.DataFrame):
        """
        Initialize Wind instance.

        :param df_storm_id: Storm ID (e.g., 'AL012019').
        :param real_time: Reference timestamp (datetime or string).
        :param wind_df: DataFrame with columns ['atcf_id', 'max_winds'] and a datetime index.
        """
        self.df_storm_id = df_storm_id
        self.real_time = pd.to_datetime(real_time)

        # Filter only the relevant storm data once to avoid repeated filtering
        self.storm_wind_df = wind_df[wind_df.atcf_id == self.df_storm_id]
        # Ensure the index is datetime for proper timedelta calculations
        self.storm_wind_df.index = pd.to_datetime(self.storm_wind_df.index)

    def get_wind_at_time(self, timestamp: pd.Timestamp):
        """
        Retrieve the maximum wind for the storm at a given timestamp.

        :param timestamp: Timestamp to query.
        :return: Wind speed or None if not found.
        """
        try:
            return self.storm_wind_df.loc[timestamp, 'max_winds']
        except KeyError:
            # Timestamp not found for this storm
            return None

    def get_wind_change(self, time_diff: pd.Timedelta, time_offset: int):
        """
        Calculate the wind speed difference over a time interval.

        :param time_diff: Time offset from the reference time.
        :param time_offset: Signed interval in hours (negative for past, positive for future)
        :return: Wind speed difference or None if unavailable.
        """
        base_wind = self.get_wind_at_time(self.real_time)
        target_wind = self.get_wind_at_time(self.real_time + time_diff)

        if base_wind is None or target_wind is None:
            return None

        # Use simple arithmetic based on time sign
        return target_wind - base_wind if time_offset > 0 else base_wind - target_wind

    def wind_change_calc(self) -> list[float | None]:
        """
        Compute wind speed change at fixed intervals around the reference time.

        :return: List of wind differences at intervals [-24, -18, -12, -6, 6, 12, 18, 24] hours.
        """
        intervals = [-24, -18, -12, -6, 6, 12, 18, 24]
        time_offsets = [pd.Timedelta(hours=h) for h in intervals]

        # Compute wind changes efficiently using list comprehension
        return [self.get_wind_change(td, h) for td, h in zip(time_offsets, intervals)]
