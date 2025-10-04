# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall

Class for calculating wind changes and plotting TCB occurrences relative to storm intensity change.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Wind:
    """
    Calculates wind changes over multiple time intervals for a given storm
    and generates plots of TCB occurrences relative to intensity change.
    """

    def __init__(self, df_storm_id: str, real_time: pd.Timestamp,
                 month_dict: dict, wind_dict: pd.DataFrame) -> None:
        """
        Initialize the Wind class.
        :param df_storm_id: Storm identifier.
        :param real_time: Timestamp for current observation.
        :param month_dict: Optional month mapping (not used directly here).
        :param wind_dict: DataFrame of max winds with storm IDs as a column and timestamps as index.
        """
        self.df_storm_id = df_storm_id
        self.real_time = real_time
        self.month_dict = month_dict
        self.wind_dict = wind_dict

    def get_wind_change(self, time_diff: pd.Timedelta):
        """
        Calculate wind difference between current time and a shifted time.
        :param time_diff: Time difference relative to current observation.
        :return: Wind change in knots, or None if data is missing.
        """
        try:
            target_time = self.real_time + time_diff

            # Efficient vectorized lookup using mask
            mask_current = (self.wind_dict.index == str(self.real_time)) & \
                           (self.wind_dict.atcf_id == self.df_storm_id)
            mask_target = (self.wind_dict.index == str(target_time)) & \
                          (self.wind_dict.atcf_id == self.df_storm_id)

            current_wind = self.wind_dict.loc[mask_current, 'max_winds'].values[0]
            target_wind = self.wind_dict.loc[mask_target, 'max_winds'].values[0]
            final_wind = current_wind - target_wind

            return final_wind
        except (IndexError, AttributeError):
            # Missing data
            return None

    def wind_change_calc(self) -> list:
        """
        Calculate wind changes for standard intervals relative to current time.
        :return: List of wind changes for [-24, -18, -12, -6, +6, +12, +18, +24] hours.
        """
        intervals = [pd.Timedelta(hours=h) for h in [-24, -18, -12, -6, 6, 12, 18, 24]]
        return [self.get_wind_change(diff) for diff in intervals]

    @staticmethod
    def plot(wind_change_df: pd.DataFrame) -> None:
        """
        Generate bar plots of TCB occurrences vs intensity change.
        :param wind_change_df: DataFrame with wind change categories and counts.
        """
        # Define plot metadata for each interval
        time_intervals = {
            0: ['Previous 24 Hours', 'm24'],
            1: ['Previous 18 Hours', 'm18'],
            2: ['Previous 12 Hours', 'm12'],
            3: ['Previous 6 Hours', 'm6'],
            4: ['Next 6 Hours', '6'],
            5: ['Next 12 Hours', '12'],
            6: ['Next 18 Hours', '18'],
            7: ['Next 24 Hours', '24']
        }

        categories = ['<=-10', '-5', 'Constant', '5', '>=10']
        count_columns = [f'{cat} Count' for cat in categories]

        for i, (label, suffix) in time_intervals.items():
            # Extract values and counts
            values = wind_change_df.loc[i, categories].to_numpy(dtype=np.float32)
            counts = wind_change_df.loc[i, count_columns].to_numpy(dtype=np.float32)

            # Avoid division by zero
            percentages = np.where(counts > 0, values / (counts * 1024 * 1024) * 100, 0)

            # Plot
            plt.figure(figsize=(10, 8))
            x_labels = ['Decrease ~10 kts or Less', 'Decrease ~5 kts', 'Constant', 'Increase ~5 kts',
                        'Increase ~10 kts or Greater']
            bars = plt.bar(x_labels, percentages, color='skyblue', edgecolor='black')

            # Annotate sample sizes above bars
            for bar, pct, count in zip(bars, percentages, counts):
                plt.text(bar.get_x() + bar.get_width() / 2, pct + 0.5, f'{int(count)}',
                         ha='center', va='bottom', fontsize=9)

            plt.xticks(rotation=30, ha='center', fontsize=9)
            plt.title(f'TCB Occurrences vs TC Intensity Change over the {label}\n'
                      f'for 2019-2023 Atlantic and Eastern Pacific TCs')
            plt.xlabel('Intensity Change')
            plt.ylabel('Percentage of Total Storm Image Pixels with TCBs', fontsize=10)
            plt.ylim(0, 25)
            plt.tight_layout()
            plt.savefig(f'strength_change_{suffix}.png', dpi=300)
            plt.close()
