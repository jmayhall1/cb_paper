# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall
Memory-efficient and optimized code for plotting TCB data by shear-relative azimuth and radius.
"""
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.interpolate
from matplotlib.cm import ScalarMappable


class Plotting:
    """
    Class for plotting heatmaps of TCB (Tropical Cyclone Banding) shear data
    relative to azimuth or storm motion.
    """

    def __init__(self, data_group: list[pd.DataFrame], plot_type: str, title_dict: dict, rows: int, columns: int):
        """
        Initializes the plotting class.

        :param data_group: List of DataFrames containing 'Radius', 'Frequency', and azimuth/shear info.
        :param plot_type: Column name representing angular data ('azimuth', 'degree', or 'stormrel').
        :param title_dict: Dictionary mapping subplot indices to titles.
        :param rows: Number of subplot rows.
        :param columns: Number of subplot columns.
        """
        self.data_group = data_group
        self.plot_type = plot_type
        self.title_dict = title_dict
        self.rows = rows
        self.columns = columns

    def main_plot(self) -> None:
        """
        Generates TCB heatmaps for all subplots in a memory-efficient manner.
        """
        print(f'Plotting {self.plot_type.capitalize()} Data for 1024 km domain')

        # Initialize figure and axes with polar subplots
        fig, axes = plt.subplots(
            nrows=self.rows,
            ncols=self.columns,
            subplot_kw={'polar': True},
            figsize=(15, 10)
        )
        axes = axes.flatten()

        # Shared interpolation grid
        n_grid = 150  # Reduces memory usage while maintaining reasonable resolution
        theta_grid = np.linspace(0, 2 * np.pi, n_grid, dtype=np.float32)
        radius_grid = np.linspace(0, 1024, n_grid, dtype=np.float32)
        theta_mesh, radius_mesh = np.meshgrid(theta_grid, radius_grid)

        # Contour levels
        level_ticks = np.linspace(0, 60, 7, dtype=np.float32)  # 0, 10, ..., 60
        levels = np.linspace(level_ticks.min(), level_ticks.max(), 500, dtype=np.float32)
        tick_labels = [str(int(tick)) for tick in level_ticks[:-1]] + [f'≥ {int(level_ticks[-1])}']

        # Loop through each subplot
        for i, ax in enumerate(axes):
            data = self.data_group[i]
            data = data[data['Radius'] <= 1024]

            # Convert to float32 for memory efficiency
            theta = (data[self.plot_type].values * np.pi / 180).astype(np.float32)
            radius = data['Radius'].values.astype(np.float32)
            count = data['Frequency'].values.astype(np.float32)

            # Grid interpolation using cubic method
            count_grid = scipy.interpolate.griddata(
                (theta, radius),
                count,
                (theta_mesh, radius_mesh),
                method='cubic',
                rescale=True
            )

            # Replace NaNs and clip values to [0, max_tick]
            nan_val = 0
            count_grid = np.nan_to_num(count_grid, nan=nan_val)
            np.clip(count_grid, 0, level_ticks.max(), out=count_grid)

            # Plot heatmap on polar subplot
            ax.contourf(theta_mesh, radius_mesh, count_grid, levels, cmap='turbo',
                        vmin=level_ticks.min(), vmax=level_ticks.max())

            # Set polar plot orientation and styling
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.tick_params(axis='y', labelsize=14, colors='black')

            # Apply white outline to tick labels for better visibility
            outline_effect = [pe.withStroke(linewidth=3, foreground='white')]
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_path_effects(outline_effect)

            # Set subplot title
            ax.set_title(self.title_dict.get(i, f'Plot {i}'), fontsize=16)

        # Shared colorbar setup
        norm = mpl.colors.Normalize(vmin=level_ticks.min(), vmax=level_ticks.max())
        sm = ScalarMappable(cmap='turbo', norm=norm)
        sm.set_array([])

        fig.subplots_adjust(bottom=0.2, hspace=0.4)  # Space for colorbar
        cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=level_ticks,
                            format=mticker.FixedFormatter(tick_labels))
        cbar.set_label('TCB Occurrence (%)', fontsize=20)
        cbar.ax.tick_params(labelsize=18)

        # Determine plot title and save filename
        title_map = {'stormrel': 'Storm Motion', 'azimuth': 'Cardinal Direction', 'degree': 'Shear Direction'}
        first_title = self.title_dict.get(0, '')
        sep_type = 'intensity' if 'TD' in first_title or 'CAT' in first_title else \
            'shear' if 'Shear' in first_title else \
                'time' if 'LST' in first_title else 'unknown'
        plot_title = title_map.get(self.plot_type, 'Unknown')
        save_title = f'{self.plot_type}_{sep_type}'

        # Final figure title and save
        fig.suptitle(f'TCB {plot_title} Relative Plots', fontsize=20)
        plt.savefig(f'{save_title}_heatmap_1024.jpeg', dpi=600)
        plt.close(fig)
