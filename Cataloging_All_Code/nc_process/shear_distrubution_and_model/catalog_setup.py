# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall

This module defines the Setup class, which runs a trained CNN model
on preprocessed tropical cyclone datasets (C8, C13 scaled/unscaled, lat/lon).
It handles input file lookup, multiprocessing over C13 files,
and orchestrates calls to the plotting routine.
"""

import glob
import os
from multiprocessing import Pool

from catalog_mp import mp_plotting


class Setup:
    """
    Handles preprocessing and execution of a trained CNN model
    on satellite tropical cyclone imagery.
    """

    def __init__(self,
                 model_path: str,
                 latlon_path: str,
                 cutoff: float,
                 c8_path: str,
                 c13_scaled_path: str,
                 c13_unscaled_path: str,
                 dataframe_path: str) -> None:
        """
        Initialize model setup parameters.

        :param model_path: Path to the trained CNN model.
        :param latlon_path: Path pattern for .npz lat/lon arrays.
        :param cutoff: Minimum probability cutoff for plotting.
        :param c8_path: Path pattern for scaled C08 channel arrays.
        :param c13_scaled_path: Path pattern for scaled C13 channel arrays.
        :param c13_unscaled_path: Path pattern for unscaled C13 channel arrays.
        :param dataframe_path: Path to interpolated HURDAT dataframe.
        """
        self.model_path = model_path
        self.latlon_path = latlon_path
        self.cutoff = cutoff
        self.c8_path = c8_path
        self.c13_scaled_path = c13_scaled_path
        self.c13_unscaled_path = c13_unscaled_path
        self.dataframe_path = dataframe_path

    # -----------------------------
    # Helper methods
    # -----------------------------
    @staticmethod
    def _load_file_dict(path_pattern: str) -> dict:
        """
        Convert a path glob into a dictionary {filename: full_path}.
        This allows O(1) lookup by filename later.

        :param path_pattern: Glob pattern for files.
        :return: Dictionary mapping filename → full file path.
        """
        return {os.path.basename(f): f for f in glob.glob(path_pattern)}

    # -----------------------------
    # Main execution
    # -----------------------------
    def main(self, num_workers: int = 1) -> None:
        """
        Run the trained model on test images using multiprocessing.

        :param num_workers: Number of worker processes (default=1).
        """

        # Preload file dictionaries for quick lookups
        c8_files = self._load_file_dict(self.c8_path)
        c13_unscaled_files = self._load_file_dict(self.c13_unscaled_path)
        latlon_files = self._load_file_dict(self.latlon_path)

        # C13 scaled files kept as a list since iteration is required
        c13_scaled_files = glob.glob(self.c13_scaled_path)

        # Define probability ticks for thresholding
        prob_ticks = [0.02, 0.2, 0.4, 0.6, 0.8, 1.0]

        # Build argument dictionaries for each job
        needed_args = [
            {
                'num': num,
                'file': file,
                'length': len(c13_scaled_files),
                'model_path': self.model_path,
                'c8_files': c8_files,
                'c13_unscaled_files': c13_unscaled_files,
                'latlon_files': latlon_files,
                'prob_ticks': prob_ticks,
                'cutoff': self.cutoff,
            }
            for num, file in enumerate(c13_scaled_files)
        ]

        # Run multiprocessing pool
        with Pool(num_workers) as pool:
            pool.map(mp_plotting, needed_args)
