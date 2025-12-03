# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall

This module defines the Setup class, which runs a trained CNN model
on preprocessed tropical cyclone datasets (C8, C13 scaled/unscaled, lat/lon).
It handles input file lookup, multiprocessing over C13 files,
and orchestrates calls to the plotting routine.
"""

import gc
import glob
import multiprocessing as mp
import os

from catalog_mp import mp_plotting, init_worker
from keras.models import load_model


def run_multiprocessing(needed_args, num_workers: int = 64, tasks_per_worker: int = 50):
    """
    Run tasks in a memory-safe way with spawn + maxtasksperchild.

    :param needed_args: list of argument dicts for mp_plotting
    :param num_workers: number of parallel processes
    :param tasks_per_worker: restart each worker after this many tasks
    """
    # Use spawn to avoid fork memory issues (TensorFlow/NumPy)
    mp.set_start_method("spawn", force=True)

    # Explicitly create the pool
    pool = mp.Pool(
        processes=num_workers,
        initializer=init_worker,
        maxtasksperchild=tasks_per_worker
    )

    try:
        # Run tasks
        pool.map(mp_plotting, needed_args)
    finally:
        # Ensure proper cleanup
        pool.close()   # no more tasks
        pool.join()    # wait for all workers to exit

    # Optional: force garbage collection in the main process
    gc.collect()


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
                 dataframe_path: str,
                 plot: bool) -> None:
        """
        Initialize model setup parameters.

        :param model_path: Path to the trained CNN model.
        :param latlon_path: Path pattern for .npz lat/lon arrays.
        :param cutoff: Minimum probability cutoff for plotting.
        :param c8_path: Path pattern for scaled C08 channel arrays.
        :param c13_scaled_path: Path pattern for scaled C13 channel arrays.
        :param c13_unscaled_path: Path pattern for unscaled C13 channel arrays.
        :param dataframe_path: Path to interpolated HURDAT dataframe.
        :param plot: Bool to say whether to plot or not.
        """
        self.model_path = model_path
        self.latlon_path = latlon_path
        self.cutoff = cutoff
        self.c8_path = c8_path
        self.c13_scaled_path = c13_scaled_path
        self.c13_unscaled_path = c13_unscaled_path
        self.dataframe_path = dataframe_path
        self.plot = plot

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
    def main(self, num_workers: int = 128) -> None:
        """
        Run the trained model on test images using multiprocessing.

        :param num_workers: Number of worker processes (default=1).
        """
        temp_files = glob.glob(self.c13_scaled_path)
        c13_scaled_files = []
        for file in temp_files:
            if not '2019' in file and not '2020' in file:
                c13_scaled_files.append(file)
        done_files = os.listdir('/rstor/jmayhall/cataloging/nc_process/shear_distrubution_and_model/tcb_probs/')
        for file in done_files:
            s_id, s_date, s_time = file[:8], file[9: 17], file[18:22]
            check_file = f'{os.path.dirname(c13_scaled_files[0])}/{s_id}_{s_date}_{s_time}_C13_scaled_cut.npz'
            c13_scaled_files.remove(check_file)

        # Define probability ticks for thresholding
        prob_ticks = [0.02, 0.2, 0.4, 0.6, 0.8, 1.0]

        # Build argument dictionaries for each job
        needed_args = [
            {
                'num': num,
                'file': file,
                'length': len(c13_scaled_files),
                'prob_ticks': prob_ticks,
                'cutoff': self.cutoff,
                'plot': self.plot
            }
            for num, file in enumerate(c13_scaled_files)
        ]

        mp.set_start_method("spawn", force=True)

        # VERY IMPORTANT:
        # maxtasksperchild forces each worker process to restart after
        # a few tasks so memory never accumulates.
        # Use 10–50 depending on how heavy each task is.

        run_multiprocessing(needed_args, num_workers=num_workers, tasks_per_worker=50)
