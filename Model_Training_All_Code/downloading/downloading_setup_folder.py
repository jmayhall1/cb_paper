# coding=utf-8
"""
@author: John Mark Mayhall
Optimized: 10/02/2025

Purpose:
Defines a Setup class to assist in downloading GOES-16 .nc files.
It provides methods to search values in a dictionary and calculate formatted
timestamps and IDs from files in a given folder.
Paths, datetime formats, and filename slicing may need to be adjusted depending
on the file structure.
"""

import os
from datetime import datetime


class Setup:
    """Class for searching dictionary values and calculating file times from GOES-16 .nc files"""

    def __init__(self, folder: str, values: dict, search_for: str, path: str):
        """
        Initialize Setup instance.

        :param folder: Name of the folder containing files
        :param values: Dictionary of values for searching
        :param search_for: Value to search for in the dictionary
        :param path: Base path to folder
        """
        self.folder = folder
        self.values = values
        self.search_for = search_for
        self.path = path

    def search(self) -> str | None:
        """
        Search the dictionary for a specific value.

        :return: The first matching value if found, otherwise None
        """
        for key, val_list in self.values.items():
            for val in val_list:
                if self.search_for in val:
                    return val
        return None

    def calc_time(self) -> tuple[list[str], list[str], list[str]]:
        """
        Calculate timestamps and storm IDs from files in the folder.

        :return: Tuple of
            - date_func: raw date strings extracted from filenames
            - actual_time_func: formatted datetime strings ("%Y-%m-%d %H:%M:%S")
            - s_id: storm IDs extracted from filenames
        """
        folder_path = os.path.join(self.path, self.folder)
        filenames = sorted(os.listdir(folder_path))

        # Extract date and storm ID based on filename format
        date_func = [fname[6:21] for fname in filenames]  # Adjust indices if filenames differ
        s_id = [fname[35:43] for fname in filenames]  # Adjust indices if filenames differ
        print(f"Storm IDs found: {s_id}")

        # Convert raw date strings to formatted datetime strings
        actual_time_func = [
            datetime.strptime(date_str, "%Y%m%dT%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
            for date_str in date_func
        ]

        return date_func, actual_time_func, s_id
