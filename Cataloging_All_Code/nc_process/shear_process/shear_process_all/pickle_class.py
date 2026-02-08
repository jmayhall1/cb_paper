# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall
@author: Rob Junod

Converts Statistical Hurricane Intensity Prediction Scheme (SHIPS) archive .dat files into a structured pickle file
for faster processing.
"""
import pickle
import re
from typing import Tuple, List


class Pickler:
    """
    Class to parse SHIPS archive .dat files and save as a pickle dictionary.
    Each storm is stored with a unique key and associated metadata.
    """

    def __init__(self, file_name: str, output_file_name: str) -> None:
        """
        Initialize the pickler.
        :param file_name: Input SHIPS archive file.
        :param output_file_name: Output pickle file path.
        """
        self.file_name = file_name
        self.output_file = output_file_name
        self.time_column_length = 31 if "7_day" in self.file_name else 23
        self.pickle_dict = {}
        self.file_lines = self._load_file()

    def _load_file(self) -> List[str]:
        """
        Read all lines from the dat file into memory.
        :return: List of lines.
        """
        with open(self.file_name) as f:
            return f.readlines()

    @staticmethod
    def _find_data_label(line: List[str]) -> Tuple[int, str]:
        """
        Find the first alphabetic label in a line, skipping the first element.
        :param line: Split line from dat file.
        :return: Tuple of (index, label string).
        """
        for idx, element in enumerate(line):
            if idx != 0 and re.match("[a-zA-Z]", element):
                return idx, element
        raise ValueError("No data label found in line.")

    def _fill_missing_data(self, line: List[int]) -> List[int]:
        """
        Prepend 9999 to lines shorter than expected time column length.
        :param line: List of integer data.
        :return: Padded list of length `time_column_length`.
        """
        while len(line) < self.time_column_length:
            line.insert(0, 9999)
        return line

    @staticmethod
    def _make_key(head_line: List[str]) -> str:
        """
        Generate unique key for the storm from header information.
        :param head_line: List containing header line info.
        :return: Unique key string.
        """
        return "_".join([head_line[-2], head_line[7], head_line[1][2:], head_line[2] + '00'])

    def _convert_to_dict(self, indices: Tuple[int, int]) -> None:
        """
        Convert a storm block from file_lines into structured dict for pickling.
        :param indices: (start_idx, end_idx) of the storm block.
        """
        data_block = self.file_lines[indices[0]: indices[1]]
        data_dict = {}
        head_info = None

        for line in data_block:
            line_parts = line.rstrip("\n").split()

            if "HEAD" not in line_parts and "TIME" not in line_parts:
                # Standard data line
                label_index, label = self._find_data_label(line_parts)
                data_values = list(map(int, line_parts[:label_index]))
                if label_index < self.time_column_length:
                    data_values = self._fill_missing_data(data_values)
                data_dict[label] = data_values

            elif "TIME" in line_parts:
                # Grabs the line containing labeled as TIME
                label_index, time_label = self._find_data_label(line_parts)
                data_dict[time_label] = line_parts[:label_index]

            else:
                # Header line
                head_info = line_parts

        if head_info is None:
            raise ValueError("Header line not found in storm block.")

        # Create pickle key and store structured data
        pickle_key = self._make_key(head_info)
        self.pickle_dict[pickle_key] = {
            "storm_name": head_info[0],
            "report_date": head_info[1],
            "report_time": head_info[2],
            "max_winds": head_info[3],
            "center_lat": head_info[4],
            "center_lon": head_info[5],
            "mslp": head_info[6],
            "atcf_id": head_info[7],
            "data": data_dict,
        }

    def stock_pickle(self) -> None:
        """
        Parse the entire dat file and create a pickle dictionary of storms.
        Saves the pickle file to `self.output_file`.
        """
        print(f"Processing and saving to: {self.output_file}")

        head_index = None
        for idx, line in enumerate(self.file_lines):
            if "HEAD" in line:
                head_index = idx
            if "LAST" in line:
                # Convert current storm block to dict
                self._convert_to_dict((head_index, idx))

        # Save final pickle
        with open(self.output_file, "wb") as f:
            pickle.dump(self.pickle_dict, f)
