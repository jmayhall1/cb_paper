# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall
@author: Rob Junod

Software to convert Statistical Hurricane Intensity Prediction Scheme (SHIPS) Archive dat files into a pickle file
"""
import pickle
from typing import Tuple


class Pickler:
    """
    Class to convert a SHIPS dat archive file into a pickle file for faster access.
    """

    def __init__(self, file_name: str, output_file_name: str) -> None:
        """
        Initialize Pickler with input and output file paths.
        :param file_name: SHIPS archive dat file.
        :param output_file_name: Pickle output file.
        """
        self.file_name = file_name
        self.output_file = output_file_name
        self.time_column_length = 31 if "7_day" in self.file_name else 23
        self.pickle_dict = {}
        self.file_lines = self._get_file_data()

    def _get_file_data(self) -> list[str]:
        """
        Read the dat file into memory.
        :return: List of file lines.
        """
        with open(self.file_name) as f:
            return f.readlines()

    @staticmethod
    def _find_data_label(line: list[str]) -> Tuple[int, str | None]:
        """
        Find the first label in the line that contains alphabetical characters (excluding first column).
        :param line: Split line from file.
        :return: Tuple (index of label, label string) or (-1, None) if not found.
        """
        for index, element in enumerate(line):
            if element.isalpha() and index != 0:
                return index, element
        return -1, None

    def _fill_empty_data(self, line: list[int]) -> list[int]:
        """
        Fill missing columns with placeholder 9999 to reach the expected time column length.
        :param line: List of numeric data from dat file.
        :return: Padded list of integers.
        """
        while len(line) < self.time_column_length:
            line.insert(0, 9999)
        return line

    @staticmethod
    def _make_key(head_line: list[str]) -> str:
        """
        Generate a unique pickle key for a storm.
        :param head_line: Line containing storm header info.
        :return: Pickle dictionary key.
        """
        return "_".join([head_line[-2], head_line[7], head_line[1][2:], head_line[2] + '00'])

    def _dict_conversion(self, indices: Tuple[int, int]) -> None:
        """
        Convert a segment of the dat file into a dictionary entry for the pickle file.
        :param indices: Tuple of (start_index, end_index) for the storm block.
        """
        data = self.file_lines[indices[0]:indices[1]]
        data_dict = {}
        head_info = None

        for element in data:
            line = element.rstrip("\n").split()
            if "HEAD" not in line and "TIME" not in line:
                label_index, label = self._find_data_label(line)
                if label_index != -1 and label:
                    data_line = list(map(int, line[:label_index]))
                    if label_index < self.time_column_length:
                        data_line = self._fill_empty_data(data_line)
                    data_dict[label] = data_line
            elif "TIME" in line:
                time_index, time_label = self._find_data_label(line)
                if time_index != -1 and time_label:
                    data_dict[time_label] = line[:time_index]
            else:
                head_info = line

        if head_info:
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
        Parse the entire dat file and create the pickle file.
        """
        print(f"Creating pickle: {self.output_file}")
        head_index = None

        # Single-pass over file to identify HEAD/LAST blocks
        for index, file_row in enumerate(self.file_lines):
            if "HEAD" in file_row:
                head_index = index
            elif "LAST" in file_row and head_index is not None:
                self._dict_conversion((head_index, index))

        # Save pickle file
        with open(self.output_file, "wb") as pkl:
            pickle.dump(self.pickle_dict, pkl)
