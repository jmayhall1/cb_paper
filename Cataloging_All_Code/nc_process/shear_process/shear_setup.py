# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall

Class for retrieving storm reports, calculating azimuths, and computing radius from a reference point.
"""
import pickle

import numpy as np
from pyproj import Geod


class Setup:
    """
    Handles storm data extraction and polar coordinate calculations (azimuth & distance)
    relative to a storm center.
    """

    def __init__(self, file_name: str, storm_id: str, dlat, dlon, clat: float, clon: float) -> None:
        """
        Initialize storm setup.
        :param file_name: Path to the Statistical Hurricane Intensity Prediction Scheme (SHIPS) pickle file.
        :param storm_id: ATCF ID of the storm.
        :param dlat: Latitudes of data points (array or scalar).
        :param dlon: Longitudes of data points (array or scalar).
        :param clat: Reference latitude (storm center).
        :param clon: Reference longitude (storm center).
        """
        self.file_name = file_name
        self.storm_id = storm_id
        self.dlat = dlat
        self.dlon = dlon
        self.clat = clat
        self.clon = clon

    def get_storm_reports(self) -> dict:
        """
        Retrieve all storm reports from the SHIPS pickle file for the given storm ID.
        :return: Dictionary of matching storm reports.
        """
        with open(self.file_name, 'rb') as f:
            pkl_data = pickle.load(f)

        # Filter relevant storms by ATCF ID
        return {key: pkl_data[key] for key in pkl_data if self.storm_id in key}

    def polar_gridder(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute azimuths and distances from the reference point (clat, clon)
        to each (dlat, dlon) point using geodesics.
        The value of distance is in km, azimuth in degrees 0-360.
        :return: Tuple (azimuths, distances)
        """
        geod = Geod(ellps='WGS84')

        # Handle both array and scalar inputs efficiently
        if isinstance(self.dlat, np.ndarray):
            # Create arrays of the reference coordinates to match point arrays
            ref_lon = np.full_like(self.dlon, self.clon, dtype=np.float64)
            ref_lat = np.full_like(self.dlat, self.clat, dtype=np.float64)
            azi, _, dist = geod.inv(ref_lon, ref_lat, self.dlon, self.dlat)
        else:
            # Scalar case
            azi, _, dist = geod.inv(self.clon, self.clat, self.dlon, self.dlat)

        # Normalize azimuth to 0-360 degrees
        azi = (azi + 360.0) % 360.0

        # Convert distance from meters to kilometers
        dist *= 0.001

        return azi, dist
