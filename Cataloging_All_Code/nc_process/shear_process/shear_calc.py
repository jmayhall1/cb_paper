# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall

Class for calculating TCB occurrences in shear-relative quadrants
and extracting related statistics from probability arrays.
"""
import numpy as np


class Shear:
    """
    Class for calculating quadrant probabilities based on shear vectors.
    """

    def __init__(self, azis: np.ndarray, vectordir: float, current_bearing: float,
                 azi_shear_rel_func: np.ndarray, bearing_rel: np.ndarray, card_rel: np.ndarray,
                 cut_off_func: float, prob_file: np.ndarray, current_id: str,
                 radius: np.ndarray, tc_time, time_lon: str, time_dict: dict, shear_dict: dict):
        """
        Initialize the shear calculator.

        :param azis: Array of azimuths for each point (0-360°).
        :param vectordir: Shear vector direction (0-360°).
        :param current_bearing: Storm motion vector (0-360°).
        :param azi_shear_rel_func: Precomputed azimuth relative to shear.
        :param bearing_rel: Bearing relative to storm motion.
        :param card_rel: Cardinal motion relative.
        :param cut_off_func: Probability threshold.
        :param prob_file: Probability array.
        :param current_id: Storm identifier.
        :param radius: Array of radial distances.
        :param tc_time: Timestamp of observation.
        :param time_lon: Center longitude.
        :param time_dict: Dictionary for diurnal binning.
        :param shear_dict: Dictionary for mapping based on the shear quadrant.
        """
        self.azis = azis.astype(np.float32)
        self.vectordir = float(vectordir)
        self.current_bearing = float(current_bearing)
        self.azi_shear_rel_func = azi_shear_rel_func.astype(np.float32)
        self.bearing_rel = bearing_rel.astype(np.float32)
        self.card_rel = card_rel.astype(np.float32)
        self.cut_off_func = float(cut_off_func)
        self.prob_file = prob_file.astype(np.float32)
        self.current_id = current_id
        self.radius = radius.astype(np.float32)
        self.time = tc_time
        self.time_lon = time_lon
        self.time_dict = time_dict
        self.shear_dict = shear_dict

    def vector_relative_azi(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute azimuths of points relative to a vector (shear or motion).
        Handles negative differences by wrapping around 360°.
        :return: Tuple of azimuth arrays relative to shear, storm motion, and original
        """
        diffs = [
            (self.azis - self.vectordir) % 360.0,
            (self.azis - self.current_bearing) % 360.0,
            self.azis.copy()
        ]
        return diffs[0], diffs[1], diffs[2]

    def quad_probs(self) -> list:
        """
        Determine if TCBs exist in shear-relative quadrants.
        Uses a probability threshold to mask relevant points and then count occurrences
        per quadrant and per diurnal bin.

        :return: List of statistics including:
                 - shear_counts (u_r, u_l, d_r, d_l)
                 - radius_arr
                 - degree_arr
                 - diurnal_counts
                 - shear_flags (presence per quadrant)
                 - ri_and_wind_count
                 - bearing_arr, card_arr
                 - flattened total arrays
        """
        # Mask points exceeding cutoff
        mask = self.prob_file > self.cut_off_func

        # Flattened arrays for masked points
        radius_arr = self.radius[mask]
        degree_arr = self.azi_shear_rel_func[mask]
        bearing_arr = self.bearing_rel[mask]
        card_arr = self.card_rel[mask]

        # Count occurrences in quadrants using vectorized mapping
        shear_labels = np.array([self.shear_dict.get(int(round(d))) for d in degree_arr])
        unique, counts = np.unique(shear_labels, return_counts=True)
        shear_counts = dict(zip(unique, counts))

        # Ensure all quadrants present in dict
        for q in ['u_r', 'u_l', 'd_r', 'd_l']:
            shear_counts.setdefault(q, 0)

        # Flags for at least one occurrence in each quadrant
        shear_flags = [shear_counts[q] > 0 for q in ['u_r', 'u_l', 'd_r', 'd_l']]

        # Diurnal counts
        diurnal_labels = np.array([self.time_dict[self.time]] * len(degree_arr))
        diurnal_counts = [np.sum(diurnal_labels == self.time_dict[h]) for h in range(24)]

        # Total number of masked points
        ri_and_wind_count = len(degree_arr)

        # Flattened total arrays (for reference / plotting)
        rad_total = self.radius.flatten()
        degree_total = self.azi_shear_rel_func.flatten()
        bear_total = self.bearing_rel.flatten()
        card_total = self.card_rel.flatten()

        return [
            list(shear_counts.values()),  # Shear counts per quadrant
            radius_arr,
            degree_arr,
            diurnal_counts,
            shear_flags,
            ri_and_wind_count,
            ri_and_wind_count,
            ri_and_wind_count,
            bearing_arr,
            card_arr,
            rad_total,
            degree_total,
            bear_total,
            card_total
        ]
