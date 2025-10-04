# coding=utf-8
"""
Optimized: 10/02/2025
@author: John Mark Mayhall
Purpose:
    Compare filenames across three directories of .npz files
    (C13 scaled, C13 unscaled, and C08 scaled). The goal is to
    find which files exist in one set but not in the others.
"""

import glob


def trim_filenames(file_list: list[str], base_len: int, suffix_len: int) -> set[str]:
    """
    Extracts the "core" part of filenames by slicing off a fixed prefix and suffix.

    Args:
        file_list (list[str]): List of full file paths.
        base_len (int): Number of characters to strip from the start.
        suffix_len (int): Number of characters to strip from the end.

    Returns:
        set[str]: Unique trimmed filenames (core identifiers).
    """
    return {f[base_len:-suffix_len] for f in file_list}


if __name__ == '__main__':
    # --- Load file lists ---
    c13_scaled = glob.glob('//uahdata/rstor/cataloging/nc_process/plotting/C13_scaled/*.npz')
    c13_unscaled = glob.glob('//uahdata/rstor/cataloging/nc_process/plotting/C13_unscaled/*.npz')
    c08_scaled = glob.glob('//uahdata/rstor/cataloging/nc_process/plotting/C08_scaled/*.npz')

    # --- Trim down to core identifiers ---
    # Note: These slice values (58, 15, etc.) depend on path length conventions.
    # A safer option would be to use os.path.basename() if the suffix lengths are consistent.
    c13_trim = trim_filenames(c13_scaled, 58, 15)
    c13_u_trim = trim_filenames(c13_unscaled, 60, 17)
    c08_trim = trim_filenames(c08_scaled, 58, 15)

    # --- Compare sets ---
    print("C13 scaled but not in C13 unscaled:", c13_trim - c13_u_trim)
    print("C13 scaled but not in C08 scaled:", c13_trim - c08_trim)

    print("C13 unscaled but not in C13 scaled:", c13_u_trim - c13_trim)
    print("C13 unscaled but not in C08 scaled:", c13_u_trim - c08_trim)

    print("C08 scaled but not in C13 unscaled:", c08_trim - c13_u_trim)
    print("C08 scaled but not in C13 scaled:", c08_trim - c13_trim)
