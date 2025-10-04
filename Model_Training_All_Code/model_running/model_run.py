# coding=utf-8
"""
@author: John Mark Mayhall
Optimized: 10/02/2025

Purpose:
This script takes preprocessed test data and runs it through a pre-trained CNN model.
It generates and saves probability images contoured over the original images.
"""

from overlay_setup import Overlay

# -----------------------------
# Run overlay processing
# -----------------------------
if __name__ == "__main__":
    # -----------------------------
    # Paths to model and data arrays
    # -----------------------------
    BASE_PATH = '/rstor/jmayhall/Model_Training_Code/cnn_creation'
    TEST_PATH_BRIGHT = "/rstor/jmayhall/Model_Training_Full/scaled_arrays/test/"
    CH8_PATH = "/rstor/jmayhall/Model_Training_Full/ch8/scaled_arrays/test/"
    LATLON_PATH = '/rstor/jmayhall/Model_Training_Full/test_latlon_arrays/'

    # -----------------------------
    # Mapping month numbers to names
    # -----------------------------
    MONTH_DICT = {
        '01': 'January', '02': 'February', '03': 'March', '04': 'April',
        '05': 'May', '06': 'June', '07': 'July', '08': 'August',
        '09': 'September', '10': 'October', '11': 'November', '12': 'December'
    }
    # Create the Overlay interface
    overlay_interface = Overlay(
        model_path=BASE_PATH,
        test_bright_path=TEST_PATH_BRIGHT,
        month_dict=MONTH_DICT,
        latlon_path=LATLON_PATH,
        ch8_path=CH8_PATH
    )

    # Execute main processing routine
    overlay_interface.main()
