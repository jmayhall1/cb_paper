# coding=utf-8
"""
CNN Model Training Script
@author: Andrew White
@author: John Mark Mayhall
Optimized: 10/02/2025

Purpose:
This script loads training and validation data from preprocessed numpy arrays, initializes a CNN via the
FinalSetup module, and saves the trained model. Adjust variables like xsize, ysize, nvars, batch_size,
epochs, and paths depending on dataset size and source.

Notes:
- Requires significant memory depending on data size.
- Ensure training arrays are divisible by the architecture's expanding/contracting stages.
"""

import glob
import os

import numpy as np
import tensorflow as tf
from FinalSetup import Final

# ===================== TensorFlow Configuration =====================
print("Available devices:", tf.config.list_physical_devices())
tf.debugging.set_log_device_placement(True)  # Logs device placement for operations


# ===================== Helper Functions =====================
def load_numpy_arrays(base_path: str, folder: str, key: str) -> np.ndarray:
    """
    Load numpy arrays from a directory and extract a specific key.
    :param base_path: Base directory path
    :param folder: Folder containing the .npz files
    :param key: Key inside the .npz file to extract
    :return: NumPy array of stacked data
    """
    folder_path = os.path.join(base_path, folder)
    files = sorted(os.listdir(folder_path))
    arrays = [np.load(os.path.join(folder_path, f))[key] for f in files]
    return np.array(arrays)


def clear_logs(log_path: str):
    """
    Remove all files in a specified log directory.
    :param log_path: Directory containing log files
    """
    for f in glob.glob(os.path.join(log_path, "*")):
        os.remove(f)


# ===================== Main Training Routine =====================
if __name__ == "__main__":
    # --------------------- Configuration ---------------------
    nvars, batch_size, epochs = 2, 4, 1000
    online = 'yes'
    base_paths = {'no': 'C:/Users/jmayhall/Documents/', 'yes': '/rstor/jmayhall/'}
    base_path = base_paths[online]

    # --------------------- Load Training & Validation Data ---------------------
    # Main brightness and geojson arrays
    x_train = load_numpy_arrays(base_path, "Model_Training_Full/scaled_arrays/training", "brightness")
    y_train = load_numpy_arrays(base_path, "Model_Training_Full/geojson_array/training", "arr_0")
    x_val = load_numpy_arrays(base_path, "Model_Training_Full/scaled_arrays/verification", "brightness")
    y_val = load_numpy_arrays(base_path, "Model_Training_Full/geojson_array/verification", "arr_0")

    # Channel 8 arrays
    x_train8 = load_numpy_arrays(base_path, "Model_Training_Full/ch8/scaled_arrays/training", "brightness")
    x_val8 = load_numpy_arrays(base_path, "Model_Training_Full/ch8/scaled_arrays/verification", "brightness")

    # --------------------- Clear Previous Logs ---------------------
    log_dir = base_path.replace('rstor', 'rhome') + 'Model_Training_Code/cnn_creation/logs'
    clear_logs(log_dir)

    # --------------------- Model Dimensions ---------------------
    # Tensor shape is (num_samples, ysize, xsize)
    ysize, xsize = x_train.shape[1], x_train.shape[2]

    # --------------------- Initialize and Train CNN ---------------------
    final_act = Final(xsize=xsize, ysize=ysize, nvars=nvars, batch_size=batch_size, epochs=epochs, x_train=x_train,
                      y_train=y_train, x_val=x_val, y_val=y_val, x_train8=x_train8, x_val8=x_val8)

    # Build the U-Net model
    unet = final_act.create_all()

    # --------------------- Save Model ---------------------
    model_save_path = os.path.join(base_path, "Model_Training_Code/cnn_creation/new/model.keras")
    unet.save(model_save_path)
    print(f"Model saved at {model_save_path}")
