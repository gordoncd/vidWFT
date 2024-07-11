import os
import numpy as np 
import sys

# Add the parent directory of the tests directory to sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from vidWFT.core.calibrate import load_camera_calibration_data, calibrate_camera

def test_load_camera_calibration_data():
    mtx_path = "./calibration/acortiz@colbydotedu_CALIB/camera_matrix_4k.npy"
    dist_path = "./calibration/acortiz@colbydotedu_CALIB/dist_coeff_4k.npy"
    mtx, dist = load_camera_calibration_data(mtx_path, dist_path)
    assert mtx.shape == (3,3)
    assert dist.shape == (1,5)
