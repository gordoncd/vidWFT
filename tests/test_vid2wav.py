import os
import numpy as np 
import sys
import cv2

# Add the parent directory of the tests directory to sys.path


from vidWFT.core.vid2wav import crop_frame, calculate_window_around_float

def test_crop_frame():
    frame = np.zeros((10,10))
    roi = (1,1,5,5)
    cropped_frame = crop_frame(frame, roi)
    assert cropped_frame.shape == (5,5)

def test_calculate_window_around_float():
    float_position = (10,10,5,5)
    frame_dims = (100,100)
    ppm = 1
    max_wave_height = 0.5
    window = calculate_window_around_float(float_position, frame_dims, ppm, max_wave_height)
    assert type(window) == tuple
    assert len(window) == 4
    assert all(isinstance(i, int) for i in window)