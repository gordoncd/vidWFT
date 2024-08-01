
# vidWFT (Wave Float Tracker)

vidWFT (Wave Float Tracker) is a software package designed to track wave characteristics using a single camera and a horizontally restricted float.

## Author

Gordon Doore (with guidance from Professor Alejandra C. Ortiz)

## Last Updated

07/31/2024

# Directory Structure

```
.
├── README.md
├── __pycache__
├── archive
├── assets
├── calibration
├── data
├── docs
├── output_data
├── pictures
├── pytest_report.html
├── run_tests.sh
├── tests
├── vidWFT
│   └── core
│       ├── __pycache__
│       ├── calibrate.py
│       ├── export.py
│       ├── orthorec.py
│       ├── process.py
│       ├── tracker.py
│       └── vid2wav.py
└── videos
```

# Documentation

Further documentation for all functions, objects, and files in vidWFT can be found in the `docs` directory or [here](LINK TO POSTED DOCUMENTATION).

# vidWFT/core

## export.py

Python script designed for exporting extracted data/files.

Functions:

- `plot_wave_positions`: Plots wave positions based on a given numpy array and saves the plot to a specified path.
- `extract_metadata_with_ffmpeg`: Uses ffmpeg to extract metadata from a video file and returns it as a dictionary.
- `write_metadata_to_txt`: Writes metadata and other text information to a specified text file, with an option to overwrite existing files.
- `generate_figures`: Generates figures from raw position data and saves them to a specified path.
- `data_np_to_csv`: Converts a numpy array to a CSV file and saves it to a specified destination.
- `clean_raw_positions`: Cleans raw position data by removing unsampled timesteps and centering around the mean.
- `assemble_text_output`: Assembles text output for a metadata file, including video metadata, calibration data, and additional parameters.
- `prepare_files`: Prepares output files from video to waveform, including generating text metadata, saving raw and cleaned data to CSV, and generating figures.

## calibrate.py

Functions:

- `calibrate_camera`: Calibrates the camera using chessboard images in a specified directory. Uses cv2's `calibrateCamera` function.
- `load_camera_calibration_data`: Loads camera calibration data from saved files.
- `extract_calibration_frames`: Given a path to a video, removes a specified number of frames randomly.
- `undistort_video`: Undistorts a video using the provided camera matrix and distance coefficients.
- `crop_and_undistort`: Crops a frame and then applies undistortion to the cropped frame. Not currently in use anywhere.
- `adjust_calibration_matrices`: Adjusts the camera matrix and distance coefficients to account for cropping.
- `crop_video`: Crops a video based on a provided cropping region.

## orthorec.py

Functions:

- `pick_points`: Allows the user to interactively select points on an image by clicking on it. The selected points are stored and displayed on the image.
- `rectify`: Transforms an image to rectify it based on given input and output points, correcting perspective distortions.
- `order_points`: Orders a set of points such that they are arranged in a specific order (top-left, top-right, bottom-right, bottom-left).
- `two_largest`: Finds the two largest contiguous subsequences in an array.
- `find_difference_gradations`: Calculates the distances between the centers of the two largest groups of gradations in each column of pixels.
- `find_gradations`: Identifies the center points of gradations in vertical slices of an image.
- `get_ppm`: Computes the pixels per millimeter (ppm) for different stakes in an image, assuming minimal perspective distortion.
- `linear_transform`: Calculates the linear transformation parameters (slope and intercept) for perspective warping between stakes with gradations.
- `rectify_by_gradation`: Uses gradation points to rectify an image by correcting size variations, assuming equally spaced gradations in real space.
- `define_stakes`: Allows the user to draw lines on an image to define stakes, returning the coordinates and pixel columns of the stakes.

## process.py

The file defines a WaveTimeSeries object which serves as the central object for analyzing wave data. The functionality is designed to break down the oceanlyz functionality into smaller blocks for increased functionality. (IN PROGRESS)

- `__init__`: Initializes the WaveTimeSeries object with wave positions, pixels per meter, sampling rate, and a flag indicating if the data is raw. It also initializes various attributes for storing wave properties and spectral data.
- `clean_raw_positions`: Cleans raw wave position data by removing gaps, converting pixel values to real values, and centering the positions around the mean.
- `get_psd`: Returns the power spectral density (PSD) of the wave data.
- `get_spec_freq`: Returns the spectral frequency of the wave data.
- `get_spec_ang_freq`: Returns the spectral angular frequency of the wave data.
- `calc_psd`: Calculates the power spectral density, spectral frequency, and spectral angular frequency using the Fast Fourier Transform (FFT).
- `get_avg_wave_number`: Returns the average wave number.
- `get_avg_wave_length`: Returns the average wave length.
- `get_avg_wave_height`: Returns the average wave height.
- `get_avg_wave_period`: Returns the average wave period.
- `get_avg_wave_speed`: Returns the average wave speed.
- `get_significant_wave_height`: Returns the significant wave height.

## tracker.py

This file contains functions for initializing and updating object trackers using OpenCV.

- `tracker_init`: Initializes multiple object trackers by allowing the user to select regions of interest (ROIs) in a given frame. It creates and initializes tracker objects for each selected ROI.
- `trackers_update`: Updates the positions of the tracked objects in each frame of a video sequence. It also optionally displays the tracking results by drawing bounding boxes around the tracked objects.

## vid2wav.py

This file contains several functions related to processing video frames to extract waveform data.

- `crop_frame`: Crops a given video frame to a specified region of interest.
- `calculate_window_around_float`: Calculates a window around a floating object in the video frame, considering the frame dimensions and pixel-per-meter ratio.
- `raw_v2w`: Processes a raw, uncalibrated video to extract the positions of tracked floats over the video frames, applying calibration data and tracking object movement at specified intervals.
- `cropped_v2w`: Similar to raw_v2w, but processes a cropped version of the video to extract waveform data.
- `test_raw_video_to_waveform`: A test function that loads calibration data and runs the raw_v2w function to process the video and extract waveform data.

______

# calibration

`calibration` represents a directory where calibration matrices and distance coefficents are stored. Additionally, images to obtain these matrices are stored here.  As it stands, the camera matrices and distance coefficents included are for the GoPro Hero 12 Black (with name acortiz@colbydotedu) in 1080p, 4k, and 5.3k video mode with horizon lock on.  A new matrix can be obtained using `vidWFT/core/calibrate.py`.


____

# tests/validation

the `tests` directory contains all existing tests.  The tests are designed to work with pytest and the shell script file `run_tests.sh` should execute all tests in the tests directory. Test files must start with the string 'test'.

Existing tests are not complete and much of the existing code does not have any test infrastructure to ensure its validity.  Additional work should be done towards ensuring the functionality of the included code. 

_____

# archive

archived/overwritten functions which could become useful someday.  This code is not documented and not designed to be used.  There are also some archived images and extracted waveforms here which are not meant to be contextualized.

