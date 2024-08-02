# Documentation for [`calibrate.py`](../vidWFT/core/calibrate.py)

## Overview

This module provides functions for camera calibration, loading calibration data, extracting frames from videos, undistorting videos, and adjusting calibration matrices. The primary use case is to calibrate a camera using chessboard images and then use the calibration data to undistort images and videos.

## Functions

### [`calibrate_camera`](../vidWFT/core/calibrate.py)

```python
def calibrate_camera(src: str, dest: str, base_filename: str = '', chessboard_size: tuple = (6,9), show: bool = False, verbose: bool = False):
```

Calibrates the camera using chessboard images.

**Args:**
- `src` (str): Path to the folder containing calibration images.
- `dest` (str): Path to the folder to save camera matrices.
- `base_filename` (str): Prefix for saved files.
- `chessboard_size` (tuple): Size of the chessboard used for calibration.
- `show` (bool): Whether to show undistorted images.
- `verbose` (bool): Whether to print detailed information.

**Returns:**
- None

### `load_camera_calibration_data`

```python
def load_camera_calibration_data(matrix_path: str, distance_coefficient_path: str) -> tuple:
```

Loads camera calibration data from saved files.

**Args:**
- [`matrix_path`](../vidWFT/core/calibrate.py) (str): Path to the camera matrix file.
- [`distance_coefficient_path`](command:_github.copilot.openSymbolFromReferences?%5B%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2FUsers%2Fgordondoore%2FDocuments%2FGitHub%2FvidWFT%2FvidWFT%2Fcore%2Fcalibrate.py%22%2C%22scheme%22%3A%22file%22%7D%2C%7B%22line%22%3A105%2C%22character%22%3A52%7D%5D ../vidWFT/core/calibrate.py) (str): Path to the distance coefficient file.

**Returns:**
- [`tuple`](../../../../.vscode/extensions/ms-python.vscode-pylance-2024.7.1/dist/typeshed-fallback/stdlib/builtins.pyi"): Camera matrix and distance coefficients.

### [`extract_calibration_frames`](../vidWFT/core/calibrate.py)

```python
def extract_calibration_frames(filepath: str, nframes: int) -> list[np.ndarray]:
```

Randomly grabs `nframes` frames from the video at `filepath` and returns them as a list of numpy arrays.

**Args:**
- `filepath` (str): Path to the video file.
- `nframes` (int): Number of frames to extract.

**Returns:**
- `list[np.ndarray]`: List of extracted frames.

### `undistort_video`

```python
def undistort_video(filepath: str, mtx: np.ndarray, dist: np.ndarray, save_path: str, show: bool = False):
```

Undistorts a video using the camera matrix and distance coefficients.

**Args:**
- [`filepath`](../vidWFT/core/calibrate.py) (str): Path to the video file.
- [`mtx`](../vidWFT/core/calibrate.py) (np.ndarray): Camera matrix.
- [`dist`](../vidWFT/core/calibrate.py) (np.ndarray): Distance coefficients.
- [`save_path`](../vidWFT/core/calibrate.py) (str): Path to save the undistorted video.
- [`show`](../vidWFT/core/calibrate.py) (bool): Whether to show the undistorted video.

**Returns:**
- None

### [`crop_and_undistort`](../vidWFT/core/calibrate.py)

```python
def crop_and_undistort(video_path: str, matrix_path: str, dist_path: str, crop_region: tuple, output_path: str):
```

Crops and undistorts a video using the camera matrix and distance coefficients.

**Args:**
- `video_path` (str): Path to the video file.
- `matrix_path` (str): Path to the camera matrix file.
- `dist_path` (str): Path to the distance coefficient file.
- `crop_region` (tuple): Crop region defined as (x, y, width, height).
- `output_path` (str): Path to save the cropped and undistorted video.

**Returns:**
- None

### `adjust_calibration_matrices`

```python
def adjust_calibration_matrices(camera_matrix: np.ndarray, dist_coeffs: np.ndarray, crop_region: tuple, W: int, H: int) -> tuple:
```

Adjusts the camera matrix and distance coefficients to account for cropping.

**Args:**
- [`camera_matrix`](../vidWFT/core/calibrate.py) (np.ndarray): Camera matrix.
- [`dist_coeffs`](../vidWFT/core/calibrate.py) (np.ndarray): Distance coefficients.
- [`crop_region`](../vidWFT/core/calibrate.py) (tuple): Crop region defined as (x, y, width, height).
- [`W`](../vidWFT/core/calibrate.py) (int): Width of the original image.
- [`H`](../vidWFT/core/calibrate.py) (int): Height of the original image.

**Returns:**
- [`tuple`](../../../../.vscode/extensions/ms-python.vscode-pylance-2024.7.1/dist/typeshed-fallback/stdlib/builtins.pyi"): Adjusted camera matrix and distance coefficients.

_____

## Examples:

### Example 1: Calibrate Camera
```python
import cv2
import numpy as np
import os
import yaml
import glob
import tqdm
from calibrate import calibrate_camera

# Paths
src = 'path/to/calibration/images'
dest = 'path/to/save/matrices'
base_filename = 'calibration_'

# Calibrate the camera
calibrate_camera(src, dest, base_filename, chessboard_size=(6, 9), show=True, verbose=True)
```

### Example 2: Load Camera Calibration Data
```python
from calibrate import load_camera_calibration_data

# Paths
matrix_path = 'path/to/save/matrices/calibration_camera_matrix.npy'
distance_coefficient_path = 'path/to/save/matrices/calibration_dist_coeff.npy'

# Load the calibration data
camera_matrix, dist_coeffs = load_camera_calibration_data(matrix_path, distance_coefficient_path)
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)
```

### Example 3: Extract Calibration Frames from a Video
```python
from calibrate import extract_calibration_frames

# Path to video file
filepath = 'path/to/video.mp4'
nframes = 10

# Extract frames
frames = extract_calibration_frames(filepath, nframes)
for i, frame in enumerate(frames):
    cv2.imshow(f'Frame {i}', frame)
    cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Example 4: Undistort a Video
```python
from calibrate import undistort_video

# Paths
filepath = 'path/to/video.mp4'
save_path = 'path/to/save/undistorted_video.mp4'
matrix_path = 'path/to/save/matrices/calibration_camera_matrix.npy'
distance_coefficient_path = 'path/to/save/matrices/calibration_dist_coeff.npy'

# Load the calibration data
camera_matrix = np.load(matrix_path)
dist_coeffs = np.load(distance_coefficient_path)

# Undistort the video
undistort_video(filepath, camera_matrix, dist_coeffs, save_path, show=True)
```

### Example 5: Crop and Undistort a Video
```python
from calibrate import crop_and_undistort

# Paths
video_path = 'path/to/video.mp4'
matrix_path = 'path/to/save/matrices/calibration_camera_matrix.npy'
dist_path = 'path/to/save/matrices/calibration_dist_coeff.npy'
output_path = 'path/to/save/cropped_undistorted_video.mp4'

# Crop region (x, y, width, height)
crop_region = (100, 100, 400, 400)

# Crop and undistort the video
crop_and_undistort(video_path, matrix_path, dist_path, crop_region, output_path)
```
