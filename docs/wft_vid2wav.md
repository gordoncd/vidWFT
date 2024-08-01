# Documentation for [`vid2wav.py`](../vidWFT/core/vid2wav.py)

## Overview

This module provides functions to process video files and convert them into waveforms by tracking objects (floats) in the video. The main functionalities include cropping frames, calculating windows around floats, and converting raw or cropped video to waveforms.

## Functions

### [`crop_frame`](../vidWFT/core/vid2wav.py)

```python
def crop_frame(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
```

Crops a video frame to a specified region of interest (ROI).

**Parameters:**
- `frame` (np.ndarray): The current video frame as a NumPy array.
- `roi` (Tuple[int, int, int, int]): The region of interest to crop the frame to, specified as (x, y, width, height).

**Returns:**
- `np.ndarray`: The cropped frame.

### `calculate_window_around_float`

```python
def calculate_window_around_float(float_position: Tuple[int, int, int, int], frame_dims: Tuple[int, int], ppm, max_wave_height: int = 0.5) -> Tuple[int, int, int, int]:
```

Calculates a window around a float in the video frame.

**Parameters:**
- [`float_position`](../vidWFT/core/vid2wav.py) (Tuple[int, int, int, int]): Position of the float specified as (x, y, width, height).
- [`frame_dims`](../vidWFT/core/vid2wav.py) (Tuple[int, int]): Dimensions of the frame.
- [`ppm`](../vidWFT/core/vid2wav.py) (float): Pixels per meter.
- [`max_wave_height`](../vidWFT/core/vid2wav.py) (int, optional): Maximum wave height. Default is 0.5.

**Returns:**
- [`Tuple[int, int, int, int]`](../../../../opt/anaconda3/lib/python3.9/typing.py"): Window around the float specified as (x, y, width, height).

### [`raw_v2w`](../vidWFT/core/vid2wav.py)

```python
def raw_v2w(video_path: str, calibration_data: tuple, num_stakes: int, track_every: int, show: bool = True, save_cal: bool = False) -> np.ndarray:
```

Converts raw (uncalibrated) video to waveform by tracking objects in the video.

**Parameters:**
- `video_path` (str): Path to the unrectified video to be processed.
- `calibration_data` (tuple): Tuple of ndarrays (matrix_data, dist_data).
- `num_stakes` (int): Number of stakes to be tracked in the video.
- `track_every` (int): Frequency to track object movement.
- `show` (bool, optional): Whether or not to show the video while tracking. Default is True.
- `save_cal` (bool, optional): Whether or not to save the calibrated video. Default is False.

**Returns:**
- `np.ndarray`: Positions of tracked floats over the input video.

### `cropped_v2w`

```python
def cropped_v2w(video_path: str, calibration_data: tuple, num_stakes: int, track_every: int, show: bool = True, save_cal: bool = False) -> np.ndarray:
```

Converts cropped video to waveform by tracking objects in the video.

**Parameters:**
- [`video_path`](../vidWFT/core/vid2wav.py) (str): Path to the unrectified video to be processed.
- [`calibration_data`](../vidWFT/core/vid2wav.py) (tuple): Tuple of ndarrays (matrix_data, dist_data).
- [`num_stakes`](../vidWFT/core/vid2wav.py) (int): Number of stakes to be tracked in the video.
- [`track_every`](../vidWFT/core/vid2wav.py) (int): Frequency to track object movement.
- [`show`](../vidWFT/core/vid2wav.py) (bool, optional): Whether or not to show the video while tracking. Default is True.
- [`save_cal`](../vidWFT/core/vid2wav.py) (bool, optional): Whether or not to save the calibrated video. Default is False.

**Returns:**
- [`np.ndarray`](../../../../opt/anaconda3/lib/python3.9/site-packages/numpy/__init__.py"): Positions of tracked floats over the input video.

### [`test_raw_video_to_waveform`](../vidWFT/core/vid2wav.py)

```python
def test_raw_video_to_waveform(video_path: str, matrix_path: str, distance_coefficient_path: str, num_stakes: int, track_every: int, show: bool, save_cal: bool) -> np.ndarray:
```

Tests the `raw_v2w` function by loading calibration data and running the calibration/waveform function.

**Parameters:**
- `video_path` (str): Path to the unrectified video to be processed.
- `matrix_path` (str): Path to the camera matrix array.
- `distance_coefficient_path` (str): Path to the distance coefficient array.
- `num_stakes` (int): Number of stakes to be tracked in the video.
- `track_every` (int): Frequency to track object movement.
- `show` (bool): Whether or not to show the video while tracking.
- `save_cal` (bool): Whether or not to save the calibrated video.

**Returns:**
- `np.ndarray`: Positions of the tracked floats over the input video.
