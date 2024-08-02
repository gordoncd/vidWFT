# Documentation for [`tracker.py`](../vidWFT/core/tracker.py)

## Overview

This module provides functions to initialize and update OpenCV object trackers for tracking objects in video frames.

## Functions

### [`tracker_init`](../vidWFT/core/tracker.py)

```python
def tracker_init(frame: np.ndarray, num_stakes: int) -> tuple[list[cv2.Tracker], list[Sequence[int]]]:
    """
    Initialize OpenCV object trackers.

    Parameters:
    - frame (np.ndarray): The initial video frame to select regions of interest (ROIs).
    - num_stakes (int): The number of objects to track.

    Returns:
    - tuple[list[cv2.Tracker], list[Sequence[int]]]: A tuple containing a list of OpenCV Tracker objects and a list of selected ROIs.
    """
```

**Description:**
This function initializes OpenCV object trackers by allowing the user to select regions of interest (ROIs) in the given frame. It returns a list of tracker objects and the corresponding ROIs.

**Parameters:**
- `frame` (np.ndarray): The initial video frame to select regions of interest (ROIs).
- `num_stakes` (int): The number of objects to track.

**Returns:**
- `tuple[list[cv2.Tracker], list[Sequence[int]]]`: A tuple containing a list of OpenCV Tracker objects and a list of selected ROIs.

### `trackers_update`

```python
def trackers_update(trackers: list[cv2.Tracker], frame: np.ndarray, cur_frame_num: int, position: np.ndarray, show: bool = True) -> np.ndarray:
    """
    Update OpenCV object trackers.

    Parameters:
    - trackers (list[cv2.Tracker]): A list of OpenCV Tracker objects.
    - frame (np.ndarray): The current video frame as a NumPy array.
    - cur_frame_num (int): The current frame number in the video sequence.
    - position (np.ndarray): The initial position of the object to track.
    - show (bool, optional): If True, display the tracking result. Defaults to True.

    Returns:
    - np.ndarray: The updated position of the tracked objects.
    """
```

**Description:**
This function updates the positions of the tracked objects in the current video frame. It optionally displays the tracking results by drawing bounding boxes around the tracked objects.

**Parameters:**
- [`trackers`](../vidWFT/core/tracker.py) (list[cv2.Tracker]): A list of OpenCV Tracker objects.
- [`frame`](../vidWFT/core/tracker.py) (np.ndarray): The current video frame as a NumPy array.
- [`cur_frame_num`](../vidWFT/core/tracker.py) (int): The current frame number in the video sequence.
- [`position`](../vidWFT/core/tracker.py) (np.ndarray): The initial position of the object to track.
- [`show`](../vidWFT/core/tracker.py) (bool, optional): If True, display the tracking result. Defaults to True.

**Returns:**
- [`np.ndarray`](../../../../opt/anaconda3/lib/python3.9/site-packages/numpy/__init__.py"): The updated position of the tracked objects.

## Usage Example

```python
import cv2
import numpy as np
from tracker import tracker_init, trackers_update

# Load a video
cap = cv2.VideoCapture('path/to/video.mp4')

# Read the first frame
ret, frame = cap.read()

# Initialize trackers
num_stakes = 3
trackers, regions = tracker_init(frame, num_stakes)

# Prepare position array
position = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), num_stakes, 2))

# Process video frames
cur_frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Update trackers
    frame = trackers_update(trackers, frame, cur_frame_num, position, show=True)
    
    # Display the frame
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    cur_frame_num += 1

cap.release()
cv2.destroyAllWindows()
```

## File Location

- `tracker.py`