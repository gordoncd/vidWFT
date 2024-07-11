
import numpy as np
import cv2 
from typing import Sequence


def tracker_init(frame: np.ndarray, num_stakes : int) -> tuple[list[cv2.Tracker],list[Sequence[int]]]:
    '''
    initialize cv2 object trackers 
    NOTE: will add ability to change tracker type later
    
    '''
    trackers = []
    regions = []
    for i in range(num_stakes):
        roi = cv2.selectROI("Select ROI", frame, False)
        regions.append(roi)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        tracker = cv2.legacy_TrackerCSRT.create() #type: ignore
        trackers.append(tracker)
        tracker.init(frame, roi)
    return trackers, regions

def trackers_update(trackers: list[cv2.Tracker],frame: np.ndarray, cur_frame_num : int, position : np.ndarray,show : bool= True)-> np.ndarray:
    '''
    update cv2 object trackers
    
    Parameters:
    trackers (list[cv2.Tracker]): A list of OpenCV Tracker objects.
    frame (np.ndarray): The current video frame as a NumPy array.
    cur_frame_num (int): The current frame number in the video sequence.
    position (np.ndarray): The initial position of the object to track.
    show (bool, optional): If True, display the tracking result. Defaults to True.

    Returns:
        np.ndarray: The updated position of the tracked objects.
    '''
    for i, tracker in enumerate(trackers):
        success, bbox = tracker.update(frame)
        if success:
            position[cur_frame_num, i] = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2 # Store the center position
            if show:
                #draw bounding box on current_frame
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 0, 255), 2)
    return frame
