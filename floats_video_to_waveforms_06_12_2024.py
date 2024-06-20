'''
This file contains function for converting videos to waveforms
NOTE: This script does NOT apply intrinsic matrices of the camera 
to adjust for distortion

Author: Gordon Doore
Created: 06/12/2024

Last Modified: 06/14/2024

'''
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import orthorec_06_03_2024 as orth
from concurrent.futures import ThreadPoolExecutor


def rect_floats_video_to_waveform(rectified_video_path, ppm, num_stakes, 
                             arr_out_path = 'wave_measurements.npy',
                             graph_out_path = 'position_graphs.png', show= True):
    '''
    Converts a rectified video of floating objects to waveforms.
    
    Parameters:
    rectified_video_path (str): The path to the rectified video file.
    ppm (float): The pixels per meter conversion factor.
    num_stakes (int): The number of floating objects to track.
    arr_out_path (str, optional): The output path for the waveform measurements array. Defaults to 'wave_measurements.npy'.
    graph_out_path (str, optional): The output path for the position graphs. Defaults to 'position_graphs.png'.
    show (bool, optional): Whether to display the tracking frames. Defaults to True.
    
    Returns:
    numpy.ndarray: The waveform measurements array.
    '''
    
    # Load the video
    cap = cv2.VideoCapture(rectified_video_path)

    # Read the first frame
    ret, frame = cap.read()

    #get the total frames:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a tracker object
    trackers = []
    for i in range(num_stakes):
        roi = cv2.selectROI(frame, False)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        tracker = cv2.legacy_TrackerCSRT.create()
        trackers.append(tracker)
        ret = tracker.init(frame, roi)
    # Initialize the tracker

    position = np.zeros((total_frames, num_stakes,2))

    def update_tracker(args):
        i,tracker = args
        ret, roi = tracker.update(frame)
        if ret:
            (x, y, w, h) = tuple(map(int, roi))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #record the position of the center of the box 
            center_x = x + w // 2
            center_y = y + h // 2
            return (i, (center_x, center_y))
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            return None

    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            break
        #get current frame number:
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
        # Update each tracker
        with ThreadPoolExecutor() as executor:
            positions = executor.map(update_tracker, enumerate(trackers))
        for pos in positions:
            if pos is not None:
                position[current_frame, pos[0]] = pos[1]
        if show:
            # Display the resulting frame
            cv2.imshow('Tracking', frame)

            # Exit if ESC key is pressed
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

    #convert position to real units: 
    position = position/ppm

    # Plot the y coordinates through time
    fig = plt.figure()
    for i in range(num_stakes):
        name = 'stake'+str(i)
        plt.plot(position[:,i,1],label = name)
    plt.xlabel('Time')
    plt.ylabel('Position (m)')
    plt.legend()
    fig.savefig(graph_out_path)

    np.save(arr_out_path,position)
    return position

def unrectified_to_rect_to_waveform(video_path, ppm, num_stakes,rect_path, 
                            arr_out_path = 'wave_measurements.npy',
                            graph_out_path = 'position_graphs.png', 
                            threshold_condition = lambda x: np.sum(x,axis=1)<300,
                            show = True):
    '''Converts an unrectified video of floating objects to waveforms.
    
    Parameters:
    video_path (str): The path to the rectified video file.
    ppm (float): The pixels per meter conversion factor.
    num_stakes (int): The number of floating objects to track.
    arr_out_path (str, optional): The output path for the waveform measurements array. Defaults to 'wave_measurements.npy'.
    graph_out_path (str, optional): The output path for the position graphs. Defaults to 'position_graphs.png'.
    threshold_condition (function): Function that is applied to ndarray which threhsolds based on some intrinsic value
    show (bool, optional): Whether to display the tracking frames. Defaults to True.
    
    Returns:
    numpy.ndarray: The waveform measurements array.
    '''
    #first we rectify our image:
    orth.rectify_video_by_gradation(video_path,rect_path, threshold_condition,show)
    return rect_floats_video_to_waveform(rect_path, ppm, num_stakes, arr_out_path, 
                             graph_out_path,show)


def unrectified_to_waveform(video_path, num_stakes, threshold_condition, show = True):
    '''
    
    '''
    
    #load in video:
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame = cap.read()

    #get the total frames:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #next, we extract the points of the gradations for n_stakes
    #for now we have the user select these points.
    all_points, all_lines = orth.define_stakes(frame,num_stakes)

if __name__ == '__main__':
    # floats_video_to_waveform('videos/noodle_float_move_rect.mp4',750,2)

    unrectified_path = 'videos/floats_perp_5k_none.MP4'
    ppm = 750
    num_stakes = 2
    rect_path = 'videos/rectified_case.mp4'

    unrectified_to_rect_to_waveform(unrectified_path, ppm, num_stakes, rect_path,
                                     show = True)