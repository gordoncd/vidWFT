'''
NOTE: THIS IS AN ARCHIVED COPY OF THE ORIGINAL FILE -- some functions have been removed from the original because they are not used/working
This file contains function for converting videos to waveforms
NOTE: This script does NOT apply intrinsic matrices of the camera 
to adjust for distortion

Author: Gordon Doore
Created: 06/12/2024

Last Modified: 07/10/2024

'''
import cv2 #type:ignore
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import orthorec as orth
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Tuple, Sequence
from collections.abc import Iterable
import time
from numba import jit #type: ignore
import os
import subprocess
import pandas as pd
import export 


def rect_floats_video_to_waveform(rectified_video_path, ppm, num_stakes, 
                             arr_out_path = 'wave_measurements.npy',
                             graph_out_path = 'position_graphs.png', show= True):
    '''
    Converts a rectified video of floating objects to waveforms.
    
    Args:
        rectified_video_path (str): The path to the rectified video file.
        ppm (float): The pixels per meter conversion factor.
        num_stakes (int): The number of floating objects to track.
        arr_out_path (str, optional): The output path for the waveform measurements array. Defaults to 'wave_measurements.npy'.
        graph_out_path (str, optional): The output path for the position graphs. Defaults to 'position_graphs.png'.
        show (bool, optional): Whether to display the tracking frames. Defaults to True.
    
    Returns:
        numpy.ndarray: The waveform measurements array.
    '''
    
    position = track_objects_in_video(rectified_video_path,num_stakes, show = show)

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

def unrectified_to_rect_to_waveform(video_path : str, ppm : np.ndarray, num_stakes : int,rect_path : str, 
                            arr_out_path : str = 'wave_measurements.npy',
                            graph_out_path : str= 'position_graphs.png', 
                            threshold_condition : Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] = lambda x: np.sum(x,axis=1)<300,
                            show : bool = True) -> np.ndarray:
    '''Converts an unrectified video of floating objects to waveforms.
    
    Args:
        video_path (str): The path to the rectified video file.
        ppm (float): The pixels per meter conversion factor.
        num_stakes (int): The number of floating objects to track.
        rect_path (str): path to save rectified video to
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


def track_objects_in_video(cap : cv2.VideoCapture, num_stakes: int, show : bool =False, track_every : int = 1) -> np.ndarray:
    """
    Tracks objects in a video using cv2 object tracker.

    Args:
        video_path (str): Path to the video file.
        num_stakes (int): Number of objects to track.
        show (bool, optional): If True, display the tracking process in real-time.
        track_every (int, optional): how often tracking occurs

    Returns:
        np.ndarray: An array containing the positions of the tracked objects.
    """
    
    ret, frame = cap.read()
    trackers, _ = tracker_init(frame, num_stakes)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    position = np.zeros((total_frames, num_stakes, 2))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if current_frame%track_every != 0:
            continue
        trackers_update(trackers,frame,current_frame,position)
        if show:
            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break
                break

    cap.release()
    cv2.destroyAllWindows()
    return position

def unrectified_to_waveform(video_path : str, num_stakes : int, track_every : int, show : bool = True) -> tuple[np.ndarray,Any]:
    '''
    converted unrectified (calibrated) video to waveform

    Args:
        video_path (str): path to unrectified video to be processed
        num_stakes (int): number of stakes to be tracked in video
        track_every (int): frequency to track object movement
        show (bool, optional): whether or not to show video while tracking

    
    Returns: 
        positions (np.ndarray): positions of tracked floats over the input video 
        ppm (np.ndarray): pixel to meter coefficients for at each stake
    
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
    arr_all_points = np.array(all_points)
    #assuming the user chooses points corresponding to the gradations
    #we use this to save the ppm for each stake:
    ppm = np.linalg.norm(arr_all_points[:,0]-arr_all_points[:,1],axis = 1)
    
    #after getting ppm, track each stake in the input space:
    
    positions = track_objects_in_video(cap, num_stakes, show = show, track_every = track_every)

    #apply derived ppm to the positions: 
    positions = positions/ppm#not sure if the axis work out written like this

    #save array
    return positions,ppm

def raw_video_to_waveform(video_path : str, calibration_data : tuple, num_stakes : int, track_every : int, show : bool = True, save_cal : bool = False) -> np.ndarray:
    '''
    converts raw (uncalibrated) video to waveform

    Args:
        video_path (str): path to unrectified video to be processed
        calibration_data (tuple): tuple of ndarrays (matrix_data, dist_data)
        num_stakes (int): number of stakes to be tracked in video
        track_every (int): frequency to track object movement
        show (bool): whether or not to show video while tracking
        save_cal (bool): whether or not to save calibrated video

    Returns: 
        positions (np.ndarray): positions of tracked floats over the input video 
    '''
    #open video 
    cap = cv2.VideoCapture(video_path)
    mtx, dist = calibration_data
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #on the first frame undistort and then select ppm
    ret, frame = cap.read()
    #get time to do this: 
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, mtx)
    
    #get ppm on undistorted frame: 
    all_points, all_lines = orth.define_stakes(undistorted_frame,num_stakes)
    all_points_arr = np.array(all_points)
    #assuming the user chooses points corresponding to the gradations
    #we use this to save the ppm for each stake:
    ppm = np.linalg.norm(all_points_arr[:,0]-all_points_arr[:,1],axis = 1) #type: ignore

    #define floats to track
    trackers, _ = tracker_init(undistorted_frame,num_stakes)
    position = np.zeros((total_frames, num_stakes, 2))
    #apply calibration: 
    while ret:
        start_time = time.time()
        loop_start = start_time
        ret, frame = cap.read()
        io_time = time.time() - start_time
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if current_frame % track_every != 0 or frame is None: 
            continue
        
        height, width = frame.shape[:2]

        # Start timer
        start_time = time.time()

        #cv2 undistort
        undistorted_frame = cv2.undistort(frame, mtx, dist, None, mtx)
        cv2_undistort_time = time.time() - start_time
        
        # Start timer for trackers_update
        start_time = time.time()
        
        trackers_update(trackers, undistorted_frame, current_frame, position)
        
        # Calculate trackers_update time
        trackers_update_time = time.time() - start_time
        
        if show:
            cv2.imshow('Tracking', undistorted_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break
                break
        
        # Calculate total loop time
        total_loop_time = time.time() - loop_start
        
        # Calculate proportion of total loop time for each chunk
        io_proportion = io_time / total_loop_time
        undistort_frame_proportion = cv2_undistort_time / total_loop_time
        trackers_update_proportion = trackers_update_time / total_loop_time
        
        # Print the proportions
        print("I/O Proportion:",io_proportion)
        print("Undistort Frame Proportion:", undistort_frame_proportion)
        print("Trackers Update Proportion:", trackers_update_proportion)

    
    #apply derived ppm to the positions: 
    position_real_space : np.ndarray = position/np.reshape(ppm,(1,4,1))
        
    cap.release()
    cv2.destroyAllWindows()
    return position_real_space

def load_camera_calibration_data(matrix_path : str, distance_coefficient_path : str) -> tuple[Any,Any]:
    '''
    load calibration matrices. Helper function for `test_raw_video_to_waveform()`

    Args:
        matrix_path (str): path to camera calibration matrix
        distance_coefficient_path (str): path to distance coefficent matrix

    Returns:
        matrix_array (np.ndarray): array of camera calibration matrix
        distance_coefficient_array (np.ndarray): array of distance coefficeint matrix
    '''
    return np.load(matrix_path), np.load(distance_coefficient_path)

def test_raw_video_to_waveform(video_path : str,matrix_path : str,distance_coefficient_path : str, num_stakes: int, track_every : int, show : bool, save_cal : bool) ->np.ndarray:
    '''
    test for `raw_video_to_waveform()` gets calibration data and runs the calibration/waveform function

    Args:
        video_path (str): path to unrectified video to be processed
        matrix_path (str): path to camera matrix array
        distance_coefficient_path (str): path to dist coefficient array
        num_stakes (int): number of stakes to be tracked in video
        track_every (int): frequency to track object movement
        show (bool): whether or not to show video while tracking
        save_cal (bool): whether or not to save calibrated video    

    Returns: 
        positions (np.ndarray): ndarray of the positions across the video for the given conditions
    '''

    calibration_data = load_camera_calibration_data(matrix_path, distance_coefficient_path)
    return raw_video_to_waveform(video_path, calibration_data,num_stakes,track_every, show, save_cal)



if __name__ == '__main__':
    #unrectified_path = 'videos/floats_perp_4k_none.MP4'
    unrectified_path = 'videos/5k_perp_salmon.MP4'
    
    # num_stakes = 2
    # rect_path = 'videos/rectified_case.mp4'

    # positions,ppm = unrectified_to_waveform(unrectified_path, num_stakes, show = True, track_every = 5)
    # print(positions)
    # print(type(positions))
    # print(ppm)
    # framerate = 30 
    # # Plot the y coordinates through time
    # fig = plt.figure()
    # for i in range(num_stakes):
    #     name = 'stake '+str(i)
    #     plt.plot(np.arange(positions[2:,i,1].shape[0])/framerate,positions[2:,i,1],label = name)
    # plt.xlabel('Time')
    # plt.ylabel('Position')
    # plt.legend()
    # fig.savefig('graph1.png')

    # np.save('array2.npy',positions[2:])
    matrix_path = 'calibration/acortiz@colbydotedu_CALIB/camera_matrix_5k.npy'
    dist_path = 'calibration/acortiz@colbydotedu_CALIB/dist_coeff_5k.npy'
    calibration_data = load_camera_calibration_data(matrix_path, dist_path)
    export.prepare_files(unrectified_path,np.ones((1000,4)),calibration_data, 
                np.arange(2), dest = 'output_data/test_salmon_perp_5k/', 
                graph_dest = 'output_figures/test_salmon_perp_5k.png', 
                raw_csv_dest = 'output_data/test_salmon_perp_5k/positions_raw.csv', 
                clean_csv_dest = 'output_data/test_salmon_perp_5k/positions_clean.csv', 
                txt_dest = 'output_data/test_salmon_perp_5k/metadata.txt',
                raw_headers = ['one','two','red','blue'])
    #graph_out = 'output_figures/test_salmon_perp_5k.png'
    # positions = test_raw_video_to_waveform(unrectified_path,matrix_path, dist_path, 2, 5,False, False)
    # np.save('output_data/test_5k_salmon.npy',positions)
    # plot_wave_positions(positions, graph_out)

