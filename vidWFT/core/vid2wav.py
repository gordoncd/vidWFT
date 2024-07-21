'''
vid2wav.py
This file contains function for converting videos to waveforms

formerly called video_to_waveform_floats.py

Author: Gordon Doore
Created: 06/12/2024

Last Modified: 07/01/2024

'''
import os
import sys
import cv2 #type:ignore
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)

import orthorec as orth
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Tuple, Sequence
from collections.abc import Iterable
import time
from numba import jit #type: ignore
import subprocess
import pandas as pd
import export 
from tracker import tracker_init, trackers_update
from calibrate import load_camera_calibration_data, undistort_video, crop_and_undistort, crop_video
from tqdm import tqdm

def crop_frame(frame: np.ndarray, roi: Tuple[int,int,int,int]) -> np.ndarray:
    '''
    crop frame to region of interest
    
    Parameters:
    frame (np.ndarray): The current video frame as a NumPy array.
    roi (Sequence[int]): The region of interest to crop the frame to.

    Returns:
        np.ndarray: The cropped frame.
    '''
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]

def calculate_window_around_float(float_position: Tuple[int,int,int,int],frame_dims : Tuple[int,int], ppm, max_wave_height: int = 0.5) -> Tuple[int,int,int,int]:
    '''
    calculate window around float

    Args:
        float_position (Tuple[int,int,int,int]): position of the float
        frame_dims (Tuple[int,int]): dimensions of the frame
        ppm (float): pixel per meter
        max_wave_height (int): maximum wave height
    Returns: 
        Tuple[int,int,int,int]: window around the float
    '''
    x, y, w, h = float_position
    #use ppm to estimate how big the window should be in pixel space
    #ppm is the pixel per meter for a particular stake
    #we use this to estimate the size of the window in meters
    #we want to be handily larger than the max_wave_height 
    #so we multiply by 2
    window_width = w*2
    window_height = ppm*max_wave_height*2

    #now since x,y,w,h are the corner of the float, we want to shift this
    #first we account for negative w or h values by converting x and y appropriately
    if w < 0:
        x += w
        w = -w #this may be wrong
    if h < 0: 
        y += h
        h = -h
    #now we want to center the window around the float
    x -= window_width/2
    if x<0:
        x=0
    y -= window_height/2
    if y<0:
        y=0

    #if x and y are greater than the frame size, we want to shift them back
    if x + window_width > frame_dims[0]:
        x = frame_dims[0] - window_width
    if y + window_height > frame_dims[1]:
        y = frame_dims[1] - window_height

    return (int(x), int(y), int(window_width), int(window_height))

def raw_v2w(video_path : str, calibration_data : tuple, num_stakes : int, track_every : int, show : bool = True, save_cal : bool = False) -> np.ndarray:
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
        np.ndarray: positions of tracked floats over the input video 
    '''
    #open video 
    cap = cv2.VideoCapture(video_path)
    mtx, dist = calibration_data
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #on the first frame undistort and then select ppm
    ret, frame = cap.read()
    #get time to do this: 
    # undistorted_frame = cv2.undistort(frame, mtx, dist, None, mtx)
    
    #get ppm on undistorted frame: 
    all_points, all_lines = orth.define_stakes(frame,num_stakes)
    all_points_arr = np.array(all_points)
    #assuming the user chooses points corresponding to the gradations
    #we use this to save the ppm for each stake:
    ppm = np.linalg.norm(all_points_arr[:,0]-all_points_arr[:,1],axis = 1) #type: ignore

    #define floats to track
    trackers, _ = tracker_init(frame,num_stakes)
    position = np.zeros((total_frames, num_stakes, 2))
    #apply calibration: 
    with tqdm(total=total_frames) as pbar:
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
            # undistorted_frame = cv2.undistort(frame, mtx, dist, None, mtx)
            # cv2_undistort_time = time.time() - start_time
            
            # Start timer for trackers_update
            start_time = time.time()
            
            trackers_update(trackers, frame, current_frame, position)
            
            # Calculate trackers_update time
            trackers_update_time = time.time() - start_time
            
            if show:
                cv2.imshow('Tracking', frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break
                    break
            
            # Calculate total loop time
            total_loop_time = time.time() - loop_start
            
            # Calculate proportion of total loop time for each chunk
            io_proportion = io_time / total_loop_time
            # undistort_frame_proportion = cv2_undistort_time / total_loop_time
            trackers_update_proportion = trackers_update_time / total_loop_time
            
            # # Print the proportions
            # print("I/O Proportion:",io_proportion)
            # # print("Undistort Frame Proportion:", undistort_frame_proportion)
            # print("Trackers Update Proportion:", trackers_update_proportion)
            
            pbar.update(1)

    
    #apply derived ppm to the positions: 
    position_real_space : np.ndarray = position/np.reshape(ppm,(1,num_stakes,1))
        
    cap.release()
    cv2.destroyAllWindows()
    return position_real_space, ppm

def cropped_v2w(video_path : str, calibration_data : tuple, num_stakes : int, track_every : int, show : bool = True, save_cal : bool = False) -> np.ndarray:
    '''
    mess not done with this yet
    cropped video to waveform
    '''
    #open video 
    cap = cv2.VideoCapture(video_path)
    mtx, dist = calibration_data
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #UNDISTORT HERE MAYBE
    
    #get ppm on undistorted fr
    # ame: 
    all_points, all_lines = orth.define_stakes(frame,num_stakes)
    all_points_arr = np.array(all_points)
    #assuming the user chooses points corresponding to the gradations
    #we use this to save the ppm for each stake:
    ppm = np.linalg.norm(all_points_arr[:,0]-all_points_arr[:,1],axis = 1) #type: ignore

    #define floats to track
    trackers, bboxes = tracker_init(frame,num_stakes)
    croppings = [calculate_window_around_float(bbox,frame.shape[:2],ppm) for bbox in bboxes]
    #redefine trackers with cropped frames; 

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

        #UNDISTORT HERE MAYBE
        
        # Start timer for trackers_update
        start_time = time.time()
        
        trackers_update(trackers, frame, current_frame, position)
        
        # Calculate trackers_update time
        trackers_update_time = time.time() - start_time
        
        if show:
            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break
                break
        
        # Calculate total loop time
        total_loop_time = time.time() - loop_start
        
        # Calculate proportion of total loop time for each chunk
        io_proportion = io_time / total_loop_time
        #undistort_frame_proportion = cv2_undistort_time / total_loop_time
        trackers_update_proportion = trackers_update_time / total_loop_time
        
        # Print the proportions
        print("I/O Proportion:",io_proportion)
        #print("Undistort Frame Proportion:", undistort_frame_proportion)
        print("Trackers Update Proportion:", trackers_update_proportion)

    
    #apply derived ppm to the positions: 
    position_real_space : np.ndarray = position/np.reshape(ppm,(1,num_stakes,1))
        
    cap.release()
    cv2.destroyAllWindows()
    return position_real_space


    pass

def test_raw_video_to_waveform(video_path : str,matrix_path : str,distance_coefficient_path : str, num_stakes: int, track_every : int, show : bool, save_cal : bool) ->np.ndarray:
    '''
    test for `raw_video_to_waveform()` gets calibration data and runs the calibration/waveform function
    NOTE: This may be moved and/or is not actually needed
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
    return raw_v2w(video_path, calibration_data,num_stakes,track_every, show, save_cal)



if __name__ == '__main__':
    #unrectified_path = 'videos/floats_perp_4k_none.MP4'
    # unrectified_path = 'videos/5k_perp_salmon.MP4'
    unrectified_path = 'videos/pingpong.mp4'
    # # num_stakes = 2
    # # rect_path = 'videos/rectified_case.mp4'

    # # positions,ppm = unrectified_to_waveform(unrectified_path, num_stakes, show = True, track_every = 5)
    # # print(positions)
    # # print(type(positions))
    # # print(ppm)
    # # framerate = 30 
    # # # Plot the y coordinates through time
    # # fig = plt.figure()
    # # for i in range(num_stakes):
    # #     name = 'stake '+str(i)
    # #     plt.plot(np.arange(positions[2:,i,1].shape[0])/framerate,positions[2:,i,1],label = name)
    # # plt.xlabel('Time')
    # # plt.ylabel('Position')
    # # plt.legend()
    # # fig.savefig('graph1.png')

    # # np.save('array2.npy',positions[2:])
    # matrix_path = 'calibration/acortiz@colbydotedu_CALIB/camera_matrix_5k.npy'
    # dist_path = 'calibration/acortiz@colbydotedu_CALIB/dist_coeff_5k.npy'
    # calibration_data = load_camera_calibration_data(matrix_path, dist_path)
    # # export.prepare_files(unrectified_path,np.ones((1000,4)),calibration_data, 
    # #             np.arange(2), dest = 'output_data/test_salmon_perp_5k/', 
    # #             graph_dest = 'output_data/test_salmon_perp_5k.png', 
    # #             raw_csv_dest = 'output_data/test_salmon_perp_5k/positions_raw.csv', 
    # #             clean_csv_dest = 'output_data/test_salmon_perp_5k/positions_clean.csv', 
    # #             txt_dest = 'output_data/test_salmon_perp_5k/metadata.txt',
    # #             raw_headers = ['one','two','red','blue'])
    # #graph_out = 'output_figures/test_salmon_perp_5k.png'
    # # positions = test_raw_video_to_waveform(unrectified_path,matrix_path, dist_path, 2, 5,False, False)
    # # np.save('output_data/test_5k_salmon.npy',positions)
    # # plot_wave_positions(positions, graph_out)
    # undistort_video(unrectified_path, calibration_data[0], calibration_data[1], 'videos/undistorted_salmon_perp_5k.mp4')



    video_path = 'videos/cropped_and_undistorted.mp4'
    # video_path = 'videos/floats_perp_5k_none.MP4'
    # video_path = 'videos/pingpong.mp4'
    matrix_path = 'calibration/acortiz@colbydotedu_CALIB/camera_matrix_5k.npy'
    dist_path = 'calibration/acortiz@colbydotedu_CALIB/dist_coeff_5k.npy'

    # #get first frame from video_path
    # cap = cv2.VideoCapture(video_path)
    # ret, frame = cap.read()
    # cap.release()
    # crop_region =  (500,1300,4000,650)
    # output_path = 'videos/cropped_and_undistorted.mp4'

    # crop_video(video_path, output_path, crop_region)
    positions, ppm = raw_v2w(video_path, load_camera_calibration_data(matrix_path, dist_path), 2, 1, False, False)
    np.save('output_data/test_salmon_perp_5k/positions_backup.npy',positions)
    export.prepare_files(video_path, positions, load_camera_calibration_data(matrix_path, dist_path),ppm, dest = 'output_data/test_salmon_perp_5k_each_frame/')



