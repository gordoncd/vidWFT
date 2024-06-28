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

def unrectified_to_rect_to_waveform(video_path, ppm, num_stakes,rect_path, 
                            arr_out_path = 'wave_measurements.npy',
                            graph_out_path = 'position_graphs.png', 
                            threshold_condition = lambda x: np.sum(x,axis=1)<300,
                            show = True):
    '''Converts an unrectified video of floating objects to waveforms.
    
    Args:
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

def tracker_init(frame, num_stakes):
    '''
    initialize cv2 object trackers 
    NOTE: will add ability to change tracker type later
    
    '''
    trackers = []
    for i in range(num_stakes):
        roi = cv2.selectROI("Select ROI", frame, False)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        tracker = cv2.legacy_TrackerCSRT.create()
        trackers.append(tracker)
        tracker.init(frame, roi)
    return trackers

def trackers_update(trackers,frame, cur_frame_num, position,show = True):
    '''
    update cv2 object trackers
    
    '''
    for i, tracker in enumerate(trackers):
        success, bbox = tracker.update(frame)
        if success:
            position[cur_frame_num, i] = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2 # Store the center position
            if show:
                #draw bounding box on current_frame
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 0, 255), 2)
    return frame


def track_objects_in_video(cap, num_stakes, show=False, track_every = 1):
    """
    Tracks objects in a video using cv2 object tracker.

    Args:
        video_path (str): Path to the video file.
        num_stakes (int): Number of objects to track.
        show (bool): If True, display the tracking process in real-time.

    Returns:
        np.ndarray: An array containing the positions of the tracked objects.
    """
    
    ret, frame = cap.read()
    trackers = tracker_init(frame, num_stakes)

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

def unrectified_to_waveform(video_path, num_stakes, track_every, show = True):
    '''
    converted unrectified (calibrated) video to waveform

    Args:
        video_path (str): path to unrectified video to be processed
        num_stakes (int): number of stakes to be tracked in video
        track_every (int): frequency to track object movement
        show (bool): whether or not to show video while tracking

    
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
    all_points = np.array(all_points)
    #assuming the user chooses points corresponding to the gradations
    #we use this to save the ppm for each stake:
    ppm = np.linalg.norm(all_points[:,0]-all_points[:,1],axis = 1)
    
    #after getting ppm, track each stake in the input space:
    
    positions = track_objects_in_video(cap, num_stakes, show = show, track_every = track_every)

    #apply derived ppm to the positions: 
    positions = positions/ppm#not sure if the axis work out written like this

    #save array
    return positions,ppm

def raw_video_to_waveform(video_path, calibration_data, num_stakes, track_every, show = True, save_cal = False):
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
    cal_frames = []
    mtx, dist = calibration_data
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #on the first frame undistort and then select ppm
    ret, frame = cap.read()
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, mtx)
    
    #get ppm on undistorted frame: 
    all_points, all_lines = orth.define_stakes(undistorted_frame,num_stakes)
    all_points = np.array(all_points)
    #assuming the user chooses points corresponding to the gradations
    #we use this to save the ppm for each stake:
    ppm = np.linalg.norm(all_points[:,0]-all_points[:,1],axis = 1)

    #define floats to track
    trackers = tracker_init(undistorted_frame,num_stakes)
    position = np.zeros((total_frames, num_stakes, 2))
    #apply calibration: 
    while ret:
        ret, frame = cap.read()
        undistorted_frame = cv2.undistort(frame, mtx, dist)
        #get current frame number: 
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if save_cal: 
            #append calframes
            cal_frames.append(undistorted_frame)
        trackers_update(trackers,undistorted_frame,current_frame,position)
        if show:
            cv2.imshow('Tracking', undistorted_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break
                break
    
    #apply derived ppm to the positions: 
    positions = positions/ppm
        
    cap.release()
    cv2.destroyAllWindows()
    return position

def load_camera_calibration_data(matrix_path, distance_coefficient_path):
    '''
    load calibration matrices. Helper function for test_raw_video_to_waveform

    Args:
        matrix_path (str): path to camera calibration matrix
        distance_coefficient_path (str): path to distance coefficent matrix

    Returns:
        matrix_array (np.ndarray): array of camera calibration matrix
        distance_coefficient_array (np.ndarray): array of distance coefficeint matrix
    '''
    return np.load(matrix_path), np.load(distance_coefficient_path)

def test_raw_video_to_waveform(video_path,matrix_path,distance_coefficient_path, num_stakes, track_every, show, save_cal):
    '''
    test for raw_video_to_waveform() gets calibration data and runs the calibration/waveform function

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
    # floats_video_to_waveform('videos/noodle_float_move_rect.mp4',750,2)

    unrectified_path = 'videos/floats_perp_4k_none.MP4'
    # unrectified_path = 'videos/floats_R3yel_4k_uv.MP4'
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
    matrix_path = 'acortiz@colbydotedu_CALIB/camera_matrix.npy'
    dist_path = 'acortiz@colbydotedu_CALIB/dist_coeff.npy'
    test_raw_video_to_waveform(unrectified_path,matrix_path, dist_path, 2, 5,True, False)