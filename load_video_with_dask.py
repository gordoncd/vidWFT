import cv2 #type:ignore
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import orthorec_06_03_2024 as orth
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Tuple
from collections.abc import Iterable
import time
from floats_video_to_waveforms_06_12_2024 import tracker_init, trackers_update
from dask import array as da
from dask import delayed

def define_discontig_slice_from_dask(dask_arr : da.Array, slices_list: Iterable, num_stakes: int):
    '''
    Loads a discontiguous chunk from a dask array into memory

    Args:
        dask_arr (dask.array): dask array to load chunk from
        chunk_inds (int): indices of chunk to load

    Returns:
        np.ndarray: chunk of dask array loaded into memory
    '''

    def load_chunk(chunk):
        '''
        here to leave room for different chunk loading methods

        Args:
            chunk (np.ndarray): chunk to load

        Returns:   
            np.ndarray: chunk (unloaded)
        '''
        return chunk #NOTE: may need to actually compute this to do stuff that we want
    
    def pad_chunk(chunk, max_rows, max_cols):
        '''
        Pads a chunk to the maximum size of the chunks in slices_list so they can be concatenated

        Args:
            chunk (np.ndarray): chunk to pad
            max_rows (int): maximum number of rows in slices_list
            max_cols (int): maximum number of columns in slices_list
        Returns: 
            np.ndarray: padded chunk
        '''
        # Calculate padding needed for rows and columns
        pad_rows = max_rows - chunk.shape[1]
        pad_cols = max_cols - chunk.shape[2]
        
        # Apply padding if necessary
        if pad_rows > 0 or pad_cols > 0:
            chunk = da.pad(chunk, ((0, 0), (0, pad_rows), (0, pad_cols), (0, 0)), mode='constant', constant_values=0)
        return chunk
    
    # Assuming dask_arr is defined and slices_list is given
    delayed_chunks = []
    max_rows = max(slice[1].stop - slice[1].start for _, slice in slices_list) #NOTE: dont know how this works
    max_cols = max(slice[0].stop - slice[0].start for slice, _ in slices_list)

    for rows_slice, cols_slice in slices_list:
        delayed_chunk = delayed(load_chunk)(dask_arr[:, rows_slice, cols_slice, :])
        padded_chunk = delayed(pad_chunk)(delayed_chunk, max_rows, max_cols)
        delayed_chunks.append(padded_chunk)

    # concat the padded chunks                           NOTE: shape may be num_frames instead of None here
    roi_slices_concat = da.concatenate([da.from_delayed(chunk, shape=(None, max_rows, max_cols, 3), dtype=dask_arr.dtype) for chunk in delayed_chunks], axis=1)

    bboxes : list= []
    roi_height, roi_width = roi_slices_concat.shape[1:3]
    slice_height = int(roi_height/num_stakes)

    
    #TODO: define chunk size?
    return roi_slices_concat

def derive_bboxes_from_regions(regions : list, roi_slices_concat : da.Array, num_stakes : int):
    '''
    Derives bounding boxes in the concatenation of regions from their original coordinates

    Args:
        regions (Iterable): regions to derive bounding boxes from

    Returns:
        Iterable: bounding boxes derived from regions
    '''
    bboxes : list= []
    roi_height, roi_width = roi_slices_concat.shape[1:3]
    slice_height = int(roi_height/num_stakes)
    #because all regions MUST be the same size, we can just use the first region to get the size, then iterate over all regions
    
    for i in range(num_stakes):
        width,height = regions[i][2:4]

        bboxes.append()

        
    #TODO: figure out how to do this


    return bboxes

def tracker_init_dask(dask_arr : da.Array, num_stakes : int, ppm : np.ndarray, max_wave_ht: float = 0.5,):
    '''
    Initializes trackers on a dask array

    Args:
        dask_arr (dask.array): dask array to initialize trackers on
        num_stakes (int): number of stakes to track

    Returns:
        cv2.Tracker: initialized tracker
    '''
    #get the first frame of the dask array
    frame = dask_arr[0].compute()
    #select ROI on the first frame
    regions = []
    for i in range(num_stakes):
        regions.append(cv2.selectROI(frame))
    
    #now define the area where the floats COULD go based on max_wave_height
    #this will generate our slices_list
    slices_list = []
    for stake_num, region in enumerate(regions):
        x,y,w,h = region
        #we define the area where the floats could go
        #as a rectangle around the ROI
        #we will track the floats in this area
        slices_list.append((slice(y-max_wave_ht*ppm[stake_num],y+h+2*max_wave_ht*ppm[stake_num]),slice(x-int(w/2),x+w)))

    roi_slice = define_discontig_slice_from_dask(dask_arr, slices_list)
    
    #initialize trackers on the sliced dask array
    #derive the new bboxes from the slices_list
    bboxes = derive_bboxes_from_regions(regions)
    trackers_list = []
    for bbox in bboxes:
        tracker = cv2.TrackerCSRT_create() #type: ignore
        tracker.init(frame, bbox)
        trackers_list.append(tracker)
    

    return trackers_list, regions, roi_slice


def trackers_update_dask(trackers, roi_slice, start_frame, all_positions):
    '''
    Updates trackers on the slice of dask array on roi_slice.shape[0] frames

    Args:
        tracker (cv2.Tracker): tracker to update
        dask_arr (dask.array): dask array of frames
        target_chunk_inds (int): indices of frames to update tracker on
        position (np.ndarray): position of tracked object

    Returns:
        None: operations to ndarray are in place
    '''
    for rows_slice, cols_slice in roi_slice:
        for i in range(roi_slice.shape[0]):
            frame = roi_slice[i, rows_slice, cols_slice, :].compute()
            trackers_update(trackers, frame, start_frame + i, all_positions)
    

    return 

def raw_video_to_waveform(video_path : str, num_stakes : int, track_every : int, show : bool = True, max_wave_ht : float = 0.5) -> np.ndarray:
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
    #open video and load in all frames to a dask array:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #now, we load in just the first frame:
    ret, frame = cap.read()
    #TODO: add calibration back with check bool
    
    #get ppm on frame: 
    all_points, all_lines = orth.define_stakes(frame,num_stakes)
    all_points_arr = np.array(all_points)
    #assuming the user chooses points corresponding to the gradations
    #we use this to save the ppm for each stake:
    ppm = np.linalg.norm(all_points_arr[:,0]-all_points_arr[:,1],axis = 1) #type: ignore

    #define floats to track
    trackers, regions = tracker_init(frame,num_stakes)
    
    #using these regions, we decide on our chunk size: TODO: leave auto for now, come back later

    #put all frames into a dask array:
    frames = da.from_array(np.empty((total_frames, *frame.shape), dtype=np.uint8), chunks='auto')
    for i in range(total_frames):
        ret, frame = cap.read()
        frames[i] = frame

    #initialize trackers on the dask array:
    trackers, dask_regions, roi_slice = tracker_init_dask(frames, num_stakes, ppm, max_wave_ht)

    #now, we track the floats in the video:
    #define positions
    position = np.zeros((total_frames, num_stakes, 2))
    current_frame = 0
    while current_frame < total_frames:
        start_time = time.time()
        loop_start = start_time

        frame = frames[current_frame].compute() #TODO: frame batching?
        io_time = time.time() - start_time
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if current_frame % track_every != 0 or frame is None: 
            continue

        # Start timer for trackers_update
        start_time = time.time()
        
        trackers_update(trackers, frame, current_frame, position) #TODO: switch to dask version
        
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
        trackers_update_proportion = trackers_update_time / total_loop_time
        
        # Print the proportions
        print("I/O Proportion:",io_proportion)
        print("Trackers Update Proportion:", trackers_update_proportion)
        current_frame += 1
    
    #apply derived ppm to the positions: 
    position_real_space : np.ndarray = position/np.reshape(ppm,(1,4,1))
        
    cap.release()
    cv2.destroyAllWindows()
    return position_real_space


if __name__ == '__main__':
    unrectified_path = 'videos/5k_perp_salmon.MP4'

    raw_video_to_waveform(unrectified_path,2,6,False)