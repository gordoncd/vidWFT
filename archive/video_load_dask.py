import cv2 #type:ignore
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import orthorec as orth
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Tuple
from collections.abc import Iterable
import time
from vid2wav import tracker_init, trackers_update
from dask import array as da
from dask import delayed
from tqdm import tqdm


class Dask_Float_Tracker:
    def __init__(self, video_path: str, num_stakes: int, max_wave_ht: float = 0.5):
            """
            Initialize the VideoLoader object.

            Args:
                video_path (str): The path to the video file.
                num_stakes (int): The number of stakes.
                max_wave_ht (float, optional): The maximum wave height. Defaults to 0.5.

            Attributes:
                video_path (str): 
                    The path to the video file.
                
                num_stakes (int): 
                    The number of stakes.
               
                max_wave_ht (float): 
                    The maximum wave height.
                
                cap (cv2.VideoCapture): 
                    The video capture object.
                
                frame (numpy.ndarray): 
                    The first frame of the video.
               
                regions (list): 
                    The regions of interest: coordinates within
                    the original frame for the area where the 
                    float COULD wind up
                
                trackers (list[cv2.Tracker]): 
                    The object trackers.
                
                roi_slices_concat (dask.array.Array): 
                    The concatenated ROI slices.
                
                all_positions (dask.array.Array):  
                    The positions of all stakes.
                
                slices_list (list): 
                    The list of slices.
                
                chunk_bboxes (list): 
                    The bounding boxes of floats in the chunked thing 
                    (could we just do this from the orignal array and not have to do this?)
                
                ppm (numpy.ndarray): 
                    The pixels per meter array for each stake
            """
            print('beginning float tracker initialization')
            self.video_path = video_path
            self.num_stakes = num_stakes
            self.max_wave_ht = max_wave_ht
            self.cap = cv2.VideoCapture(video_path)
            self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame = self.cap.read()[1]
            self.regions : list = []
            self.trackers : list[cv2.Tracker]= []
            self.roi_slices_concat = da.zeros((0,), dtype=int)
            self.all_positions : da.Array = da.zeros((self.num_frames, self.num_stakes, 4), dtype=int)
            self.slices_list : list = []
            self.chunk_bboxes : list = []
            self.ppm = np.empty((num_stakes,))
            print('float tracker initialized')
            
    def load_video_to_da(self, chunk_size=100, print_every=25):
        '''
        Efficiently loads the video file into a Dask array using chunks.

        Args:
            chunk_size (int): Number of frames to read in each chunk.
            print_every (int): Interval for printing progress.

        Returns:
            da.Array: The video file in a Dask array.
        '''
        chunks_list = []
        total_frames_read = 0
        start_time = time.time()
        
        while True:
            frames_chunk = []
            for _ in range(chunk_size):
                ret, frame = self.cap.read()
                if not ret:
                    break
                frames_chunk.append(frame)
            
            if frames_chunk:
                frames_chunk_np = np.array(frames_chunk, dtype='float32')
                chunks_list.append(da.from_array(frames_chunk_np, chunks='auto'))
                total_frames_read += len(frames_chunk)
                
                if total_frames_read % print_every < chunk_size:
                    elapsed_time = time.time() - start_time
                    print(f'Loading frame {total_frames_read} of {self.num_frames}')
                    print(f'Time elapsed: {elapsed_time:.2f} seconds')
                    tqdm.update(len(frames_chunk))
            else:
                break
        
        if chunks_list:
            video_dask_array = da.concatenate(chunks_list, axis=0)
        else:
            video_dask_array = da.zeros((0,), dtype='float32')  # Fallback to an empty array if no frames were read
        
        self.dask_arr = video_dask_array
        return video_dask_array

    def derive_ppm(self):
        
        #user selects first frame
        all_points, all_lines = orth.define_stakes(self.frame,self.num_stakes)
        self.ppm = np.array(all_points)
        return self.ppm
    
    def load_chunk(self, chunk):
        '''
        here to leave room for different chunk loading methods

        Args:
            chunk (da.Array): chunk to load

        Returns:   
            da.Array: chunk (unloaded)
        '''
        return chunk
    
    def pad_chunk(self, chunk, max_rows, max_cols):
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
    
    def define_discontig_slice_from_dask(self, dask_arr : da.Array, slices_list: Iterable, num_stakes: int):
        '''
        Returns:
            np.ndarray: reference to chunk of dask array 
        '''
        delayed_chunks = []
        max_rows = max(slice[1].stop - slice[1].start for _, slice in slices_list)
        max_cols = max(slice[0].stop - slice[0].start for slice, _ in slices_list)

        for rows_slice, cols_slice in slices_list:
            delayed_chunk = delayed(self.load_chunk)(dask_arr[:, rows_slice, cols_slice, :])
            padded_chunk = delayed(self.pad_chunk)(delayed_chunk, max_rows, max_cols)
            delayed_chunks.append(padded_chunk)

        # concat the padded chunks
        self.roi_slices_concat = da.concatenate([da.from_delayed(chunk, shape=(None, max_rows, max_cols, 3), dtype=dask_arr.dtype) for chunk in delayed_chunks], axis=1)

        return self.roi_slices_concat
    
    def dask_tracker_init(self):
        '''
        Returns:
            list: The object trackers
        '''
        # Initialize the trackers
        if self.regions.empty() == True:
            for i in range(self.num_stakes):
                selected_region = cv2.selectROI(self.frame)
                self.regions.append(selected_region)
                self.trackers.append(cv2.TrackerCSRT_create(self.roi_slices_concat, selected_region))
        else: 
            for i in range(self.num_stakes):#TODO: derive selected region from self.regions and set it to self.chunk_bboxes
                self.chunk_bboxes.append(self.derive_bboxes_from_regions(self.regions, self.roi_slices_concat, self.num_stakes))
                self.trackers.append(cv2.TrackerCSRT_create(self.roi_slices_concat, self.chunk_bboxes[i]))

        return self.trackers

    def trackers_update(self, batch_size: int, frame_start: int, frame_interval : int):
        '''
        dask version of trackers_update, supports batched updating (load in x frames at once...)

        Args:
            trackers (list): The object trackers.
            frame (np.ndarray): The frame.

        Returns:
            list: The updated object trackers.
        '''
        
        #for each tracker in trackers 
        #we want to update it appropriately
        #get batch_size frames separated by frame_interval
        try:
            chunk = self.roi_slices_concat[frame_start:frame_start+batch_size*frame_interval:frame_interval].compute()
        except IndexError:
            print('IndexError -- You tried to index beyond the end of your array. \nLoading with defined interval until end of array!')
            chunk = self.roi_slices_concat[frame_start::frame_interval].compute()

        for i in range(chunk.shape[0]):
            for tracker in self.trackers:
                success, bbox = tracker.update(chunk[i])
                if success:
                    self.all_positions[frame_start+i*frame_interval] = bbox

        return self.all_positions
    
    def video_to_waveforms(self):
        '''
        Returns:
            np.ndarray: The waveforms
        '''
        #TODO
        pass
    
if __name__ == "__main__":
    dft = Dask_Float_Tracker('videos/5k_perp_salmon.MP4', 2, 0.5)
    dft.load_video_to_da()
    dft.derive_ppm()
    dft.define_discontig_slice_from_dask(dft.dask_arr, dft.slices_list, dft.num_stakes)
    dft.dask_tracker_init()
    dft.trackers_update(10, 0, 1)
    dft.video_to_waveforms()