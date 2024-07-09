'''
Gordon Doore 
07/09/2024
tests for Dask_Video_Loader class
'''
import pytest
import numpy as np
import cv2
import dask.array as da
from video_load_dask import Dask_Float_Tracker

def test_Dask_Float_Tracker(video_path):
    '''
    Tests the Dask_Float_Tracker class
    '''
    # Initialize the Dask_Float_Tracker object
    dft = Dask_Float_Tracker(video_path, 2, 0.5)

    
    # test pad_chunk:
    
    test_pad_chunk()
    # test load_chunk:


def test_load_video_to_da(dft):
    '''
    Tests the load_video_to_da method of the Dask_Float_Tracker class
    '''
    dft.load_video_to_da()
    assert dft.dask_arr != None
    assert type(dft.dask_arr) == da.Array


def test_pad_chunk(dft):
    '''
    Tests the pad_chunk method of the Dask_Float_Tracker class
    '''

    #create a test array:
    test_arr = da.arange(1000).reshape(10,10,10,1)
    chunk = test_arr[0:5].compute()
    #pad the chunk
    padded_chunk = dft.pad_chunk(chunk)
    #check that the padded chunk has the correct shape
    assert padded_chunk.shape == (10, 10, 10, 3)

    #load in some portion of the video


