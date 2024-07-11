import os
import numpy as np 
import sys
import cv2

# Add the parent directory of the tests directory to sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now you can import the tracker module
from vidWFT.core.tracker import tracker_init, trackers_update

#put tests here


