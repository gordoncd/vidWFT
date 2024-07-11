'''
Gordon Doore 
camera-calibration-05-31-2024.py

scripts to obtain calibration data for camera
'''

import numpy as np
import cv2
import glob
import yaml #type: ignore
import os

def calibrate_camera(src : str, dest : str, base_filename : str = '', chessboard_size : tuple = (6,9), show = False, verbose = False):
    '''
    calibrate camera using chessboard images

    Args: 
        src (str): path to folder containing calibration images
        dest (str): path to folder to save camera matrices
        base_filename (str): prefix for saved files
        chessboard_size (tuple): size of chessboard used for calibration
        show (bool): whether to show undistorted images
    '''
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    num_sq = chessboard_size[0]*chessboard_size[1]
    objp = np.zeros((num_sq,3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

    # store points
    objpoints = [] # made up for the most part
    imgpoints = [] # points

    # list of image filepaths
    images = glob.glob(src+'/*.jpg') 

    for fname in images:
        img = cv2.imread(fname)
        #img = cv2.resize(img, (350,350))

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find corners
        ret, corners = cv2.findChessboardCorners(gray, (chessboard_size[0],chessboard_size[1]), flags = cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        #add points to image
        if ret:
            objpoints.append(objp)

            # draw corners
            img = cv2.drawChessboardCorners(img, (chessboard_size[0],chessboard_size[1]), corners, ret)
            imgpoints.append(corners)
            if show:
                cv2.imshow('img', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else: 
            print('Could not find corners in image: ',fname)

    # calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) #type: ignore

    if verbose: 
        print("Camera matrix : \n")
        print(mtx)
        print("dist : \n")
        print(dist)
        print("rvecs : \n")
        print(rvecs)
        print("tvecs : \n")
        print(tvecs)

    if show:
        # Undistort image
        undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

        # Display undistorted image
        cv2.imshow('Undistorted Image', undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #save camera matrices to yaml file: 
    data = {
        'camera_matrix': mtx.tolist(),
        'dist_coeff': dist.tolist(),
        'rvecs': [rvec.tolist() for rvec in rvecs],
        'tvecs': [tvec.tolist() for tvec in tvecs]
    }
    #make dir if it does not exist
    if not os.path.exists(dest+'/' + base_filename+'vid_camera_matrices.yaml'):
        if verbose: 
            print('could not find file: ',dest+'/' + base_filename+'vid_camera_matrices.yaml'+'\ncreating directory')
        os.makedirs(dest)
    with open(dest+'/' + base_filename+'vid_camera_matrices.yaml', 'w') as file:
        yaml.dump(data, file)

    #also save as numpy arrays: 

    np.save(dest+'/' + base_filename + 'camera_matrix.npy', mtx)
    np.save(dest+'/' + base_filename + 'dist_coeff.npy', dist)
    np.save(dest+'/' + base_filename + 'rvecs.npy', rvecs)
    np.save(dest+'/' + base_filename + 'tvecs.npy', tvecs)

def load_camera_calibration_data(matrix_path : str, distance_coefficient_path : str) -> tuple:
    '''
    load camera calibration data from saved files

    Args: 
        matrix_path (str): path to camera matrix file
        distance_coefficient_path (str): path to distance coefficient file

    Returns:
        tuple: camera matrix and distance coefficients
    '''
    mtx = np.load(matrix_path)
    dist = np.load(distance_coefficient_path)
    return mtx, dist

def extract_calibration_frames(filepath : str, nframes : int) ->list[np.ndarray]:
    '''
    randomly grab nframes frames from the mp4 at
    filepath and return them as a list of numpy arrays 
    
    Args: 
        filepath (str): path to video file
        nframes (int): number of frames to extract
    '''
    #randomly grab nframes frames from the mp4 at
    # filepath and return them as a list of numpy arrays 
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.random.choice(total_frames, nframes, replace=False)
    frames = []
    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


if __name__ == '__main__':
    calibrate_camera('calibration/acortiz@colbydotedu_CALIB/calib_frames_5k','calibration/acortiz@colbydotedu_CALIB/hmm','5k_', (7,9))

