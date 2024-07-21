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
import tqdm

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

def undistort_video(filepath : str, mtx : np.ndarray, dist : np.ndarray, save_path : str, show = False):
    '''
    undistort video using camera matrix and distance coefficients

    Args: 
        filepath (str): path to video file
        mtx (np.ndarray): camera matrix
        dist (np.ndarray): distance coefficients
        save_path (str): path to save undistorted video
        show (bool): whether to show undistorted video
    '''
    cap = cv2.VideoCapture(filepath)
    
    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create the VideoWriter object
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    pbar = tqdm.tqdm(total=total_frames)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Undistort the frame
            dst = cv2.undistort(frame, mtx, dist, None)
            
            # Write the undistorted frame to the output video
            out.write(dst)
            
            if show:
                cv2.imshow('frame', dst)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            pbar.update(1)
        else:
            break
    pbar.close()
    
    # Release the resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def crop_and_undistort(video_path, matrix_path, dist_path, crop_region, output_path):
    # Load camera calibration data
    camera_matrix = np.load(matrix_path)
    print(camera_matrix)
    dist_coeffs = np.load(dist_path)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the crop region (x, y, width, height)
    x, y, w, h = crop_region

    # Define the codec and create VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))

    camera_matrix, dist_coeffs = adjust_calibration_matrices(camera_matrix, dist_coeffs, crop_region, W, H)
    print(camera_matrix)
    pbar = tqdm.tqdm(total=total_frames)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame
        cropped_frame = frame[y:y+h, x:x+w]

        # Undistort the cropped frame
        undistorted_frame = cv2.undistort(cropped_frame, camera_matrix, dist_coeffs, None)
        # Write the frame
        out.write(undistorted_frame)

        pbar.update(1)
    pbar.close()

    # Release everything if job is finished
    cap.release()
    out.release()

def adjust_calibration_matrices(camera_matrix, dist_coeffs, crop_region, W, H):
    '''
    adjust camera matrix and distance coefficients to account for cropping

    Args: 
        camera_matrix (np.ndarray): camera matrix
        distances (np.ndarray): distance coefficients
    '''
    x, y, w, h = crop_region

    # # Get focal lengths and principal point from camera matrix
    # f_x = camera_matrix[0, 0]
    # f_y = camera_matrix[1, 1]
    # c_x = camera_matrix[0, 2]
    # c_y = camera_matrix[1, 2]

    # # Compute new principal point for the cropped region
    # c_x_crop = c_x - x
    # c_y_crop = c_y - y

    # # Compute new focal lengths (adjusted for cropping)
    # f_x_crop = f_x * (w / camera_matrix[0, 2])
    # f_y_crop = f_y * (h / camera_matrix[1, 2])

    # # New camera matrix for the cropped region
    # K_crop = np.array([[f_x_crop, 0, c_x_crop],
    #                 [0, f_y_crop, c_y_crop],
    #                 [0, 0, 1]])

    # return K_crop, dist_coeffs
        # Unpack the crop region
    x, y, w, h = crop_region

    # Copy the original camera matrix to avoid modifying it directly
    adjusted_camera_matrix = camera_matrix.copy()

    # Adjust the principal point based on the crop
    adjusted_camera_matrix[0, 2] -= x  # cx adjusted
    adjusted_camera_matrix[1, 2] -= y  # cy adjusted

    # No adjustment needed for distortion coefficients in most cases
    # If needed, this would be the place to do it

    return adjusted_camera_matrix, dist_coeffs

def crop_video(src, dest, crop_region):
    '''
    
    '''
    cap = cv2.VideoCapture(src)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the crop region (x, y, width, height)
    x, y, w, h = crop_region

    # Define the codec and create VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(dest, fourcc, 30.0, (w, h))

    pbar = tqdm.tqdm(total=total_frames)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame
        cropped_frame = frame[y:y+h, x:x+w]

        # Check if the cropped frame is not empty
        if cropped_frame.size != 0:
            # Write the frame
            out.write(cropped_frame)

        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()



if __name__ == '__main__':
    import cv2
    #calibrate_camera('calibration/acortiz@colbydotedu_CALIB/calib_frames_5k','calibration/acortiz@colbydotedu_CALIB/hmm','5k_', (7,9))
   # Example usage
    video_path = 'videos/5k_perp_salmon.MP4'
    matrix_path = 'calibration/acortiz@colbydotedu_CALIB/camera_matrix_5k.npy'
    dist_path = 'calibration/acortiz@colbydotedu_CALIB/dist_coeff_5k.npy'

    #get first frame from video_path
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    crop_region = cv2.selectROI(frame)
    output_path = 'output_data/cropped_and_undistorted.mp4'

    crop_and_undistort(video_path, matrix_path, dist_path, crop_region, output_path)
