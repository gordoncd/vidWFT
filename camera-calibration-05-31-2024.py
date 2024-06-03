import numpy as np
import cv2
import glob
import yaml 

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((70,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:10].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# List of images
images = glob.glob('/Users/gordondoore/Documents/GitHub/waves-summer-2024/calib_frames/*.jpg') 
for fname in images:
    img = cv2.imread(fname)
    #img = cv2.resize(img, (350,350))

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,10), flags = cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        print(objp)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,10), corners, ret)
        imgpoints.append(corners)
        cv2.imshow('img', img)
        cv2.waitKey(0)

cv2.destroyAllWindows()

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

# Undistort image
undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

# Display undistorted image
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#save camera matrices to yml file: 
# Save camera matrices to .yaml file
data = {
    'camera_matrix': mtx.tolist(),
    'dist_coeff': dist.tolist(),
    'rvecs': [rvec.tolist() for rvec in rvecs],
    'tvecs': [tvec.tolist() for tvec in tvecs]
}

with open('/Users/gordondoore/Documents/GitHub/waves-summer-2024/gopro_1080_vid_camera_matrices.yaml', 'w') as file:
    yaml.dump(data, file)

#also save as numpy arrays: 

np.save('/Users/gordondoore/Documents/GitHub/waves-summer-2024/camera_matrix.npy', mtx)
np.save('/Users/gordondoore/Documents/GitHub/waves-summer-2024/dist_coeff.npy', dist)
np.save('/Users/gordondoore/Documents/GitHub/waves-summer-2024/rvecs.npy', rvecs)
np.save('/Users/gordondoore/Documents/GitHub/waves-summer-2024/tvecs.npy', tvecs)