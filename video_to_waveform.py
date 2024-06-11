'''
Gordon Doore
Convert video of ____ experimental setup into waveforms representing wave characteristics
06/10/2024
'''
import cv2
import numpy as np
import datetime
from matplotlib import pyplot as plt

def find_centers(floats):
    centers = []
    for float_contour in floats:
        # Calculate the moments of the contour
        M = cv2.moments(float_contour)
        # Calculate the center of the contour
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        center = (center_x, center_y)
        centers.append(center)
    return np.array(centers)


def video_to_waveform(rectified_video, graph_dest, arr_dest,ppm, color_low, color_high, roi, candidate_function, n_stakes):
    '''
    input: 
    rectified_video: filepath to rectified video (first frame must have unobstructed view of all floats and stakes)
    num_stakes: number of stakes in the video
    graph_dest: filepath to write images representing each wave
    arr_dest: filepath to write array representing each wave

    '''
    #load in video
    cap = cv2.VideoCapture(rectified_video)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error reading the first frame")
        return
    #might need to switch this detector with something else eventually
    object_detector = cv2.createBackgroundSubtractorMOG2()
    positions = []
    #for each frame, record the vertical position of some constant part of each float
    while True:

        #read the next frame
        ret, frame = cap.read()
        if frame is None:
            #if the frame is empty (we reached the end) break
            break
        
        height, width, _ = frame.shape
        #we use hsv so we can work with hue and saturation (very helpful)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #create mask of all 'orange enough' objects
        mask = cv2.inRange(hsv, color_low, color_high)
        
        #look for num_stakes floats
        #now we make object detection of the roi of the mask
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #contours are our 'found objects'
        contours = np.array(contours)
        #iterate through each contour: can vectorize?
        scores = np.zeros((contours.shape[0],))
        for i, cont in enumerate(contours): 
            #assign each a score using the candidate function
            score = candidate_function(cont)
            scores[i] = score
            #put positions into relevent arrays for storing the positions of each float
        #find the best scores by getting the top n_stakes scores
        ind = np.argpartition(scores, -n_stakes)[-n_stakes:]
        #now get those contours:
        floats = contours[ind]
        # Call the function in the main code
        positions.append(find_centers(floats))
    #now positions should be filled out:
    #convert the positions to an array for easier everything
    positions = np.array(positions)
    #we only care about offset for for each set of positions, we subtract the mean
    centered_positions = positions-np.mean(positions, axis = 1)
    #using the ppm, convert from pixel space to real space
    real_movements = centered_positions/ppm

    #save arrays 
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = 'float_positions_' + current_time
    np.save(arr_dest+'/'+filename+'.npz', real_movements)

    #make and save figure of the waveform for each float
    for i in range(real_movements.shape[1]):
        plt.figure()
        plt.plot(real_movements[:, i])
        plt.xlabel('Frame')
        plt.ylabel('Position')
        plt.title('Waveform for Float {}'.format(i+1))
        plt.savefig(graph_dest + '/waveform_float_{}.png'.format(i+1))
        plt.close()
    


