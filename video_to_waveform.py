'''
THIS FILE IS DATED AND CONTAINS NO RELEVANT OR CURRENTLY USED CODE.
Gordon Doore
Convert video of ____ experimental setup into waveforms representing wave characteristics
06/10/2024
'''
import cv2
import numpy as np
import datetime
from matplotlib import pyplot as plt

def find_centers(contours):
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Check if the contour is not empty
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            centers.append([center_x, center_y])
    return centers


def video_to_waveform(rectified_video, graph_dest, arr_dest,ppm, color_low, color_high, candidate_function, n_stakes):
    '''
    input: 
    rectified_video: filepath to rectified video (first two frames must have unobstructed view of all floats and stakes)
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
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #contours are our 'found objects'
        #iterate through each contour: can vectorize?
        scores = np.zeros((len(contours),))
        for i, cont in enumerate(contours): 
            area = cv2.contourArea(cont)
            if area>50:
                # Assign each a score using the candidate function
                score = candidate_function(cont, hsv, color_low, color_high)
                scores[i] = score
        #find the best scores by getting the top n_stakes scores
        
        ind = np.argpartition(scores, -n_stakes)[-n_stakes:]
        #now get those contours:
        floats = np.array(contours, dtype = object)[ind]
        # Call the function in the main code
        positions.append(find_centers(floats))
        print('frame complete')
    #now positions should be filled out:
    #convert the positions to an array for easier everything
    positions = np.array(positions)
    #order positions from left to right
    positions = positions[np.argsort(positions[:, :, 0])]#might not be right
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
        plt.savefig(graph_dest + '/waveform_float_{}'+current_time+ '.svg'.format(i+1))
        plt.close()
    
def candidate_score(contour, hsv, light_green_low, light_green_high):
    # Create a black image to draw the contour on
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    mask = cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Calculate the color score as the percentage of pixels in the contour that are light green
    contour_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    light_green_pixels = cv2.inRange(contour_hsv, light_green_low, light_green_high)

    color_score = cv2.countNonZero(light_green_pixels) / float(cv2.countNonZero(mask))
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Calculate the shape score as the percentage of vertices that form a rectangle
    shape_score = len(approx) / 4 if len(approx) <= 4 else 0
    print(shape_score)

    # Combine the color and shape scores
    score = color_score * shape_score

    return score

if __name__ == "__main__":

    # Define lower and upper bounds for "light green" in HSV
    light_green_low = np.array([35, 50, 70], dtype=np.uint8)
    light_green_high = np.array([85, 255, 250], dtype=np.uint8)

    video_to_waveform('noodle_float_move_rect.mp4','','',0.2,light_green_low,light_green_high,candidate_score, 2)