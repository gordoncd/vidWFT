'''
Gordon Doore
Summer 2024 Waves
05-28-2024
waves-analysis-05-28-2024.py
__________________
This code tracks a ball, specifically tuned for pingpong.mp4 on the project github
It generates a graph of the y position of the ball through each frame that it is detected
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

#load in video
cap = cv2.VideoCapture("pingpong.mp4")
image_name = 'pingpong_testgraph.png'
#this detector works with a stable camera
object_detector = cv2.createBackgroundSubtractorMOG2()
#initiate list of y coords so we can look more at that later
ylist= []
#infinite loop (but will always break)
while True:
    #read the next frame
    ret, frame = cap.read()
    if frame is None:
        #if the frame is empty (we reached the end) break
        break
    #frame characteristics
    height, width, _ = frame.shape
    #we use hsv so we can work with hue and saturation (very helpful)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #define the colors of the ball (which should sufficiently contrast with environment)
    lower_orange = np.array([5, 100, 100]) 
    upper_orange = np.array([15, 255, 255])

    #create mask of all 'orange enough' objects
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    #doesn't do anything here, but if we want to 
    #work only with a small part of an image, this is where we can specify
    roi = cv2.bitwise_and(frame, frame, mask=mask)

    #now we make object detection of the roi of the mask
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY) #type: ignore
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #type: ignore
    #contours are our 'found objects'
    frame_ylist : list[int]= []
    for cnt in contours:
        #for each contour we determine if its our ball

        area = cv2.contourArea(cnt)
        #circularity calculated but not used here, may be useful later so left in
        circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2) if cv2.arcLength(cnt, True) > 0 else 0
        #area must be sufficiently large to value object
        if area > 500:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            frame_ylist.append(y)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    if len(frame_ylist)>0:
            #find the highest y value to avoid shadow objects
        ylist.append(min(frame_ylist))
    #image show code
    cv2.imshow('image',frame)
    cv2.waitKey(0)

#generates graph of y position, will be useful in actual project
print(ylist)
fig = plt.figure()
plt.plot(ylist)
plt.draw()
plt.waitforbuttonpress(0)
plt.title('graph of ball y position')
plt.xlabel('frame')
plt.ylabel('pixel position')
fig.savefig('output_figures/'+image_name)
plt.close('all')