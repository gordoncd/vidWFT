import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("noodle_float_move_rect.mp4")
object_detector = cv2.createBackgroundSubtractorMOG2()
ylist= []

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    height, width, _ = frame.shape

    # Define the colors of the rectangle
    lower_green = np.array([80, 150, 80])  # Light green lower bound in BGR
    upper_green = np.array([180, 255, 180])  # Light green upper bound in BGR

    mask = cv2.inRange(frame, lower_green, upper_green)
    roi = cv2.bitwise_and(frame, frame, mask=mask)
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame_ylist = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Rectangularity calculated but not used here, may be useful later so left in
        x, y, w, h = cv2.boundingRect(cnt)
        rectangularity = area / (w * h) if w * h > 0 else 0
        # Area must be sufficiently large to value object
        if area > 100:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            frame_ylist.append(y)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    if len(frame_ylist)>0:
        ylist.append(min(frame_ylist))
    cv2.imshow('image',frame)
    cv2.waitKey(0)

print(ylist)
fig = plt.figure()
plt.plot(ylist)
plt.draw()
plt.waitforbuttonpress(0)
plt.title('graph of rectangle y position')
plt.xlabel('frame')
plt.ylabel('pixel position')
fig.savefig('noodle.png')
plt.close('all')