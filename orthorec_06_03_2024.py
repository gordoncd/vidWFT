'''
Gordon Doore 
06/03/2024
orthorec-06-03-2024.py

functions to rectify image based on input quadrangle

Last Modified: 06/03/2024

'''


import cv2
import numpy as np
import matplotlib.pyplot as plt

def pick_points(img):
    chosen_points = []
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def onclick(event):
        if event.inaxes is not None:
            chosen_points.append((event.xdata, event.ydata))
            print(f"Point picked: ({event.xdata},{event.ydata})")
            ax.plot(event.xdata, event.ydata, marker='o', color='red', markersize=5)
            fig.canvas.draw()
            if len(chosen_points) == 4:
                print("All points picked.")
                plt.close()  # Close the figure once all points are picked

    fig, ax = plt.subplots()
    ax.imshow(rgb_img)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    print("points finalized")
    return chosen_points

def rectify(img, inpoints, outpoints):
    '''
    img: input image
    inpoints: original points
    outpoints: points to map to
    dst: rectified image
    '''
    out = np.copy(img)
    matrix = cv2.getPerspectiveTransform(inpoints, outpoints)
    result = cv2.warpPerspective(out, matrix, (img.shape[1], img.shape[0]))

    return result


def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


if __name__ == '__main__':
    #get input image
    img = cv2.imread('orthotest_frames/orthotest_frame_8.jpg')

    old_points = np.array(pick_points(img), dtype=np.float32)
    old_points = order_points(old_points)  

    new_points = np.array([[0,0],[old_points[-1,0],0],[0,old_points[-1,1]],old_points[-1]], dtype=np.float32)
    rectified = rectify(img, old_points, new_points)
    cv2.imshow('Rectified Image', rectified)
    cv2.waitKey(0)