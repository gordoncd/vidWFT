'''
Gordon Doore 
06/03/2024
orthorec-06-03-2024.py

functions to rectify image

Last Modified: 06/07/2024

'''

import skimage
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

def two_largest(column):
    indices = set()
    first =  set()
    second = set()
    prev = None
    indices.add(column[0])
    for index in column:   
        if prev ==None or len(indices) == 1:
            prev = index
            indices.add(index)
            continue
        if index == prev+1:
            indices.add(index)
        else: 
            if len(indices)>len(first): 
                proxy = first
                first = indices
                if len(proxy)>len(second):
                    second = proxy
            elif len(indices) > len(second):
                second = indices

            indices = set()
        prev = index 
        indices.add(index)
    if column[-1] in indices:
        #do the comparison again
        if len(indices)>len(first): 
            proxy = first
            first = indices
            if len(proxy)>len(second):
                second = proxy
        elif len(indices) > len(second):
            second = indices
    return first, second

def find_difference_gradations(gradation_pix):
    distances = []
    for column in gradation_pix:
        first,second = two_largest(column)
        center1 = sum(first)/len(first)
        center2 = sum(second)/len(second)
        distance = abs(center1-center2)
        distances.append(distance)
        
    return distances

def find_gradations(img, lines, threshold_condition):
    #lines is the lines produced when selecting one vertical slice of each stake
    #returns points of the center of our gradations
    all_stake_points = []
    for line in lines:
        #get the pixel value of the pixels defined by lines
        pixels = img[line]
        #threshold those values
        gradation_idx = np.argwhere(threshold_condition(pixels))
        #find the midpoint of the two biggest groups
        first, second = two_largest(gradation_idx)
        #first and second are sets, so we get their average value and just get the int of that up to half a pixel of error introduced here
        mid1 = int(sum(first)/len(first))
        mid2 = int(sum(second)/len(second))

        stake_points = np.array([[line[mid1],line[mid2]]])
        all_stake_points.append(stake_points)
    #now return the indices of lines based on the index from we just got
    return np.array(all_stake_points)

def get_ppm(img, points, pixel_columns, gradation_size, stake_thresh, stake_grad_thresh, peaks_sampled):
    '''
    get ppm of different stakes, assumes it is relatively similar for the stake
    ie that the camera is sufficiently far away that there is insignificant perspective warping
    in the vertical direction and that stake is vertically aligned (parallel to the y axis)
    '''
    gradation_pix = [np.argwhere(column < stake_grad_thresh) for column in pixel_columns]

    distances = np.array(find_difference_gradations(gradation_pix))

    #estimate ppm with distances: 
    ppm_stake = distances/gradation_size
    
    return ppm_stake

def linear_transform(points):
    #returns linear function for perspective warping between two stakes with gradations at points
    #points: ndarray shape: (N_stakes, 2,2) 
    #points[:,0] are the first gradation for a given stake
    #points[:,1] are the second gradation for a given stake
    
    # Calculate the slope and intercept for each stake
    slopes = (points[:, 1, 1] - points[:, 0, 1]) / (points[:, 1, 0] - points[:, 0, 0])
    intercepts = points[:, 0, 1] - slopes * points[:, 0, 0]
    
    return slopes, intercepts



def rectify_by_gradation(img,n_stakes, stake_thresh, stake_grad_thresh, threshold_condition):
    '''
    find gradated stakes and rectify image by size variation

    have user draw lines on the first frame to find each stake

    get column of pixel values, grayscale them

    then compute number of pixels separating stake

    use distances to rectify image assuming that gradations are equally 
    spaced in real space. 
    '''
    stake_columns = []
    #get points and lines
    all_points, all_lines = define_stakes(img,n_stakes)

    img_with_lines = np.copy(img)
    #get pixel values of the lines and add them to stake_columns
    for stake in range(n_stakes):
        rr = all_lines[stake][1]
        cc = all_lines[stake][0]
        stake_line = img[rr,cc]
        stake_columns.append(stake_line)
        # Draw the line on the image
        img_with_lines = cv2.line(img_with_lines, (cc[0], rr[0]), (cc[-1], rr[-1]), (255, 0, 0), 2)

    # Display the image with the lines
    cv2.imshow('Image with Lines', img_with_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    
    #find the points of the gradations: 
    gradation_points = find_gradations(img,all_lines,threshold_condition)

    #get linear functions for transformation: 
    slopes, intercepts = linear_transform(gradation_points)

    #get linear transformation matrix for perspective shift: 
    matrix = np.zeros((3, 3), dtype=np.float32)
    matrix[0, 0] = slopes[0]
    matrix[0, 2] = intercepts[0]
    matrix[1, 1] = slopes[1]
    matrix[1, 2] = intercepts[1]
    matrix[2, 2] = 1.0

    # Apply perspective transformation to rectify image
    rectified = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))

    return rectified

def define_stakes(img, n_stakes):
    '''
    user draws lines on images for n_stakes
    returns pixel columns and stake coordinates

    assume stakes are straight lines,
    use only 2 points per stake
    '''
    def onclick(event):
        if event.inaxes is not None:
            chosen_points.append((event.xdata, event.ydata))
            print(f"Point picked: ({event.xdata},{event.ydata})")
            # Assuming 'img' is your image and 'event.xdata' and 'event.ydata' are the coordinates
            rgb_img[int(event.ydata), int(event.xdata)] = [255, 0, 0]  # Change the pixel to red
            if len(chosen_points) == 2:
                print("All points for this stake picked.")
                plt.close()  # Close the figure once all points are picked


    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    all_points=[]
    all_lines = []
    for n in range(n_stakes):
        print("Stake number: "+str(n))
        chosen_points = []
        fig, ax = plt.subplots()
        ax.imshow(rgb_img)
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        # Use skimage to get the points on the line
        line_points = skimage.draw.line(int(chosen_points[0][0]), int(chosen_points[0][1]), int(chosen_points[1][0]), int(chosen_points[1][1]))
        all_points.append(chosen_points)
        all_lines.append(line_points)
                

    return all_points, all_lines


def rectify_video(input_video_path, output_video_path):
    '''USED COPILOT WITH PROMPT: I want a function which uses the following 
    code to first pick points based on the first frame of some input video.
    Next, we will use those points project onto ALL frames. 
    The function will return the new projceted video and save it 
    locally
    '''
    # Load the video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error reading the first frame")
        return

    # Pick points from the first frame
    old_points = np.array(pick_points(first_frame), dtype=np.float32)
    old_points = order_points(old_points)

    # Define the new points
    new_points = np.array([[0,0],[old_points[-1,0],0],[0,old_points[-1,1]],old_points[-1]], dtype=np.float32)

    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    # Rectify and write each frame
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                rectified = rectify(frame, old_points, new_points)
                # Resize the rectified frame to match the original frame size
                rectified_resized = cv2.resize(rectified, (frame_width, frame_height))
                frames.append(rectified_resized)
                cv2.imshow('Rectified Frame', rectified_resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    finally:
        # Release the VideoCapture and VideoWriter objects
        cap.release()
        cv2.destroyAllWindows()

    #now we save frames as an mp4:
    # Create a VideoWriter object
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    # Write the frames to the video file
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()

if __name__ == '__main__':
    #get input image
    #img = cv2.imread('orthotest_frames/orthotest_frame_8.jpg')

    # old_points = np.array(pick_points(img), dtype=np.float32)
    # old_points = order_points(old_points)  

    # new_points = np.array([[0,0],[old_points[-1,0],0],[0,old_points[-1,1]],old_points[-1]], dtype=np.float32)
    # rectified = rectify(img, old_points, new_points)
    # cv2.imshow('Rectified Image', rectified)
    # cv2.waitKey(0)

    #rectify_video('pole_movement_1.MP4', 'rectified.mp4')
    #points, lines = define_stakes(img, 3)

    # Load the video
    cap = cv2.VideoCapture('gp4k_stakes.MP4')
    # Get the first frame
    ret, first_frame = cap.read()

    #example columns 
    col1 = np.array([2,3,4,5,6,7,8,9,30,31,51,52,53,54,55])
    col2 = np.array([6,7,8,9,20,21,38,39,40,41,42,43])
    col3 = np.array([7,8,9,10,11,12,13,14,32,33,34,35,61,62,63,64,65,66,67,68])

    print(find_difference_gradations([col1,col2,col3]))

    