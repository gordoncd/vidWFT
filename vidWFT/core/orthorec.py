'''
Gordon Doore 
06/03/2024
orthorec-06-03-2024.py

functions to rectify image

Last Modified: 06/13/2024

'''

import skimage #type: ignore
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from collections.abc import Iterable
from typing import Any, Callable, Tuple

def pick_points(img : np.ndarray) -> list[Any]:
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

def rectify(img : np.ndarray, inpoints : np.ndarray, outpoints : np.ndarray) -> np.ndarray:
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


def order_points(pts : np.ndarray) -> np.ndarray:
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

def two_largest(arr : np.ndarray) -> Tuple[list[Any], list[Any]]:
    if arr is None:
        return [], []
    print(arr)
    first : list = []
    second : list = []
    current_subsequence = [arr[0]]

    for i in range(1, len(arr)):
        # Check if current element continues the subsequence
        if arr[i] == arr[i-1] + 1:
            current_subsequence.append(arr[i])
        else:
            # Compare and possibly update first and second largest subsequences
            if len(current_subsequence) > len(first):
                first, second = current_subsequence, first
            elif len(current_subsequence) > len(second):
                second = current_subsequence
            current_subsequence = [arr[i]]

    # Check the last subsequence
    if len(current_subsequence) > len(first):
        first, second = current_subsequence, first
    elif len(current_subsequence) > len(second):
        second = current_subsequence

    return first, second

def find_difference_gradations(gradation_pix : Iterable[np.ndarray]):
    distances = []
    for column in gradation_pix:
        first,second = two_largest(column)
        center1 = sum(first)/len(first)
        center2 = sum(second)/len(second)
        distance = abs(center1-center2)
        distances.append(distance)
        
    return distances

def find_gradations(img : np.ndarray, lines : list[np.ndarray], threshold_condition : Callable[[np.ndarray], np.ndarray]):
    #lines is the lines produced when selecting one vertical slice of each stake
    #returns points of the center of our gradations
    all_stake_points = []

    #pair parts of lines
    i=0
    for line in lines:
        print('line ')
        #get the pixel value of the pixels defined by lines
        pixels = img[line[1],line[0]]
        #threshold those values
        gradation_idx = np.argwhere(threshold_condition(pixels))
        #find the midpoint of the two biggest groups
        gradation_idx = np.squeeze(gradation_idx)
        first, second = two_largest(gradation_idx)
        print(first,second)
        #first and second are sets, so we get their average value and just get the int of that up to half a pixel of error introduced here
        mid1 = int(sum(first)/len(first))
        mid2 = int(sum(second)/len(second))

        stake_points = np.squeeze(np.array([[line[1][mid1],line[0][mid1]],[line[1][mid2],line[0][mid2]]]))
        i+=1
        all_stake_points.append(stake_points)
    #now return the indices of lines based on the index from we just got
    return np.array(all_stake_points)

def get_ppm(img : np.ndarray, points : np.ndarray, pixel_columns : Iterable[np.ndarray], gradation_size : np.ndarray, stake_thresh : int, stake_grad_thresh : int, peaks_sampled : int) -> np.ndarray:
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

def linear_transform(points : np.ndarray):
    '''
    returns linear function for perspective warping between two stakes with gradations at points
    points: ndarray shape: (N_stakes, 2,2) 
    points[:,0] are the first gradation for a given stake
    points[:,1] are the second gradation for a given stake
    '''
    
    # Calculate the slope and intercept for each stake
    slopes = (points[:, 1, 1] - points[:, 0, 1]) / (points[:, 1, 0] - points[:, 0, 0])
    intercepts = points[:, 0, 1] - slopes * points[:, 0, 0]
    
    return slopes, intercepts



def rectify_by_gradation(img : np.ndarray ,n_stakes : int, stake_thresh : int, stake_grad_thresh : int, threshold_condition : Callable[[np.ndarray], np.ndarray], load_prev_grad : np.ndarray = np.zeros((4,))) -> Tuple[np.ndarray, np.ndarray]:
    '''
    find gradated stakes and rectify image by size variation

    have user draw lines on the first frame to find each stake

    get column of pixel values, grayscale them

    then compute number of pixels separating stake

    use distances to rectify image assuming that gradations are equally 
    spaced in real space. 
    '''
    #get points and lines
    if np.sum(load_prev_grad) == 0:
        all_points, all_lines = define_stakes(img,n_stakes)
        #find the points of the gradations: 
        gradation_points = find_gradations(img,all_lines,threshold_condition)
        gradation_points = np.squeeze(gradation_points)
        old_points = np.flip(gradation_points, axis = 2)
        old_points = old_points.reshape((old_points.shape[0]*old_points.shape[1],old_points.shape[2]))
        old_points = order_points(old_points)
        print(old_points, np.array([old_points[0], [old_points[1,0], old_points[0,1]], [old_points[0,0], old_points[2,1]], [old_points[1,0], old_points[2,1]]], dtype=np.float32))
    else:
        old_points = load_prev_grad

    # Define new points as a rectangle
    new_points = np.array([old_points[0], [old_points[1,0], old_points[0,1]], [old_points[0,0], old_points[2,1]], [old_points[1,0], old_points[2,1]]], dtype=np.float32)
    #we use the gradation points as the input points
    #now we need the destination poirnts
    rectified = rectify(img, old_points.astype(np.float32), new_points.astype(np.float32))
    return rectified, old_points

def define_stakes(img : np.ndarray, n_stakes : int) -> Tuple[list[list[Any]], list[Any]]:
    """
    user draws lines on images for n_stakes
    returns pixel columns and stake coordinates

    assume stakes are straight lines,
    use only 2 points per stake

    Args: 
        img: ndarray
            numPy array represetning the image to define stakes on,
            usually a selected frame from video
        n_stakes: int
            number of stakes to 

    Returns: 
        all_points: list
            coordinates for each stake
            in the form [[[s1x1,s1y1],[s1x2,s1y2]],...]
        all_lines: list of ndarray
            line computed using skimage.draw.line over the points in all_points
    """
    def onclick(event):
        # Check if the toolbar's current tool is 'zoom' or 'pan'
        current_tool = plt.get_current_fig_manager().toolbar.mode
        if event.inaxes is not None and current_tool == '':
            chosen_points.append((event.xdata, event.ydata))
            print(f"Point picked: ({event.xdata},{event.ydata})")
            ax.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()
            plt.show()
            if len(chosen_points) == 2:
                print("All points for this stake picked.")
                plt.close()  # Close the figure once all points are picked

    # Convert the image to RGB for display
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    all_points = []
    all_lines = []

    for n in range(n_stakes):
        print("Stake number:", n)
        chosen_points : list[Tuple[float,float]] = []
        fig, ax = plt.subplots()
        ax.imshow(rgb_img)
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.title("Zoom in as needed and select 2 points. Close the window when done.")
        plt.show()

        if len(chosen_points) == 2:
            # Use skimage to get the points on the line between the selected points
            line_points = skimage.draw.line(int(chosen_points[0][1]), int(chosen_points[0][0]),
                                            int(chosen_points[1][1]), int(chosen_points[1][0]))
            all_points.append(chosen_points)
            all_lines.append(line_points)

    return all_points, all_lines


def rectify_video(input_video_path : str, output_video_path : str) -> None:
    '''USED COPILOT WITH PROMPT: I want a function which uses the following 
    code to first pick points based on the first frame of some input video.
    Next, we will use those points project onto ALL frames. 
    The function will return the new projceted video and save it 
    locally
    
    Rectify a video by gradations

    Args:
        input_video_path: string
            The path to the unrectified video
        output_video_path: string
            The path to where the rectified video should go

    
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
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height)) # type: ignore

    # Write the frames to the video file
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()



def rectify_video_by_gradation(input_video_path : str, output_video_path : str, threshold_condition : Callable[[np.ndarray], np.ndarray], show : bool = False) -> None:
    '''
    Rectify a video by gradations

    Args:
        input_video_path (str): The path to the unrectified video
        output_video_path (str): The path to where the rectified video should go
        threshold_condition (func): function whcih returns binary or boolean array of the same size as its input
        show (bool): whether or not to show the rectified video during computation

    Returns: 
        None
w
    '''
    # Load the video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get the first frame
    ret, first_frame = cap.read()
    rectified, gradation_pts = rectify_by_gradation(first_frame,2,100,100,threshold_condition)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    if not ret:
        print("Error reading the first frame")
        return

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
                rectified, old_points = rectify_by_gradation(frame,2,100,100, threshold_condition, load_prev_grad = gradation_pts)
                if rectified.shape != (frame_width,frame_height,3):
                    # Resize the rectified frame to match the original frame size
                    rectified = cv2.resize(rectified, (frame_width, frame_height))
                frames.append(rectified)
                if show:
                    cv2.imshow('Rectified Frame', rectified)
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
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))#type: ignore

    # Write the frames to the video file
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()
    return

if __name__ == '__main__':
    #get input image
    #img = cv2.imread('orthotest_frames/orthotest_frame_8.jpg')

    # old_points = np.array(pick_points(img), dtype=np.float32)
    # old_points = order_points(old_points)  

    # new_points = np.array([[0,0],[old_points[-1,0],0],[0,old_points[-1,1]],old_points[-1]], dtype=np.float32)
    # rectified = rectify(img, old_points, new_points)
    # cv2.imshow('Rectified Image', rectified)
    # cv2.waitKey(0)
    #points, lines = define_stakes(img, 3)

    threshold_condition = lambda x: np.sum(x,axis=1)<300

    rectify_video_by_gradation('videos/gp1080p_noodle_float_move.mp4', 'videos/noodle_float_move_rect2.mp4',threshold_condition, show = True)
    


    