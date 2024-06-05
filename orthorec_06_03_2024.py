'''
Gordon Doore 
06/03/2024
orthorec-06-03-2024.py

functions to rectify image based on input quadrangle

Last Modified: 06/03/2024

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

def get_ppm(img, points, pixel_column, gradation_size):
    '''
    get ppm of different stakes, assumes it is relatively similar for the stake
    ie that the camera is sufficiently far away that there is no perspective warping
    in the vertical direction
    '''
    


def rectify_by_gradation(img,n_stakes):
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

    #get pixel values of the lines and add them to stake_columns
    for stake in range(n_stakes):
        rr = all_lines[stake][0]
        cc = all_lines[stake][1]
        stake_line = img[rr,cc]
        stake_columns.append(stake_line)
    
    #once we have the columns we need to get ppm (pixel per meter) for each stake
    #We will use pixel values according to spikes (gradations)
    ppm_each_stake = get_ppm(img, all_points, stake_columns)
    
    #we will first threshold the image to binarize from spike to not spike










    pass

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
    img = cv2.imread('orthotest_frames/orthotest_frame_8.jpg')

    # old_points = np.array(pick_points(img), dtype=np.float32)
    # old_points = order_points(old_points)  

    # new_points = np.array([[0,0],[old_points[-1,0],0],[0,old_points[-1,1]],old_points[-1]], dtype=np.float32)
    # rectified = rectify(img, old_points, new_points)
    # cv2.imshow('Rectified Image', rectified)
    # cv2.waitKey(0)

    #rectify_video('pole_movement_1.MP4', 'rectified.mp4')
    points, lines = define_stakes(img, 3)
    print(lines[0])
    