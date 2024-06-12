import cv2 
import numpy as np
import matplotlib.pyplot as plt


def floats_video_to_waveform(rectified_video_path, ppm, num_stakes, 
                             arr_out_path = 'wave_measurements.npz',
                             graph_out_path = 'position_graphs.png'):
    # Load the video
    cap = cv2.VideoCapture(rectified_video_path)

    # Read the first frame
    ret, frame = cap.read()

    #get the total frames:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a tracker object
    trackers = []
    for i in range(num_stakes):
        roi = cv2.selectROI(frame, False)
        tracker = cv2.legacy_TrackerCSRT.create()
        trackers.append(tracker)
        ret = tracker.init(frame, roi)
    # Initialize the tracker

    position = np.zeros((total_frames, num_stakes,2))

    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            break
        #get current frame number:
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # Update each tracker
        for i,tracker in enumerate(trackers):
            ret, roi = tracker.update(frame)

            # Draw the ROI on the frame
            if ret:
                (x, y, w, h) = tuple(map(int, roi))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                #record the positoin of the center of the box 
                center_x = x + w // 2
                center_y = y + h // 2
                position[current_frame-1,i] = [center_x,center_y]
            else:
                '''Tracking failure
                when tracking failure occurs, we search for objects similar to the one we have been tracking
                we do this in a vertical column around where we lost the object since we are really only concerned 
                with that type of movement'''

                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Tracking', frame)

        # Exit if ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

    #convert position to real units: 
    position = position/ppm


    # Extract x and y coordinates from the position data
    x = position[:,0,1]
    y = position[:,1,1]

    # Plot the x and y coordinates through time
    fig = plt.figure()
    plt.plot(x, label='y1')
    plt.plot(y, label='y2')
    plt.xlabel('Time')
    plt.ylabel('Coordinates')
    plt.legend()
    fig.savefig(graph_out_path)

    np.save(arr_out_path,position)

if __name__ == '__main__':
    floats_video_to_waveform('videos/noodle_float_move_rect.mp4',0.2,2)