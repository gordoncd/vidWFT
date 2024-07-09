'''
This file has function extract_calibration_frames which randomly
samples frames from an input video and returns them as np arrays

the __name__ __main__ part of the file sends some frames from a calibration video to a folder 
so that each file has path orthotest_frames/orthotest_frame{i} where i is the order in which it was selected

Author: Gordon Doore
Created: 06/03/2024

Last Modified: 06/04/2024

'''

import numpy as np
import cv2
def extract_calibration_frames(filepath : str, nframes : int) ->list[np.ndarray]:
    #randomly grab nframes frames from the mp4 at
    # filepath and return them as a list of numpy arrays 
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.random.choice(total_frames, nframes, replace=False)
    frames = []
    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

if __name__ == '__main__':
    calibration_frames = extract_calibration_frames('calibration/acortiz@colbydotedu_CALIB/gopro_5k_vid_calib.MP4',11)
    #save frames to their own images 
    dest_folder = 'calibration/acortiz@colbydotedu_CALIB/calib_frames_5k/'
    for i, frame in enumerate(calibration_frames):
        cv2.imwrite(dest_folder + f'orthotest_frame_{i}.jpg', frame)