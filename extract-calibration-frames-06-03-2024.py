import numpy as np
import cv2
def extract_calibration_frames(filepath, nframes):
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
    calibration_frames = extract_calibration_frames('ortho_test_video.MP4',11)
    #save frames to their own images 
    dest_folder = 'orthotest_frames/'
    for i, frame in enumerate(calibration_frames):
        cv2.imwrite(dest_folder + f'orthotest_frame_{i}.jpg', frame)