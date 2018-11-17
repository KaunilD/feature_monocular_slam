import numpy as np
import cv2


MP4 = 'data/videos/1.mp4'

ROWS = 1920//2
COLS = 1080//2


"""
    Process every frame here.
    1. Idenfify the features in the images.
    2. 

"""
def process_frame(frame):
    frame = cv2.resize(frame, (ROWS, COLS))
    return frame

if __name__ == '__main__':
    frame_idx = 0
    video_cap = cv2.VideoCapture(MP4)

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if ret:
            frame = process_frame(frame)
            frame_idx += 1
        else:
            break

        print('CAPTURE [shape: {}, frame_idx: {}]'.format(frame.shape, frame_idx))
        cv2.imshow('slam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
