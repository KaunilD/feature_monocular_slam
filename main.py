import numpy as np
import cv2
from feature_extractor import FeatureExtractor

MP4 = 'data/videos/1.mp4'

ROWS = 1920//2
COLS = 1080//2
orb = cv2.ORB_create()

"""
    Process every frame here.
    1. Idenfify the features in the images.
    2.

"""
def process_frame(frame, feature_extractor):
    frame = cv2.resize(frame, (ROWS, COLS))
    keypoints, descriptors = feature_extractor.extract(frame)
    for kp in keypoints:
        u, v = map(lambda x: int(round(x)), kp.pt)
        cv2.circle(
            frame, (u, v), color=(0, 0, 255), radius = 2
        )
    return frame

if __name__ == '__main__':
    frame_idx = 0
    video_cap = cv2.VideoCapture(MP4)
    feature_extractor = FeatureExtractor()
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if ret:
            frame = process_frame(frame, feature_extractor)
            frame_idx += 1
        else:
            break

        print('CAPTURE: [shape: {}, frame_idx: {}]'.format(frame.shape, frame_idx))
        cv2.imshow('slam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_cap.release()
    cv2.destroyAllWindows()
