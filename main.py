import numpy as np
import cv2
from feature_extractor import FeatureExtractor

MP4 = 'data/videos/1.mp4'

ROWS = 1920//4
COLS = 1080//4
F = 1
K = np.array(([F, 0, ROWS], [0, F, COLS], [0, 0, 1]))

def process_frame(frame, feature_extractor):
    frame = cv2.resize(frame, (ROWS, COLS))
    matches, (R, t) = feature_extractor.extract(frame)
    if len(matches) == 0:
        return frame
    for p1, p2 in matches:
        u1, v1 = feature_extractor.denormalize(p1)
        u2, v2 = feature_extractor.denormalize(p2)
        cv2.circle(
            frame, (u1, v1), color=(0, 0, 255), radius = 2
        )

        cv2.line(
            frame, (u1, v1), (u2, v2), color=(0, 255, 0)
        )
    # print(R, t)
    return frame

if __name__ == '__main__':
    frame_idx = 0
    video_cap = cv2.VideoCapture(MP4)
    feature_extractor = FeatureExtractor(K)

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if ret:
            frame = process_frame(frame, feature_extractor)
            frame_idx += 1
        else:
            break
        # print('CAPTURE: [shape: {}, frame_idx: {}]'.format(frame.shape, frame_idx))
        cv2.imshow('slam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_cap.release()
    cv2.destroyAllWindows()
