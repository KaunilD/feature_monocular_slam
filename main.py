#!/usr/bin/env python3
import os
import sys
sys.path.append('lib/')
import g2o
import numpy as np
import cv2
from feature_extractor import Frame, denormalize, match


MP4 = 'data/videos/1.mp4'
W, H = 1920//2, 1080//2
F = 1
K = np.array(([F, 0, W], [0, F, H], [0, 0, 1]))

frames = []
def process_frame(img):
    img = cv2.resize(img, (W, H))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    frame = Frame(img_gray, K)
    frames.append(frame)

    if len(frames) <= 1:
        return img

    ret, Rt = match(frames[-1], frames[-2])
    for p1, p2 in ret:
        u1, v1 = denormalize(K, p1)
        u2, v2 = denormalize(K, p2)
        cv2.circle(
            img, (u1, v1), color=(0, 0, 255), radius = 2
        )
        cv2.line(
            img, (u1, v1), (u2, v2), color=(0, 255, 0)
        )

    return img

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
        # print('CAPTURE: [shape: {}, frame_idx: {}]'.format(frame.shape, frame_idx))
        cv2.imshow('slam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_cap.release()
    cv2.destroyAllWindows()
