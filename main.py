#!/usr/bin/env python3
import os
import sys
sys.path.append('lib/')
import g2o
import numpy as np
import cv2
from feature_extractor import Frame, denormalize, match_frames, IRt


MP4 = 'data/videos/1.mp4'
W, H = 1920//2, 1080//2
F = 270
K = np.array(([F, 0, W//2], [0, F, H//2], [0, 0, 1]))


class Point(object):
    def __init__(self, loc):
        self.frames = []
        self.location = loc
        self.idx = []

    def add_obervation(self, frame, idx):
        self.frames.append(frame)
        self.idx.append(idx)


def triangulate(pose1, pose2, points1, points2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], points1.T, points2.T).T


frames = []
def process_frame(img):
    img = cv2.resize(img, (W, H))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    frame = Frame(img_gray, K)
    frames.append(frame)

    if len(frames) <= 1:
        return img

    idx1, idx2, Rt = match_frames(frames[-1], frames[-2])
    frames[-1].pose = np.dot(frames[-2].pose, Rt)


    points4d = triangulate(
        frames[-1].pose, frames[-2].pose,
        frames[-1].points[idx1], frames[-2].points[idx2]
        )
    points4d /= points4d[:, 3:]
    good_points4d = (np.abs(points4d[:, 3]) > 0.005) & (points4d[:, 2] > 0)

    points4d = points4d[good_points4d]

    for idx, point in enumerate(points4d):
        if not good_points4d[idx]:
            continue
        p = Point(point)
        p.add_obervation(frames[-1], idx1[idx])
        p.add_obervation(frames[-2], idx2[idx])


    for p1, p2 in zip(frames[-1].points[idx1], frames[-2].points[idx2]):
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
