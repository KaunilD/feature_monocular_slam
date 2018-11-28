#!/usr/bin/env python3
import os
import sys
sys.path.append('./lib')
import g2o
import numpy as np
import cv2
from feature_extractor import Frame, denormalize, match_frames, IRt
import OpenGL.GL as gl
from point import Point
from map import Map

from multiprocessing import Process, Queue
"""
MP4 = 'data/videos/output.mp4'
W, H = 1238, 374
"""
MP4 = 'data/videos/5.mp4'
W, H = 1920//2, 1080//2


F = 270
K = np.array(([F, 0, W//2], [0, F, H//2], [0, 0, 1]))

def triangulate(pose1, pose2, points1, points2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], points1.T, points2.T).T


global_map = Map(W, H)

def process_frame(img):
    img = cv2.resize(img, (W, H))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    frame = Frame(img_gray, K, global_map)
    global_map.frames.append(frame)

    if len(global_map.frames) <= 1:
        return img

    f1 = global_map.frames[-1]
    f2 = global_map.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(f2.pose, Rt)


    points4d = triangulate(
        f1.pose, f2.pose,
        f1.points[idx1], f2.points[idx2]
        )
    points4d /= points4d[:, 3:]
    good_points4d = (np.abs(points4d[:, 3]) > 0.005) & (points4d[:, 2] > 0)

    points4d = points4d[good_points4d]

    for idx, point in enumerate(points4d):
        if not good_points4d[idx]:
            continue
        p = Point(point, global_map)
        global_map.points.append(p)
        p.add_obervation(f1, idx1[idx])
        p.add_obervation(f2, idx2[idx])


    for p1, p2 in zip(f1.points[idx1], f2.points[idx2]):
        u1, v1 = denormalize(K, p1)
        u2, v2 = denormalize(K, p2)
        cv2.circle(
            img, (u1, v1), color=(0, 0, 255), radius = 2
        )
        cv2.line(
            img, (u1, v1), (u2, v2), color=(0, 255, 0)
        )
    global_map.display()
    return img

if __name__ == '__main__':
    frame_idx = 0
    video_cap = cv2.VideoCapture(MP4)

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        H, W = frame.shape[:2]
        if ret:
            frame = process_frame(frame)
            frame_idx += 1
        else:
            break
        # print('CAPTURE: [shape: {}, frame_idx: {}]'.format(frame.shape, frame_idx))
        cv2.imshow('FRAMES', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_cap.release()
    cv2.destroyAllWindows()
