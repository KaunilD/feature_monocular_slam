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


F = 800
K = np.array(([F, 0, W//2], [0, F, H//2], [0, 0, 1]))

def triangulate(pose1, pose2, points1, points2):
    ret = np.zeros((points1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, p in enumerate(zip(points1, points2)):
        A = np.zeros((4,4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
    return ret


global_map = Map(W, H)
global_map.create_display()

def process_frame(img):
    img = cv2.resize(img, (W, H))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    frame = Frame(img_gray, K, global_map)


    if len(global_map.frames) <= 1:
        return img

    f1 = global_map.frames[-1]
    f2 = global_map.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(f2.pose, Rt)

    for i in range(len(f2.points)):
        if f2.points is not None:
            f2.points



    points4d = triangulate(
        f1.pose, f2.pose,
        f1.key_points[idx1], f2.key_points[idx2]
        )
    points4d /= points4d[:, 3:]

    unmatched_points = np.array([f1.points[i] is None for i in idx1]).astype(np.bool)
    good_points4d = (np.abs(points4d[:, 3]) > 0.005) & (points4d[:, 2] > 0) & unmatched_points

    for idx, point in enumerate(points4d):
        if not good_points4d[idx]:
            continue
        p = Point(point, global_map)
        global_map.points.append(p)
        p.add_obervation(f1, idx1[idx])
        p.add_obervation(f2, idx2[idx])


    for p1, p2 in zip(f1.key_points[idx1], f2.key_points[idx2]):
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
