#!/usr/bin/env python3
import os
import sys
sys.path.append('./lib')
import g2o
import pangolin
import numpy as np
import cv2
from feature_extractor import Frame, denormalize, match_frames, IRt
import OpenGL.GL as gl

from multiprocessing import Process, Queue

MP4 = 'data/videos/output.mp4'
W, H = 1238, 374
F = 270
K = np.array(([F, 0, W//2], [0, F, H//2], [0, 0, 1]))


class Map(object):
    def __init__(self):
        self.points = []
        self.frames = []
        self.state = None
        self.q = Queue()

        p = Process(target = self.viewer_thread, args = (self.q,))
        p.daemon = True
        p.start()

    def viewer_thread(self, q):
        self.viewer_init(W*2, H*2)
        while True:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        pangolin.CreateWindowAndBind('Main', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.1, 100),
            pangolin.ModelViewLookAt(0, -10, -20, 0, 0, 0, 0, -1, 0)
        )

        self.handler = pangolin.Handler3D(self.scam)

        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
        self.dcam.SetHandler(self.handler)

    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        self.dcam.Activate(self.scam)


        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawCameras(self.state[0])

        gl.glPointSize(2.0)
        gl.glColor3f(0.0, 1.0, 0.0)
        if len(np.array(self.state[1]).shape) == 2:
            pangolin.DrawPoints(self.state[1])

        pangolin.FinishFrame()

    def display(self):
        poses, pts = [], []

        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.location)
        self.q.put((poses, pts))



class Point(object):
    def __init__(self, loc, global_map):
        self.frames = []
        self.location = loc
        self.idx = []
        self.id = len(global_map.points)

    def add_obervation(self, frame, idx):
        self.frames.append(frame)
        self.idx.append(idx)


def triangulate(pose1, pose2, points1, points2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], points1.T, points2.T).T


global_map = Map()

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
