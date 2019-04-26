import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform


IRt = np.eye(4)

def extract(image):
    orb = cv2.ORB_create()
    features = cv2.goodFeaturesToTrack(
                    image,
                    1000,
                    qualityLevel=0.01,
                    minDistance=7
                )

    keypoints = [
            cv2.KeyPoint(
                x = feature[0][0],
                y = feature[0][1],
                _size = 20
            ) for feature in features
    ]
    keypoints, descriptors = orb.compute(image, keypoints)
    features = np.array([(keypoint.pt[0], keypoint.pt[1]) for keypoint in keypoints])
    return features, descriptors

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis = 1)

def normalize(Kinv, point):
    return np.dot(Kinv, add_ones(point).T).T[:, 0:2]

def denormalize(K, point):
    denorm = np.dot(K, [point[0], point[1], 1.0])
    denorm/=denorm[2]
    return int(round(denorm[0])), int(round(denorm[1]))

def match_frames(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(f1.des, f2.des, k=2)

    ret = []
    idx1 = []
    idx2 = []

    for m, n in matches:
        if m.distance < 0.75*n.distance:
            keypoint_1 = f1.key_points[m.queryIdx]
            keypoint_2 = f2.key_points[m.trainIdx]
            if np.linalg.norm((keypoint_1-keypoint_2)) < 0.1*np.linalg.norm([f1.w, f1.h]) \
            and m.distance < 32:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                ret.append((keypoint_1, keypoint_2))
    assert len(ret) >= 8

    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    model, inliers = ransac(
                        (ret[:, 0], ret[:, 1]),
                        FundamentalMatrixTransform,
                        min_samples = 8, residual_threshold  = 0.001,
                        max_trials = 100
                    )
    points = ret[inliers]

    Rt = calc_pose_matrices(model)

    # print('FEATUREEXTRACTOR: matches: {}'.format(len(ret)))
    return idx1[inliers], idx2[inliers], Rt


def calc_pose_matrices(model):
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float)
    U, w, V = np.linalg.svd(model.params)

    assert np.linalg.det(U) > 0

    if np.linalg.det(V) < 0:
        V *= -1.0

    R = np.dot(np.dot(U, W), V)

    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), V)
    t = U[:, 2]
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret

class Frame(object):
    def __init__(self, img, K, global_map):

        self.K = K
        self.Kinv = np.linalg.inv(self.K)

        self.pose = IRt

        points, self.des = extract(img)
        self.h, self.w = img.shape[0:2]
        self.key_points = normalize(self.Kinv, points)
        self.points = [None]*len(self.key_points)

        self.id = len(global_map.frames)
        global_map.frames.append(self)
