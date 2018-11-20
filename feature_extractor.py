import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform


class FeatureExtractor:
    def __init__(self, K):
        self.orb = cv2.ORB_create()
        self.last = None
        self.bf_matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING
        )

        self.K = K
        self.Kinv = np.linalg.inv(K)
        print('FeatureExtractor: initialized feature extractor')

    def add_ones(self, x):
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis = 1)

    def normalize(self, point):
        return np.dot(self.Kinv, self.add_ones(point).T).T[:, 0:2]

    def denormalize(self, point):
        denorm = np.dot(self.K, [point[0], point[1], 1.0])
        return int(round(denorm[0])), int(round(denorm[1]))

    def extract(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        features = cv2.goodFeaturesToTrack(
                        image,
                        qualityLevel = 0.01,
                        maxCorners = 3000,
                        minDistance = 3
                    )

        keypoints = [
                cv2.KeyPoint(
                    x = feature[0][0],
                    y = feature[0][1],
                    _size = 30
                ) for feature in features
        ]
        keypoints, descriptors = self.orb.compute(image, keypoints)

        ret = []
        R, t = None, None
        if self.last is not None:
            matches = self.bf_matcher.knnMatch(
                        descriptors,
                        self.last['descriptors'], k = 2
                    )
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    keypoint_1 = keypoints[m.queryIdx].pt
                    keypoint_2 = self.last['keypoints'][m.trainIdx].pt
                    ret.append((keypoint_1, keypoint_2))

        self.last = {'keypoints': keypoints, 'descriptors': descriptors}
        if len(ret) > 0:
            ret = np.array(ret)
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])
            model, inliers = ransac(
                                (ret[:, 0], ret[:, 1]),
                                FundamentalMatrixTransform,
                                min_samples = 8, residual_threshold  = 3,
                                max_trials = 100
                            )
            ret = ret[inliers]

            R, t = self.calc_pose_matrices(model)

        # print('FEATUREEXTRACTOR: matches: {}'.format(len(ret)))
        return ret, (R, t)

    def calc_pose_matrices(self, model):
        W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float)
        U, w, V = np.linalg.svd(model.params)

        assert np.linalg.det(U) > 0

        if np.linalg.det(V) < 0:
            V *= -1.0

        R = np.dot(np.dot(U, W), V)

        if np.sum(R.diagonal()) < 0:
            R = np.dot(np.dot(U, W.T), V)
        t = U[:, 2]
        return R, t
