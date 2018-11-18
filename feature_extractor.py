import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

class FeatureExtractor:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.last = None
        self.bf_matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING
        )
        print('FeatureExtractor: initialized feature extractor')

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

        if len(ret) > 0:
            ret = np.array(ret)

            model, inliers = ransac(
                                (ret[:, 0], ret[:, 1]),
                                FundamentalMatrixTransform,
                                min_samples = 8, residual_threshold  = 1,
                                max_trials = 100
                            )
            ret = ret[inliers]
        self.last = {'keypoints': keypoints, 'descriptors': descriptors}
        return ret
