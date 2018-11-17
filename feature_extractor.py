import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.orb = cv2.ORB_create(100)
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
            cv2.KeyPoint(x = feature[0][0], y = feature[0][1], _size = 20) \
            for feature in features
            ]
        descriptors = self.orb.compute(image, keypoints)
        return keypoints, descriptors
