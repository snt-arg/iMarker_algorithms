import cv2 as cv
import numpy as np


def alignImages(frame1: np.ndarray, frame2: np.ndarray,
                maxFeatures: int = 500, goodMatchPercentage: float = 0.4):
    """
    Aligns a frame to another using ORB features and descriptors.

    Parameters
    ----------
    frame1: numpy.ndarray
        Frame obtained from the first camera
    frame2: numpy.ndarray
        Frame obtained from the second camera
    maxFeatures: int
        Maximum number of features to use for aligning the frames
    goodMatchPercentage: float
        The percentage threshold to be used for matching

    Returns:
    --------
    frame1Reg: numpy.ndarray
        Registered version of the first frame
    """
    try:
        # Detect ORB features and compute descriptors
        orb = cv.ORB_create(maxFeatures)
        keypoints1, descriptors1 = orb.detectAndCompute(frame1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(frame2, None)
        # Match features
        descriptorMatcher = cv.DescriptorMatcher_create(
            cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = descriptorMatcher.match(descriptors1, descriptors2, None)
        matches = list(matches)
        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)
        # Remove improper matches
        bestMatchesLength = int(len(matches) * goodMatchPercentage)
        matches = matches[:bestMatchesLength]
        # When there are no matches, return the original frame
        if (matches == []):
            return frame1
        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        # Iterate over matches
        for index, match in enumerate(matches):
            points1[index, :] = keypoints1[match.queryIdx].pt
            points2[index, :] = keypoints2[match.trainIdx].pt
        # Making sure that there are at least four corresponding point sets
        if len(points1) < 4 or len(points2) < 4:
            return frame1
        # Find and use homography
        homography, mask = cv.findHomography(points1, points2, cv.RANSAC)
        # To avoid error when there are no matches
        if (homography is None):
            return frame1
        height, width = frame2.shape[:2]
        # Create registered image for left camera frame
        return cv.warpPerspective(
            frame1, homography, (width, height))
    except Exception as exception:
        print(f'Error occurred in alignImages!\n{exception}', 'error')
        return frame1
