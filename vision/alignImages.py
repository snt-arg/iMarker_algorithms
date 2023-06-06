import cv2 as cv
import numpy as np


def alignImages(frame1: np.ndarray, frame2: np.ndarray,
                maxFeatures: int = 500, goodMatchPercentage: float = 0.4):
    """
    Aligns two frames using ORB features and descriptors.

    Parameters
    ----------
    frame1: numpy.ndarray
        Frame obtained from the left camera
    frame2: numpy.ndarray
        Frame obtained from the right camera

    Returns:
    --------
    frame1Reg: numpy.ndarray
        Registered version of the first frame
    """
    try:
        # Detect ORB features and compute descriptors
        orb = cv.ORB_create(maxFeatures)
        keypointsL, descriptorsL = orb.detectAndCompute(frame1, None)
        keypointsR, descriptorsR = orb.detectAndCompute(frame2, None)
        # Match features
        descriptorMatcher = cv.DescriptorMatcher_create(
            cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = descriptorMatcher.match(descriptorsL, descriptorsR, None)
        matches = list(matches)
        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)
        # Remove improper matches
        bestMatchesLength = int(len(matches) * goodMatchPercentage)
        matches = matches[:bestMatchesLength]
        # When there are no matches, return the original frame
        if (matches == []):
            return frame2
        # Extract location of good matches
        pointsL = np.zeros((len(matches), 2), dtype=np.float32)
        pointsR = np.zeros((len(matches), 2), dtype=np.float32)
        # Iterate over matches
        for index, match in enumerate(matches):
            pointsL[index, :] = keypointsL[match.queryIdx].pt
            pointsR[index, :] = keypointsR[match.trainIdx].pt
        # Making sure that there are at least four corresponding point sets
        if len(pointsL) < 4 or len(pointsR) < 4:
            return frame2
        # Find and use homography
        homography, mask = cv.findHomography(pointsL, pointsR, cv.RANSAC)
        # To avoid error when there are no matches
        # assert homography != None, 'No homography found!'
        if (homography is None):
            return frame1
        height, width = frame2.shape[:2]
        # Create registered image for left camera frame
        return cv.warpPerspective(
            frame1, homography, (width, height))
    except Exception as exception:
        print(f'Error occurred in alignImages!\n{exception}', 'error')
        return frame1
