"""
📝 Frame Alignment Module

This module contains functions to align images using ORB features and descriptors.
"""

import cv2 as cv
import numpy as np


def alignFrames(frame1: np.ndarray, frame2: np.ndarray,
                maxFeatures: int = 500, goodMatchPercentage: float = 0.4):
    """
    Aligns a frame to another using ORB features and descriptors.

    Parameters
    ----------
    frame1: numpy.ndarray
        First frame to be aligned.
    frame2: numpy.ndarray
        Second frame to align to.
    maxFeatures: int
        Maximum number of features to detect.
    goodMatchPercentage: float
        Percentage of good matches to keep.
        This value should be between 0 and 1.
        A value of 0.4 means that 40% of the best matches will be kept.

    Returns:
    --------
    frameReg: numpy.ndarray
        Registered image.
        If no matches are found or an error occurs, the original frame1 is returned.
    """
    try:
        # Detect ORB features and compute descriptors
        orb = cv.ORB_create(maxFeatures)
        keypoints1, descriptors1 = orb.detectAndCompute(frame1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(frame2, None)

        # Check if descriptors are None
        if descriptors1 is None or descriptors2 is None:
            print(
                "[Warning] No descriptors found in one or both images. Returning the original frame ...")
            return frame1

        # Convert descriptors to the appropriate type if necessary
        if descriptors1.dtype != np.uint8:
            descriptors1 = descriptors1.astype(np.uint8)
        if descriptors2.dtype != np.uint8:
            descriptors2 = descriptors2.astype(np.uint8)

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

        # Create registered image for left camera frame
        height, width = frame2.shape[:2]
        frameReg = cv.warpPerspective(
            frame1, homography, (width, height))

        # Return the registered image
        return frameReg
    except Exception as exception:
        print(f'Error occurred in alignFrames!\n{exception}', 'error')
        return frame1


def alignFramesWithMatrix(frame: np.ndarray, homographyMat: np.ndarray):
    """
    Aligns two images using a pre-defined homography matrix.

    Parameters
    ----------
    frame: numpy.ndarray
        Frame obtained from the input visual sensor.
    homographyMat: numpy.ndarray
        Homography matrix to be used for alignment.
        This matrix should be obtained from the calibration process.

    Returns
    ----------
    frameReg: numpy.ndarray
        Registered image
    """
    # Check if the homography matrix is valid
    if homographyMat is None or not isinstance(homographyMat, np.ndarray):
        print("[Warning] Invalid homography matrix. Returning the original frame ...")
        return frame

    # Check if the homography matrix has the correct shape
    if homographyMat.shape != (3, 3):
        print(
            "[Warning] Invalid homography matrix shape. Returning the original frame ...")
        return frame

    # Get the dimensions of the input frame
    height, width = frame.shape[:2]

    # Generate the registered image using the homography matrix and the input frame
    frameReg = cv.warpPerspective(
        frame, homographyMat, (width, height))

    # Return the registered image
    return frameReg
