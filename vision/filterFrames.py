"""
📝 'iMarker Detector Algorithms' Software
    SPDX-FileCopyrightText: (2025) University of Luxembourg
    © 2025 University of Luxembourg
    Developed by: Ali TOURANI et al. at SnT / ARG.

'iMarker Detector Algorithms' is licensed under the "SNT NON-COMMERCIAL" License.
You may not use this file except in compliance with the License.
"""

import cv2 as cv
import numpy as np


def applyCircularMask(image: np.ndarray, coverage: float = 1.0, isInverted: bool = False):
    """
    Filters a region of interest from an image based on the provided configurations.
    The main functionality is to remove circular obstacles in the image hardware setup.

    Parameters:
    -----------
    image: numpy.ndarray
        Image to be filtered
    coverage: float
        The coverage of the circle diameter
    isInverted: bool
        If True, the mask will be black, otherwise white

    Returns:
    --------
    mask: numpy.ndarray
        Filtered image based on region of interest
    """
    # Grab the frame's height and width and center points
    frameH, frameW = image.shape[:2]
    centerX, centerY = frameW // 2, frameH // 2

    # Define the circle equations
    radius = int(centerY * coverage) if frameH < frameW else int(
        centerX * coverage)

    # Create a blank image of the same size as the original image
    mask = np.zeros_like(image)
    mask = cv.circle(mask, (centerX, centerY), radius, (255, 255, 255), -1)

    # Subtraction
    if isInverted:
        mask = cv.bitwise_and(mask, cv.bitwise_not(image))
        mask = cv.bitwise_not(mask)
    else:
        mask = cv.bitwise_and(mask, image)

    # Return the result
    return mask
