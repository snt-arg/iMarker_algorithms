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


def channelSeparatorRGB(frame: np.ndarray, channel: str):
    """
    Separates the channels of an RGB image based on the configuration.
    The function uses the RGB color space to separate the channels.

    Parameters
    ----------
    frame: numpy.ndarray
        Frame obtained from the camera.
    channel: str
        Channel to be separated.

    Returns
    ----------
    procFrame: numpy.ndarray
        Grayscale image with chosen channel.
    """
    # Variables
    procFrame = np.copy(frame)

    # Split the channels
    blue, green, red = cv.split(frame)

    # Which channel do we need?
    if (channel == 'r'):
        procFrame = red
    if (channel == 'g'):
        procFrame = green
    if (channel == 'b'):
        procFrame = blue
    if (channel == 'r' or channel == 'g' or channel == 'b'):
        # Enhance the frames
        procFrame = cv.equalizeHist(procFrame)
        procFrame = cv.cvtColor(procFrame, cv.COLOR_GRAY2BGR)

    # Return the processed frame
    return procFrame


def channelSeparatorHSV(frame: np.ndarray, channel: str, range: dict):
    """
    Separates the channels of an HSV image based on the configurations.
    The function uses the HSV color space to separate the channels.

    Parameters
    ----------
    frame: numpy.ndarray
        Frame obtained from the camera.
    channel: str
        Channel to be separated.
    range: dict
        Dictionary containing the HSV range for the colors.

    Returns
    ----------
    procFrame: numpy.ndarray
        Grayscale image with chosen color
    """
    # Variables
    procFrame = np.copy(frame)
    greenL = np.array(range['hsv_green']['lower'])
    greenH = np.array(range['hsv_green']['upper'])

    # Fixed HSV values
    lowerHsvRed1 = np.array([0, 175, 20])
    lowerHsvRed2 = np.array([170, 175, 20])
    higherHsvRed1 = np.array([10, 255, 255])
    higherHsvRed2 = np.array([180, 255, 255])

    lowerHsvBlue = np.array([100, 50, 50])
    higherHsvBlue = np.array([130, 255, 255])

    # Masks for different colors
    maskGreen = cv.inRange(frame, greenL, greenH)
    maskRed1 = cv.inRange(frame, lowerHsvRed1, higherHsvRed1)
    maskRed2 = cv.inRange(frame, lowerHsvRed2, higherHsvRed2)
    maskBlue = cv.inRange(frame, lowerHsvBlue, higherHsvBlue)

    # Which channel do we need?
    if (channel == 'r'):
        procFrame = maskRed1 + maskRed2
    if (channel == 'g'):
        procFrame = maskGreen
    if (channel == 'b'):
        procFrame = maskBlue
    if (channel == 'r' or channel == 'g' or channel == 'b'):
        procFrame = cv.cvtColor(procFrame, cv.COLOR_GRAY2BGR)

    # Return the processed frame
    return procFrame
