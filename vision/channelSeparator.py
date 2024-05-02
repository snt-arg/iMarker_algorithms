import cv2 as cv
import numpy as np


def channelSeparatorRGB(frame: np.ndarray, params: dict):
    """
    Separates the channels of an RGB image based on the configuration

    Parameters
    ----------
    frame: numpy.ndarray
        Frame obtained from the camera
    params: dict
        Dictionary with the channel-related parameters

    Returns
    ----------
    procFrame: numpy.ndarray
        Grayscale image with chosen channel
    """
    procFrame = frame
    blue, green, red = cv.split(frame)

    if (params['rChannel']):
        procFrame = red
    if (params['gChannel']):
        procFrame = green
    if (params['bChannel']):
        procFrame = blue
    if (params['rChannel'] or params['gChannel'] or params['bChannel']):
        procFrame = cv.cvtColor(procFrame, cv.COLOR_GRAY2BGR)

    return procFrame


def channelSeparatorHSV(frame: np.ndarray, params: dict):
    """
    Separates the channels of an HSV image based on the configuration

    Parameters
    ----------
    frame: numpy.ndarray
        Frame obtained from the camera
    params: dict
        Dictionary with the color-related parameters

    Returns
    ----------
    procFrame: numpy.ndarray
        Grayscale image with chosen color
    """
    procFrame = frame

    # Fixed HSV values
    lowerHsvRed1 = np.array([0, 175, 20])
    lowerHsvRed2 = np.array([170, 175, 20])
    higherHsvRed1 = np.array([10, 255, 255])
    higherHsvRed2 = np.array([180, 255, 255])

    lowerHsvGreen = np.array([35, 50, 50])
    higherHsvGreen = np.array([90, 255, 255])

    lowerHsvBlue = np.array([100, 50, 50])
    higherHsvBlue = np.array([130, 255, 255])

    # Masks for different colors
    maskRed1 = cv.inRange(frame, lowerHsvRed1, higherHsvRed1)
    maskRed2 = cv.inRange(frame, lowerHsvRed2, higherHsvRed2)
    maskBlue = cv.inRange(frame, lowerHsvBlue, higherHsvBlue)
    maskGreen = cv.inRange(frame, lowerHsvGreen, higherHsvGreen)

    if (params['rChannel']):
        procFrame = maskRed1 + maskRed2
    if (params['gChannel']):
        procFrame = maskGreen
    if (params['bChannel']):
        procFrame = maskBlue
    if (params['rChannel'] or params['gChannel'] or params['bChannel']):
        procFrame = cv.cvtColor(procFrame, cv.COLOR_GRAY2BGR)

    return procFrame
