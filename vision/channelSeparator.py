import cv2 as cv
import numpy as np


def channelSeparator(frame: np.ndarray, params: dict):
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
    blue, green, red = cv.split(frame)

    if (params['rChannel']):
        procFrame = red
    if (params['gChannel']):
        procFrame = green
    if (params['bChannel']):
        procFrame = blue
    if (params['rChannel'] or params['gChannel'] or params['bChannel']):
        procFrame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

    return procFrame
