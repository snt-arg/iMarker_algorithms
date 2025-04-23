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


def resizeFrame(frame: np.ndarray, width: int = 200):
    """
    Resizes a frame to a given width while maintaining the aspect ratio.

    Parameters
    ----------
    frame: numpy.ndarray
        Frame to be resized.
    width: int
        Desired width of the resized frame.
        Default is 200.

    Returns
    -------
    scaledFrame: numpy.ndarray
        Resized frame with the given width.
    """
    # Calculating frame dimensions and aspect ration
    height, width = frame.shape[:2]
    aspectRatio = width / height

    # Scale the frame's width, while keeping its aspect ratio
    scaledFrame = cv.resize(
        frame, (width, int(width / aspectRatio)), interpolation=cv.INTER_AREA)

    # Return
    return scaledFrame


def concatFramesHorizontal(imageList: list, windowWidth: int):
    """
    Concatenates a list of images horizontally into a single image.
    The images are resized to fit within the specified window width.

    Parameters
    ----------
    imageList: list
        List of images to be concatenated.
    windowWidth: int
        Desired width of the concatenated image.

    Returns
    -------
    concatenatedImage: numpy.ndarray
        Concatenated image with the given width.
    """
    try:
        numberOfImages = len(imageList)

        # Resizing the images
        resizedImageList = [resizeFrame(
            img, int(windowWidth / numberOfImages)) for img in imageList]

        # Returning the final image
        return cv.hconcat(resizedImageList)
    except Exception as exception:
        print(
            f'Error occurred in concatFramesHorizontal!\n{exception}', 'error')
        return np.empty((0, 0), dtype=np.uint8)
