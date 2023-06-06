import cv2 as cv
import numpy as np


def frameResize(frame: np.ndarray, width: int = 200):
    """
    Resizes the frame to a given width while maintaining the aspect ratio

    Parameters
    ----------
    frame: numpy.ndarray
        Frame to resize
    width: int
        Width to resize the frame to

    Returns
    -------
    scaledFrame: numpy.ndarray
        Resized frame
    """
    # Calculating frame dimensions and aspect ration
    frameH, frameW = frame.shape[:2]
    aspectRatio = frameW / frameH

    # Scale the frame's width, while keeping its aspect ratio
    scaledFrame = cv.resize(
        frame, (width, int(width / aspectRatio)), interpolation=cv.INTER_AREA)

    # Return
    return scaledFrame


def imageConcatHorizontal(imageList: list, windowWidth: int):
    """
    Concatenates a list of images horizontally.

    Parameters
    ----------
    imageList: list
        List of images to concatenate.
    """
    try:
        numberOfImages = len(imageList)

        # Resizing the images
        resizedImageList = [frameResize(
            img, int(windowWidth / numberOfImages)) for img in imageList]

        # Returning the final image
        return cv.hconcat(resizedImageList)
    except Exception as exception:
        print(
            f'Error occurred in imageConcatHorizontal!\n{exception}', 'error')
