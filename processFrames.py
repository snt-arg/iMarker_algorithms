import cv2 as cv
import numpy as np
from vision.alignImages import alignImages
from vision.postProcessing import postProcessing
from vision.concatImages import imageConcatHorizontal


def processFrames(frameL: np.ndarray, frameR: np.ndarray,
                  retL: bool, retR: bool, procParams: dict):
    """
    Process the frames obtained from cameras and return the detected markers.

    Parameters
    ----------
    frameL : numpy.ndarray
        Left camera frame
    frameR : numpy.ndarray
        Right camera frame
    retL : bool
        True if the left camera frame is valid
    retR : bool
        True if the right camera frame is valid
    procParams : dict
        Dictionary containing the parameters for the processing

    Returns
    -------
    frame: numpy.ndarray
        The processed frame
    """
    # Define a notFound image
    notFoundImage = cv.imread('src/notFound.png', cv.IMREAD_COLOR)
    # Retrieve camera frames (and check if they are valid)
    frameL = frameL if retL else notFoundImage
    frameR = frameR if retR else notFoundImage
    processedFrameL, processedFrameR = frameL, frameR
    # Which channels do we need?
    blueL, greenL, redL = cv.split(frameL)
    blueR, greenR, redR = cv.split(frameR)
    if (procParams['rChannel']):
        processedFrameL = redL
        processedFrameR = redR
    if (procParams['gChannel']):
        processedFrameL = greenL
        processedFrameR = greenR
    if (procParams['bChannel']):
        processedFrameL = blueL
        processedFrameR = blueR
    if (procParams['rChannel'] or procParams['gChannel'] or procParams['bChannel']):
        processedFrameL = cv.cvtColor(processedFrameL, cv.COLOR_GRAY2BGR)
        processedFrameR = cv.cvtColor(processedFrameR, cv.COLOR_GRAY2BGR)
    try:
        # Align images (if both are retrieved, align them, otherwise, return the notFound image)
        frameLReg = alignImages(processedFrameL, processedFrameR)
        frameRReg = alignImages(processedFrameR, processedFrameL)
        # Frames Subtraction
        frameLR = cv.subtract(frameLReg, processedFrameR)
        frameRL = cv.subtract(frameRReg, processedFrameL)
        # Post-processing
        frameLR = postProcessing(frameLR, procParams)
        frameRL = postProcessing(frameRL, procParams)
        frameOR = frameLR + frameRL
        # Concatenate frames
        frame = imageConcatHorizontal(
            [frameL, frameR, frameRL if (procParams['isMarkerLeftHanded']) else frameLR])
        # Return the frame to be shown in a window
        return frame
    except Exception as exception:
        print(f'Running failed in processFrames!\n{exception}', 'error')
        return imageConcatHorizontal([frameL, frameR, notFoundImage])
