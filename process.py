import cv2 as cv
import numpy as np
from vision.postProcessing import postProcessing
from vision.concatImages import imageConcatHorizontal
from vision.alignImages import alignImages, alignImagesWithMatrix


def processFrames(frameL: np.ndarray, frameR: np.ndarray,
                  retL: bool, retR: bool, params: dict):
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
    params : dict
        Dictionary containing the parameters for the processing

    Returns
    -------
    frame: numpy.ndarray
        The processed frame
    """
    # Define a null frame
    height, width = frameL.shape[:2]
    emptyImage = np.empty((width, height), frameL.dtype)

    # Retrieve camera frames (and check if they are valid)
    frameL = frameL if retL else emptyImage
    frameR = frameR if retR else emptyImage
    procFrameL, procFrameR = frameL, frameR

    # Which channels do we need?
    blueL, greenL, redL = cv.split(frameL)
    blueR, greenR, redR = cv.split(frameR)

    if (params['rChannel']):
        procFrameL = redL
        procFrameR = redR
    if (params['gChannel']):
        procFrameL = greenL
        procFrameR = greenR
    if (params['bChannel']):
        procFrameL = blueL
        procFrameR = blueR
    if (params['rChannel'] or params['gChannel'] or params['bChannel']):
        procFrameL = cv.cvtColor(procFrameL, cv.COLOR_GRAY2BGR)
        procFrameR = cv.cvtColor(procFrameR, cv.COLOR_GRAY2BGR)

    try:
        # Align images (if both are retrieved, align them, otherwise, return the notFound image)
        frameLReg = alignImagesWithMatrix(
            procFrameL, params['homographyMat']) if params['preAligment'] else alignImages(
            procFrameL, procFrameR)
        frameRReg = alignImagesWithMatrix(
            procFrameR, params['homographyMat']) if params['preAligment'] else alignImages(
            procFrameR, procFrameL)

        # Frames Subtraction
        frameLR = cv.subtract(frameLReg, procFrameR)
        frameRL = cv.subtract(frameRReg, procFrameL)

        # Post-processing
        frameLR = postProcessing(frameLR, params)
        frameRL = postProcessing(frameRL, params)

        # Concatenate frames
        frame = imageConcatHorizontal(
            [frameL, frameR, frameRL if (params['isMarkerLeftHanded']) else frameLR])

        # Return the frame to be shown in a window
        return frame

    except Exception as exception:
        print(f'Running failed in processFrames!\n{exception}', 'error')
        return imageConcatHorizontal([frameL, frameR, emptyImage])
