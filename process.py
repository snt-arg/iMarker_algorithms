import cv2 as cv
import numpy as np
from vision.postProcessing import postProcessing
from vision.channelSeparator import channelSeparator
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
        The processed frame after concatanation
    mask: numpy.ndarray
        The processed frame mask
    """
    # Define a null frame
    height, width = frameL.shape[:2]
    emptyImage = np.empty((width, height), frameL.dtype)

    # Retrieve camera frames (and check if they are valid)
    frameL = frameL if retL else emptyImage
    frameR = frameR if retR else emptyImage
    procFrameL, procFrameR = frameL, frameR

    # Which channels do we need?
    procFrameL = channelSeparator(frameL, params)
    procFrameR = channelSeparator(frameR, params)

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

        # Obtaining the mask image
        mask = frameRL if (
            params['isMarkerLeftHanded']) else frameLR

        # Concatenate frames
        frame = imageConcatHorizontal(
            [frameL, frameR, mask], params['windowWidth'])

        # Return the frame to be shown in a window
        return frame, mask

    except Exception as exception:
        print(f'Running failed in processFrames!\n{exception}', 'error')
        return imageConcatHorizontal([frameL, frameR, emptyImage], params['windowWidth'])
