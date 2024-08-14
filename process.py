import cv2 as cv
import numpy as np
from .vision.postProcessing import postProcessing
from .vision.alignImages import alignImages, alignImagesWithMatrix
from .vision.channelSeparator import channelSeparatorRGB, channelSeparatorHSV


def processStereoFrames(frameL: np.ndarray, frameR: np.ndarray,
                        retL: bool, retR: bool, config: dict, isUsb: bool):
    """
    Process the frames obtained from two cameras and return the detected markers.

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
    config : dict
        Dictionary containing the parameters for the processing
    isUsb : bool
        True if the sensor is USB and False if it is iDS

    Returns
    -------
    frame: numpy.ndarray
        The processed frame after concatanation
    mask: numpy.ndarray
        The processed frame mask
    """
    # Get the config values
    cfgMarker = config['marker']
    cfgAlgorithm = config['algorithm']
    cfgUsbCam = config['sensor']['usbCam']
    cfgGeneral = config['sensor']['general']

    # Define a not found image
    notFoundImage = cv.imread('./src/notFound.png')

    # Check if the frames are valid
    if not retL:
        height, width = frameR.shape[:2]
        emptyImage = np.empty((width, height), frameR.dtype)
        return notFoundImage, frameR, emptyImage
    if not retR:
        height, width = frameL.shape[:2]
        emptyImage = np.empty((width, height), frameL.dtype)
        return frameL, notFoundImage, emptyImage

    # Which channels do we need?
    procFrameL = channelSeparatorRGB(
        frameL, cfgAlgorithm['process']['channel'])
    procFrameR = channelSeparatorRGB(
        frameR, cfgAlgorithm['process']['channel'])

    try:
        # Alignment based on setup
        if isUsb:
            frameLReg = alignImages(procFrameL, procFrameR)
            frameRReg = alignImages(procFrameR, procFrameL)
        else:
            # Use the preset alignment or not
            usePreset = config['algorithm']['process']['alignment']['usePreset']
            frameLReg = alignImagesWithMatrix(
                procFrameL, config['presetMat']) if usePreset else alignImages(
                procFrameL, procFrameR)
            frameRReg = alignImagesWithMatrix(
                procFrameR, config['presetMat']) if usePreset else alignImages(
                procFrameR, procFrameL)

        # Frames Subtraction
        frameLR = cv.subtract(frameLReg, procFrameR)
        frameRL = cv.subtract(frameRReg, procFrameL)

        # Post-processing
        frameLR = postProcessing(frameLR, config)
        frameRL = postProcessing(frameRL, config)

        # Obtaining the mask image
        mask = frameRL if (
            cfgMarker['structure']['leftHanded']) else frameLR

        # Return the frame to be shown in a window
        return frameL, frameR, mask

    except Exception as exception:
        print(f'Running failed in processStereoFrames!\n{exception}', 'error')
        return frameL, frameR, emptyImage


def processSingleFrame(frame: np.ndarray, ret: bool, config: dict):
    """
    Process the frames obtained from a single camera and return the thresholded image.

    Parameters
    ----------
    frame : numpy.ndarray
        Camera frame
    ret : bool
        True if the camera frame is valid
    config : dict
        Dictionary containing the parameters for the processing

    Returns
    -------
    frame: numpy.ndarray
        The processed frame
    mask: numpy.ndarray
        The processed frame mask
    """
    # Parameters
    cfgProc = config['algorithm']['process']

    # Define a null frame
    height, width = frame.shape[:2]
    emptyImage = np.empty((width, height), frame.dtype)

    # Retrieve camera frames (and check if they are valid)
    frame = frame if ret else emptyImage
    procFrame = frame

    # Which channels do we need?
    procFrame = channelSeparatorHSV(frame, cfgProc['channel'])

    try:
        # Post-processing
        mask = postProcessing(procFrame, config)

        # Return the frame to be shown in a window
        return frame, mask

    except Exception as exception:
        print(f'Running failed in processSingleFrame!\n{exception}', 'error')
        return frame, emptyImage


def processSequentialFrames(prevFrame: np.ndarray, currFrame: np.ndarray, ret: bool, config: dict):
    """
    Process sequential frames obtained from a mono camera and return the subtracted image.

    Parameters
    ----------
    prevFrame : numpy.ndarray
        Camera's previous frame
    currFrame : numpy.ndarray
        Camera's current frame
    ret : bool
        True if the camera frame is valid
    config : dict
        Dictionary containing the parameters for the processing

    Returns
    -------
    frame: numpy.ndarray
        The processed frame
    mask: numpy.ndarray
        The processed frame mask
    """
    # Parameters
    cfgProc = config['algorithm']['process']

    # Define a null frame
    height, width = currFrame.shape[:2]
    emptyImage = np.empty((width, height), currFrame.dtype)

    # Retrieve camera frames (and check if they are valid)
    currFrame = currFrame if ret else emptyImage
    prevFrame = prevFrame if ret else emptyImage

    procCurrFrame = currFrame
    procPrevFrame = prevFrame

    # Which channels do we need?
    procCurrFrame = channelSeparatorRGB(currFrame, cfgProc['channel'])
    procPrevFrame = channelSeparatorRGB(prevFrame, cfgProc['channel'])

    try:
        # Thresholding
        subFrame = cv.subtract(procCurrFrame, procPrevFrame)
        # Post-processing
        mask = postProcessing(subFrame, config)
        # Return the frame to be shown in a window
        return prevFrame, currFrame, mask

    except Exception as exception:
        print(
            f'Running failed in processSequentialFrames!\n{exception}', 'error')
        return prevFrame, currFrame, emptyImage
