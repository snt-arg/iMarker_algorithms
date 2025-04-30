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
from .vision.postProcessing import postProcessing
from .vision.frameAlignment import alignFrames, alignFramesWithMatrix
from .vision.channelSeparator import channelSeparatorRGB, channelSeparatorHSV


def stereoFrameProcessing(frameL: np.ndarray, frameR: np.ndarray,
                          retL: bool, retR: bool, config: dict, isUsb: bool):
    """
    Process the frames obtained from two cameras and return the detected markers.

    Parameters
    ----------
    frameL: numpy.ndarray
        Left camera frame in RGB format
    frameR: numpy.ndarray
        Right camera frame in RGB format
    retL: bool
        True if the left camera frame is valid
    retR: bool
        True if the right camera frame is valid
    config: dict
        Dictionary containing the parameters for the processing
    isUsb: bool
        True if the sensor is USB and False if it is iDS

    Returns
    -------
    frame: numpy.ndarray
        The processed frame after concatanation
    mask: numpy.ndarray
        The processed frame mask
    """
    # Get the config values
    cfgProc = config['algorithm']['process']
    cfgAlign = cfgProc['alignment']
    cfgColorRange = cfgProc['colorRange']

    # Variables
    matchRate = int(cfgAlign['matchRate'])
    maxFeatures = int(cfgAlign['maxFeatures'])

    # Define an empty image
    emptyImage = np.zeros((300, 300), dtype=np.uint8)

    # Check if the frames are valid
    if not retL:
        height, width = frameR.shape[:2]
        emptyImage = np.empty((width, height), frameR.dtype)
        return emptyImage, frameR, emptyImage
    if not retR:
        height, width = frameL.shape[:2]
        emptyImage = np.empty((width, height), frameL.dtype)
        return frameL, emptyImage, emptyImage

    # Convert the frames to HSV
    frameLHSV = cv.cvtColor(frameL, cv.COLOR_RGB2HSV)
    frameRHSV = cv.cvtColor(frameR, cv.COLOR_RGB2HSV)

    # Which channels do we need?
    procFrameL = channelSeparatorHSV(
        frameLHSV, cfgProc['channel'], cfgColorRange)
    procFrameR = channelSeparatorHSV(
        frameRHSV, cfgProc['channel'], cfgColorRange)

    try:
        # Alignment based on setup
        if isUsb:
            frameLReg = alignFrames(
                procFrameL, procFrameR, maxFeatures, matchRate)
            frameRReg = alignFrames(
                procFrameR, procFrameL, maxFeatures, matchRate)
        else:
            # Use the preset alignment or not
            usePreset = cfgAlign['usePreset']
            procFrameL = channelSeparatorRGB(frameL, cfgProc['channel'])
            procFrameR = channelSeparatorRGB(frameR, cfgProc['channel'])
            frameLReg = alignFramesWithMatrix(
                procFrameL, config['presetMat']) if usePreset else alignFrames(
                procFrameL, procFrameR)
            frameRReg = alignFramesWithMatrix(
                procFrameR, config['presetMat']) if usePreset else alignFrames(
                procFrameR, procFrameL)

        # Frames Subtraction
        frameLR = cv.subtract(frameLReg, procFrameR)
        frameRL = cv.subtract(frameRReg, procFrameL)

        # Post-processing
        frameLR = postProcessing(frameLR, config)
        frameRL = postProcessing(frameRL, config)

        # Obtaining the mask image
        mask = frameRL if cfgProc['subtractRL'] else frameLR

        # Return the frame to be shown in a window
        return frameL, frameR, mask

    except Exception as exception:
        print(
            f'Running failed in stereoFrameProcessing!\n{exception}', 'error')
        return frameL, frameR, emptyImage


def singleFrameProcessing(frame: np.ndarray, ret: bool, config: dict):
    """
    Process the frames obtained from a single camera and return the thresholded image.

    Parameters
    ----------
    frame: numpy.ndarray
        Camera frame in RGB format
    ret: bool
        True if the camera frame is valid
    config: dict
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
    cfgColorRange = cfgProc['colorRange']
    isIR = config['mode']['runner'] == 'sv_ir'
    isUV = config['mode']['runner'] == 'offimguv' or config['mode']['runner'] == 'usbuv'

    # Define a null frame
    height, width = frame.shape[:2]
    emptyImage = np.empty((width, height), frame.dtype)

    try:
        # Variables
        mask = None

        # Check if grayscale
        if isUV or isIR:
            # Convert the frame to grayscale
            frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Apply the histogram equalization
            # frameGray = cv.equalizeHist(frameGray)
            # Retrieve camera frames (and check if they are valid)
            frameGray = frameGray if ret else emptyImage
            # Convert the frame to RGB
            procFrame = cv.cvtColor(frameGray, cv.COLOR_GRAY2BGR)
            # Post-processing
            mask = postProcessing(procFrame, config, False)
            # Convert back to RGB
            frameRGB = procFrame
        else:
            # Convert the frame to HSV
            frameHSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            # Retrieve camera frames (and check if they are valid)
            frameHSV = frameHSV if ret else emptyImage
            procFrame = frameHSV
            # Which channels do we need?
            procFrame = channelSeparatorHSV(
                frameHSV, cfgProc['channel'], cfgColorRange)
            # Post-processing
            mask = postProcessing(procFrame, config, True)
            # Convert back to RGB
            frameRGB = cv.cvtColor(frameHSV, cv.COLOR_HSV2BGR)

        # Return the frame to be shown in a window
        return frameRGB, mask

    except Exception as exception:
        print(
            f'Running failed in singleFrameProcessing!\n{exception}', 'error')
        return frame, emptyImage


def sequentialFrameProcessing(prevFrame: np.ndarray, currFrame: np.ndarray, ret: bool, config: dict):
    """
    Process sequential frames obtained from a mono camera and return the subtracted image.

    Parameters
    ----------
    prevFrame: numpy.ndarray
        Camera's previous frame
    currFrame: numpy.ndarray
        Camera's current frame
    ret: bool
        True if the camera frame is valid
    config: dict
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
    cfgColorRange = cfgProc['colorRange']

    # Define a null frame
    height, width = currFrame.shape[:2]
    emptyImage = np.empty((width, height), currFrame.dtype)

    # Retrieve camera frames (and check if they are valid)
    currFrame = currFrame if ret else emptyImage
    prevFrame = prevFrame if ret else emptyImage

    # Which channels do we need?
    procCurrFrame = channelSeparatorRGB(
        currFrame, cfgProc['channel'])
    procPrevFrame = channelSeparatorRGB(
        prevFrame, cfgProc['channel'])

    try:
        # Thresholding
        subFrame = cv.subtract(procCurrFrame, procPrevFrame) if cfgProc['subtractRL'] else cv.subtract(
            procPrevFrame, procCurrFrame)
        # Post-processing
        mask = postProcessing(subFrame, config)
        # Return the frame to be shown in a window
        return prevFrame, currFrame, mask

    except Exception as exception:
        print(
            f'Running failed in sequentialFrameProcessing!\n{exception}', 'error')
        return prevFrame, currFrame, emptyImage
