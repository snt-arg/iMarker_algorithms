import cv2 as cv
import numpy as np
from .filterROI import applyCircularMask


def postProcessing(frame: np.ndarray, config: dict):
    """
    Post-processing of the frame.

    Parameters:
    -----------
    frame: numpy.ndarray
        Frame obtained from the camera
    config: dict
        Dictionary with the parameters for the post-processing

    Returns:
    --------
    processedMask: numpy.ndarray
        Processed frame
    """
    try:
        # Processing parameters
        cfgUsbCam = config['sensor']['usbCam']
        cfgPostprocess = config['algorithm']['postprocess']
        cfgThreshold = cfgPostprocess['threshold']
        # Preparing the thresholding method
        threshMethod = cv.THRESH_BINARY if cfgThreshold['method'] == 'binary' else cv.THRESH_OTSU if cfgThreshold[
            'method'] == 'otsu' else cv.THRESH_BINARY + cv.THRESH_OTSU

        # Convert image to grayscale
        frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Apply threshold
        frameGray = cv.GaussianBlur(
            frameGray, (int(cfgPostprocess['gaussianKernelSize']), int(cfgPostprocess['gaussianKernelSize'])), 0)
        _, mask = cv.threshold(
            frameGray, cfgPostprocess['threshold']['size'], 255, threshMethod)

        # Inverting the binary image
        if (cfgPostprocess['invertBinary']):
            mask = cv.bitwise_not(mask)

        # Apply ROI
        isUsbCam = config['mode']['runner'] == 'usb'
        if isUsbCam and cfgUsbCam['enableMask']:
            mask = applyCircularMask(mask, cfgUsbCam['maskSize'])

        # Apply morphological operations
        erodeKernel = cv.getStructuringElement(
            cv.MORPH_RECT, (int(cfgPostprocess['erosionKernelSize']), int(cfgPostprocess['erosionKernelSize'])))
        mask = cv.morphologyEx(mask, cv.MORPH_ERODE, erodeKernel)

        # Return the value
        return mask
    except Exception as exception:
        print(f'Error occurred in postProcessing!\n{exception}', 'error')
