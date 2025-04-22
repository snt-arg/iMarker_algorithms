import cv2 as cv
import numpy as np
from .filterFrames import applyCircularMask


def postProcessing(frame: np.ndarray, config: dict, isHSV: bool = False):
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
        isUsbCam = config['mode']['runner'] == 'usb'
        cfgUsbCam = config['sensor']['usbCam'] if isUsbCam else None
        cfgPostprocess = config['algorithm']['postprocess']
        cfgThreshold = cfgPostprocess['threshold']['method']

        # Check if frame is RGB or HSV
        if (isHSV):
            frame = cv.cvtColor(frame, cv.COLOR_HSV2BGR)

        # Convert image to grayscale
        frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        frameGray = cv.GaussianBlur(
            frameGray, (int(cfgPostprocess['gaussianKernelSize']), int(cfgPostprocess['gaussianKernelSize'])), 0)

        # Apply thresholding
        if (cfgThreshold == 'adaptive'):
            # Check the block size to be odd and greater than 1
            blockSize = int(cfgPostprocess['threshold']['size'])
            if (blockSize < 2):
                blockSize = 3
            if (blockSize % 2 == 0):
                blockSize += 1
            # Apply adaptive thresholding
            mask = cv.adaptiveThreshold(
                frameGray,
                maxValue=255,
                blockSize=blockSize,
                thresholdType=cv.THRESH_BINARY,
                adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                C=2
            )
        else:
            # Check the method
            threshMethod = cv.THRESH_BINARY if cfgThreshold == 'binary' else cv.THRESH_OTSU
            _, mask = cv.threshold(
                frameGray, cfgPostprocess['threshold']['size'], 255, threshMethod)

        # Inverting the binary image
        if (cfgPostprocess['invertBinary']):
            mask = cv.bitwise_not(mask)

        # Apply ROI
        if isUsbCam and cfgUsbCam['enableMask']:
            mask = applyCircularMask(
                mask, cfgUsbCam['maskSize'], cfgPostprocess['invertBinary'])

        # Apply morphological operations
        erodeKernel = cv.getStructuringElement(
            cv.MORPH_RECT, (int(cfgPostprocess['erosionKernelSize']), int(cfgPostprocess['erosionKernelSize'])))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, erodeKernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, erodeKernel)

        # Return the value
        return mask
    except Exception as exception:
        print(f'Error occurred in postProcessing!\n{exception}', 'error')
