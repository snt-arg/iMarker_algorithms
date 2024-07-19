import cv2 as cv
import numpy as np
from .filterROI import applyCircularMask


def postProcessing(frame: np.ndarray, params: dict):
    """
    Post-processing of the frame.

    Parameters:
    -----------
    frame: numpy.ndarray
        Frame obtained from the camera
    params: dict
        Dictionary with the parameters for the post-processing

    Returns:
    --------
    processedMask: numpy.ndarray
        Processed frame
    """
    try:
        # Preparing the thresholding method
        threshMethod = cv.THRESH_BINARY if params['threshbin'] else cv.THRESH_OTSU if params[
            'threshots'] else cv.THRESH_BINARY + cv.THRESH_OTSU

        # Convert image to grayscale
        frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Apply threshold
        frameGray = cv.GaussianBlur(
            frameGray, (int(params['gaussianKernel']), int(params['gaussianKernel'])), 0)
        _, mask = cv.threshold(frameGray, params['threshold'], 255,
                               threshMethod)

        # Inverting the binary image
        if (params['invertBinaryImage']):
            mask = cv.bitwise_not(mask)

        # Apply ROI
        if 'enableCircularMask' in params and params['enableCircularMask']:
            mask = applyCircularMask(mask, params['circlularMaskCoverage'])

        # Apply morphological operations
        erodeKernel = cv.getStructuringElement(
            cv.MORPH_RECT, (int(params['erosionKernel']), int(params['erosionKernel'])))
        mask = cv.morphologyEx(mask, cv.MORPH_ERODE, erodeKernel)

        # Create updated frame
        # processedMask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

        # Return the value
        return mask
    except Exception as exception:
        print(f'Error occurred in postProcessing!\n{exception}', 'error')
