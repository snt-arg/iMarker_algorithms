import cv2 as cv
import numpy as np


def applyCircularMask(image: np.ndarray, coverage: float = 1.0):
    """
    Filters a region of interest from an image based on the provided configurations.
    The main functionality is to remove circular obstacles in the image hardware setup.

    Parameters:
    -----------
    image: numpy.ndarray
        Image to be filtered
    coverage: float
        The coverage of the circle diameter

    Returns:
    --------
    mask: numpy.ndarray
        Filtered image based on region of interest
    """
    # Grab the frame's height and width and center points
    frameH, frameW = image.shape[:2]
    centerX, centerY = frameW // 2, frameH // 2
    # Define the circle equations
    radius = int(centerY * coverage) if frameH < frameW else int(
        centerX * coverage)
    # Create a blank image of the same size as the original image
    mask = np.zeros_like(image)
    mask = cv.circle(mask, (centerX, centerY), radius, (255, 255, 255), -1)
    # Subtraction
    mask = cv.bitwise_and(mask, image)
    # Return the result
    return mask


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
        # Apply ROI
        if (params['enableCircularMask']):
            mask = applyCircularMask(mask, params['circlularMaskCoverage'])

        # Apply morphological operations
        erodeKernel = cv.getStructuringElement(
            cv.MORPH_RECT, (int(params['erosionKernel']), int(params['erosionKernel'])))
        mask = cv.morphologyEx(mask, cv.MORPH_ERODE, erodeKernel)

        # Create updated frame
        processedMask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

        # Return the value
        return processedMask
    except Exception as exception:
        print(f'Error occurred in postProcessing!\n{exception}', 'error')
