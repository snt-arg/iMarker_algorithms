import cv2 as cv
from filterROI import applyCircularMask


def postProcessing(frame, procParams):
    """
    Post-processing of the frame.

    Parameters:
    -----------
    frame: numpy.ndarray
        Frame obtained from the camera
    procParams: dict
        Dictionary with the parameters for the post-processing

    Returns:
    --------
    processedMask: numpy.ndarray
        Processed frame
    """
    try:
        # Preparation
        thresholdingMethod = cv.THRESH_BINARY if procParams['threshbin'] else cv.THRESH_OTSU if procParams[
            'threshots'] else cv.THRESH_BINARY + cv.THRESH_OTSU
        # Convert image to grayscale
        frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Apply threshold
        frameGray = cv.GaussianBlur(
            frameGray, (int(procParams['gaussianKernel']), int(procParams['gaussianKernel'])), 0)
        _, mask = cv.threshold(frameGray, procParams['threshold'], 255,
                               thresholdingMethod)
        # Apply region of interest
        if (procParams['enableCircularMask']):
            mask = applyCircularMask(mask, procParams['circlularMaskCoverage'])
        # Apply morphological operations
        erodeKernel = cv.getStructuringElement(
            cv.MORPH_RECT, (int(procParams['erosionKernel']), int(procParams['erosionKernel'])))
        mask = cv.morphologyEx(mask, cv.MORPH_ERODE, erodeKernel)
        # Create updated frame
        processedMask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        return processedMask
    except Exception as exception:
        print(f'Error occurred in postProcessing!\n{exception}', 'error')
