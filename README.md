# iMarker Detector Algorithms

![Detector](demo.png "Detector")

This repository contains the algorithms to detect iMarkers after receiving the feed from [the detector sensors](https://github.com/snt-arg/csr_sensors). It is mainly used alongside [the detector sensors](https://github.com/snt-arg/csr_detector) and wrapped by [GUI-enabled standalone version](https://github.com/snt-arg/csr_detector_standalone) and [ROS-based version](https://github.com/snt-arg/csr_detector_ros) frameworks.

## ⚙️ Installation

Install the required libraries for running the functions of this repository using the command `pip install numpy opencv-python` (tested with `opencv-python>4.10` and `numpy==1.x`).

## ⚒️ Algorithm Variations <a id="algorithms"></a>

Considering the setup chosen in [the detector sensors](https://github.com/snt-arg/csr_sensors#setup), the algorithm to detect iMarkers and CSRs may vary:

### A. Dual-vision Setup

In this setup, where beamsplitter plays a key role, [ELP](https://github.com/snt-arg/csr_sensors#usb-cam) and [iDS](https://github.com/snt-arg/csr_sensors#ids-cam) cameras are used to fetch visual data. The algorithm to process the frames is as follows:

```markdown
- Receiving fetched frames from a dual-vision sensor
- Aligning the images using ORB features
- Applying frame-level subtraction
- Applying thresholding and post-processing
- Send the final image for ArUco Marker detection
```

### B. Single-vision Setup with Fixed Polarizer (Camouflaged iMarker)

In this setup, [RealSense](https://github.com/snt-arg/csr_sensors#rs-cam) is used to fetch visual data. The algorithm to process the frames is as follows:

```markdown
- Receiving fetched frames from a single-vision sensor
- Set boundaries for HSV primary colors (red, green, and blue)
- Conversion to HSV
- Filtering the image based on the set color channel and boundaries
- Applying thresholding and post-processing
- Send the final image for ArUco Marker detection
```

### C. Single-vision Setup with Changable Polarizer (Function Generator)

In this setup, [RealSense](https://github.com/snt-arg/csr_sensors#rs-cam) is used to fetch visual data. The algorithm to process the frames is as follows:

```markdown
- Receiving fetched frames from a single-vision sensor
- Fetch each frame and subtract it from its previous frame
- Filtering the image based on the set color channel and boundaries
- Applying thresholding and post-processing
- Send the final image for ArUco Marker detection
```

## 📑 Code Structure

It should be noted that this repository contains the functions to use the introduced algorithms as below:

- **A. Image Processing Algorithms:** in the `vision/` directory, you can find below functions:
  - `alignImages.py`: contains `alignImages` to align two given images (online) and `alignImagesWithMatrix` to align two images using a pre-defined homography matrix (offline)
  - `channelSeparator.py`: contains `channelSeparatorRGB` and `channelSeparatorHSV` functions to filter the input image based on a given channel in RGB and HSV, respectively.
  - `concatImages.py`: contains `frameResize` to resize an image and `imageConcatHorizontal` to concat a set of images for visualization.
  - `filterROI.py`: contains `applyCircularMask` to apply a filtration mask on a given image (for dual-vision ELP camera setup)
  - `postProcessing.py`: contains `postProcessing` function to improve the final processed image
- **B. Core Runner:** in `process.py`, you can find three main functions for each of the algorithms introduced in [the algorithm variations](https://github.com/snt-arg/csr_detector#algorithms) section:
  - `processStereoFrames`: for dual-vision setups
  - `processSingleFrame`: for the single-vision with fixed polarizer (camouflaged iMarker) setup
  - `processSequentialFrames`: for the single-vision with changable polarizer (using the function generator) setup

## 🚀 Running the Code

As mentioned before, the current repository is a sub-module and wrapped by [GUI-enabled standalone version](https://github.com/snt-arg/csr_detector_standalone) and [ROS-based version](https://github.com/snt-arg/csr_detector_ros) frameworks. Accordingly, take a look at the mentioned repositories to see examples of using iMarker detection algorithms.
