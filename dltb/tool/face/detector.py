"""Base class for face detectors.

Some implementations:
* 'haar': dltb.thirdparty.opencv.face.DetectorHaar
* 'ssd': dltb.thirdparty.opencv.face.DetectorSSD
* 'hog': dltb.thirdparty.dlib.face.DetectorHOG
* 'cnn': dltb.thirdparty.dlib.face.DetectorCNN
"""

# toolbox imports
from ...base.implementation import Implementable
from ..detector import ImageDetector


class Detector(ImageDetector, Implementable):
    # pylint: disable=too-many-ancestors
    """Base class for face detectors.
    """
