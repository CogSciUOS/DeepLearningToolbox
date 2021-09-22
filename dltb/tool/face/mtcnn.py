"""The Multi-task Cascaded Convolutional Network (MTCNN) face
detection and landmarking model.  The model can perform face detection
and facial landmarking.

"""

# standard imports
from typing import Iterable

# thirdparty imports
import numpy as np

# toolbox imports
from ...base.image import Imagelike
from ...base.implementation import Implementable
from .landmarks import FacialLandmarksBase, DetectorBase as LandmarksDetector
from .detector import Detector as FaceDetector


class Landmarks(FacialLandmarksBase):
    """Landmarks annotation scheme used by the MTCNN detector.
    The MTCNN landmarking scheme consists of 5 points.
    """

    keypoint_names = ('nose', 'mouth_right', 'right_eye', 'left_eye',
                      'mouth_left')

    def __init__(self, keypoints=None, points=None, **kwargs):
        if keypoints is not None:
            points = np.ndarray((len(self.keypoint_names), 2))
            for i, name in enumerate(self.keypoint_names):
                points[i] = keypoints[name]
        super().__init__(points, **kwargs)


# FIXME[problem]: should inherit from LandmarksDetector[Landmarks],
# not simply LandmarksDetector.
# But as LandmarksDetector is RegisterClass, the [] is overloaded,
# trying to initialize an instance of that class ...
class Detector(FaceDetector, LandmarksDetector, Implementable):
    """A :py:class:`DetectorMTCNN` combines a :py:class:`FaceDetector`
    with a :py:class:`LandmarkDetector`.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.detect_boxes = True
        self.detect_landmarks = True

    def detect_faces_and_landmarks(self, image: Imagelike) -> Iterable:
        """Detect faces and facial landmarks.
        """
