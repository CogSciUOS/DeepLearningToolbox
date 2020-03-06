
import os
import numpy as np

import mtcnn

from .detector import Detector as FaceDetector
from .landmarks import Detector as LandmarkDetector, FacialLandmarks
from datasources import Metadata
from util.image import BoundingBox

class FacialLandmarksMTCNN(FacialLandmarks):

    keypoint_names = ('nose', 'mouth_right', 'right_eye', 'left_eye',
                      'mouth_left')
    
    def __init__(self, keypoints=None, points=None, **kwargs):
        if keypoints is not None:
            points = np.ndarray((len(self._points,2)))
            for i, name in enumerate(self.keypoint_names):
                points[i] = keypoints[name]
        super.__init__(points, **kwargs)


class DetectorMTCNN(FaceDetector, LandmarkDetector):
    """The Multi-task Cascaded Convolutional Network (MTCNN) face and
    facial landmark detector.

    This detector uses the model from the 'mtcnn' python
    module [1,2,3]. It can be installed via `pip install mtcnn` and
    provides a pre-trained model to be run out of the box.
    The module is based on Keras (>=2.3, with TensorFlow backend) and
    in addition requires OpenCV (>4.1).

    The detector can take an image of arbitrary size as input and
    returns a list of detected faces, for each face providing
    * a bounding box
    * facial landmarks (using a 5-point scheme)
    * a confidence value

    Hence this class realizes both, a FaceDetector and a
    LandmarkDetector.
    
    [1] https://pypi.org/project/mtcnn/
    [2] https://github.com/ipazc/mtcnn
    [3] https://towardsdatascience.com/how-does-a-face-detection-program-work-using-neural-networks-17896df8e6ff

    Attributes
    ----------
    _detector:
        The actual detector, an instance of the class mtcnn.MTCNN.
    """
    _detector = None


    def _prepare(self) -> None:
        """Prepare this DetectorMTCNN by loading the model data.
        """
        self._detector = mtcnn.MTCNN()

    def prepared(self) -> bool:
        """The DetectorMTCNN is prepared, once the model data
        have been loaded.
        """
        return self._detector is not None

    def _detect(self, image) -> Metadata:
        """Apply the MTCNN detector to detect faces in the given image.

        Arguments
        ---------
        image:
        """

        faces = self._detector.detect_faces(image)

        detections = Metadata(description=
                              'Detections by the DLib HOG detector')
        for id, face in enumerate(faces):
            confidence = face['confidence']

            x, y, w, h = face['box']
            detections.add_region(BoundingBox(x=x, y=y, width=w, height=h),
                                  confidence=confidence, id=id)

            keypoints = face['keypoints']
            detections.add_region(FacialLandmarksMTCNN(points),
                                  confidence=confidence, id=id)        

        rects = self._detector(image, 2)

        return detections
