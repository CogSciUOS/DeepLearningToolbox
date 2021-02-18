"""Base class for face detectors
"""

# third party imports
# FIXME[todo]: remove third party dependencies (there are also inline import down in the code)!
 
import numpy as np

# toolbox imports
from ..detector import ImageDetector


class Detector(ImageDetector):
    # pylint: disable=too-many-ancestors
    """Base class for face detectors.
    """

    @staticmethod
    def create(name: str, prepare: bool = True):
        """Create a new face detector.
        """
        if name == 'haar':
            from .opencv import DetectorHaar
            detector = DetectorHaar()
        elif name == 'ssd':
            from .opencv import DetectorSSD
            detector = DetectorSSD()
        elif name == 'hog':
            from .dlib import DetectorHOG
            detector = DetectorHOG()
        elif name == 'cnn':
            from .dlib import DetectorCNN
            detector = DetectorCNN()
        else:
            raise ValueError("face.create_detector: "
                             f"unknown detector name '{name}'.")

        if prepare:
            detector.prepare()
        return detector

    #
    # FIXME[old]
    #

    def paint_detect(self, canvas: np.ndarray, rects, color=(255, 0, 0)):
        """Mark detected faces in an image.

        Here we assume that faces are detected by rectangular bounding
        boxes, provided as (x,y,width,height) or dlib.rectangle.

        Arguments:
        ==========
        canvas: numpy.ndarray of size (*,*,3)
            The image in which the faces will be marked.
        rects: iterable
        color: (int,int,int)
        """
        if rects is None:
            return

        import cv2
        import dlib

        # check to see if a face was detected, and if so, draw the total
        # number of faces on the image
        if True or len(rects) > 0:
            text = "{} face(s) found".format(len(rects))
            if canvas is not None:
                cv2.putText(canvas, text,
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
        # loop over the face detections
        if canvas is not None:
            for rect in rects:
                # compute the bounding box of the face and draw it on the
                # image
                if isinstance(rect, dlib.rectangle):
                    # rect is of type <class 'dlib.rectangle'>
                    pos_x, pos_y = rect.left(), rect.top()
                    width, height = rect.right()-pos_x, rect.bottom()-pos_y
                else:
                    pos_x, pos_y, width, height = rect
                cv2.rectangle(canvas, (pos_x, pos_y),
                              (pos_x + width, pos_y + height), color, 1)
