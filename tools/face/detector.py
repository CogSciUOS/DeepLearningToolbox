import imutils
import imutils.face_utils
import numpy as np

from util.image import imresize, BoundingBox
from base.observer import Observable, change
from base.register import RegisterMetaclass
from datasource import Metadata
from tools.detector import (ImageDetector as BaseDetector,
                            ImageController as Controller)


class Detector(BaseDetector, metaclass=RegisterMetaclass):
    """Base class for face detectors.
    """

    @staticmethod
    def create(name: str, prepare: bool=True):
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


    
    # FIXME[old]
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
                    bX, bY, bW, bH = imutils.face_utils.rect_to_bb(rect)
                    # rect is of type <class 'dlib.rectangle'>
                else:
                    bX, bY, bW, bH = rect
                cv2.rectangle(canvas, (bX, bY), (bX + bW, bY + bH), color, 1)


Detector.register('haar', 'tools.face.opencv', 'DetectorHaar')
Detector.register('ssd', 'tools.face.opencv', 'DetectorSSD')
Detector.register('hog', 'tools.face.dlib', 'DetectorHOG')
Detector.register('cnn', 'tools.face.dlib', 'DetectorCNN')
Detector.register('mtcnn', 'tools.face.mtcnn', 'DetectorMTCNN')
