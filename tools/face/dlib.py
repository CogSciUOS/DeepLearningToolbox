
import os
import numpy as np

import dlib


from .detector import Detector as FaceDetector
from .landmarks import Detector as LandmarkDetector
from datasource import Metadata
from util.image import BoundingBox


class DetectorHOG(FaceDetector):
    """The dlib HOG face detector.
    """
    _detector = None


    def _prepare(self) -> None:
        """Prepare this DetectorHOG by loading the model data.
        """
        self._detector = dlib.get_frontal_face_detector()

    def prepared(self) -> bool:
        """The DetectorHOG is prepared, once the model data
        have been loaded.
        """
        return self._detector is not None

    def _detect(self, image) -> Metadata:
        """Apply the dlib histogram of gradients detector (HOG) to
        detect faces in the given image.

        Arguments
        ---------
        image:
        """
        if self._detector is None:
            return None

        rects = self._detector(image, 2)

        detections = Metadata(description=
                              'Detections by the DLib HOG detector')
        for rect in rects:
            detections.add_region(BoundingBox(x=rect.left(), y=rect.top(),
                                              width=rect.width(),
                                              height=rect.height()))
        return detections

class DetectorCNN(FaceDetector):
    """The dlib CNN detector.
    """
    _model_file: str = None
    _detector = None

    def __init__(self, model_file='mmod_human_face_detector.dat',
                 *args, **kwargs):
        """The OpenCV Single Shot MultiBox Detector (SSD).

        Arguments
        ---------
        dnn: str
            The model to use. There are currently two models available:
            'CAFFE' is the original 16-bit floating point model trained
            with Caffe, and 'TF' is a 8-bit quantized version for TensorFlow.
        """
        super().__init__(*args, **kwargs)
        self.set_model_file(model_file)

    def set_model_file(self, model_file):       
        if not os.path.isabs(model_file):
            dlib_model_directory = os.environ.get('DLIB_MODELS', '.')
            model_file = os.path.join(dlib_model_directory, model_file)

        if model_file != self._model_file:
            self._model_file = model_file
            self._add_requirement('model', 'file', model_file)

    def _prepare(self):
        self._detector = dlib.cnn_face_detection_model_v1(self._model_file)

    def prepared(self):
        return self._detector is not None

    def _detect(self, image):
        """The dlib CNN face detector.
        """
        if self._detector is None:
            return None


        # It is also possible to pass a list of images to the
        # detector - like this:
        #   dets = detector([image # list], upsample_num, batch_size = 128)
        # In this case it will return a mmod_rectangless object. This object
        # behaves just like a list of lists and can be iterated over.
        dets = self._detector(image, 2)

        # The result is of type dlib.mmod_rectangles, which is
        # basically a list of rectangles annotated with conficence
        # values.  For an individual detection d (of type
        # dlib.mmode_rectangle), the information can be accessed by
        # d.rect and d.confidence.
        
        detections = Metadata(description=
                              'Detections by the dlib CNN face detector')
        for d in dets:
            detections.add_region(BoundingBox(x=d.rect.left(), y=d.rect.top(),
                                              width=d.rect.width(),
                                              height=d.rect.height()),
                                  confidence=d.confidence)
        return detections


class FacialLandmarkDetector(LandmarkDetector):
    """A facial landmark detector based on dlib.

    """

    _predictor: dlib.shape_predictor = None

    def __init__(self, model_file: str=None):
        # shape_predictor_5_face_landmarks.dat
        # shape_predictor_68_face_landmarks.dat
        
        #
        # The dlib facial landmark detector
        #
        if not os.path.exists(model_file):
            predictor_name = os.path.join(os.environ.get('DLIB_MODELS', '.'),
                                          model_file)
        if os.path.exists(predictor_name):
            self._predictor = dlib.shape_predictor(model_file)
        else:
            print(f"FIXME: not found '{predictor_name}'")
            raise ValueError(f"Dlib predictor model file ''{predictor_name}' "
                             "not found.")

    def _prepare(self) -> None:
        """Prepare this DetectorHOG by loading the model data.
        """
        self._detector = dlib.get_frontal_face_detector()

    def prepared(self) -> bool:
        """The DetectorHOG is prepared, once the model data
        have been loaded.
        """
        return self._detector is not None

    def _detection_to_landmarks(self, detection: dlib.full_object_detection):
        points = face_utils.shape_to_np(detection)

        # Construct a Metdata object holding the detected landmarks
        return FacialLandmarks68(points)

    def _detect(self, image: np.ndarray, box: BoundingBox=None):
        """Do the actual facial landmark detection.
        Notice, that the facial detector expects to work on a single face,
        not on a complete image with multiple faces.
        That is, the image provided to this function may be a crop
        from a larger image.
        """

        # The dlib.shape_predictor takes an image region, provided
        # by an image and a rectangle. As in our API, the facial
        # landmark detector alread expects cropped image region
        # as argument, we simply set the rectangle to include the
        # whole image.
        rect = (dlib.rectangle(0,0,image.shape[1],image.shape[0])
                if rect is None else
                dlib.rectangle(box.x, box.y, box.width, box.height))

        detection = self._predictor(image, rect)

        # detection is of type dlib.full_object_detection, which basically
        # is a list of N points. These can be transformed into
        # a two dimensional array of shape (N,2) for further processing.
        metadata = Metadata(description=
                            'Facial landmarks detectec by the dlib detctor')
        metadata.add_region(self._detection_landmarks(detection))

        return detections

    
    ## FIXME[old]:
    def predict_old(self, image, rects):
        if self._predictor is None:
            self._shapes = None
            return None

        self._shapes = []

        
        # loop over the face detections
        for rect in rects:
        
            # determine the facial landmarks for the face region, then
            # shape is of type <class 'dlib.full_object_detection'>
            shape = self._predictor(image, rect)
            self._shapes.append(shape)

            if self._canvas_predict is not None:
                # convert the facial landmark (x, y)-coordinates to a
                # NumPy array (shape: (n, 2)), with n being 0 (no
                # landmarks detected), 5 (for )
                shape = imutils.face_utils.shape_to_np(shape)
                show_numbers = len(shape) < 10
 
                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw each of them
                for (i, (x, y)) in enumerate(shape):
                    cv2.circle(self._canvas_predict,
                               (x, y), 1, (0, 0, 255), -1)
                    if show_numbers:
                        cv2.putText(self._canvas_predict,
                                    str(i + 1), (x - 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                    (0, 0, 255), 1)
