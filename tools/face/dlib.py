
import os
import dlib

from .detector import Detector


class DetectorHOG(Detector):
    """The dlib HOG detector.
    """
    _detector = None


    def _prepare(self):
        self._detector = dlib.get_frontal_face_detector()

    def prepared(self):
        return self._detector is not None

    def detect(self, image):
        """Apply the dlib historgram of gradients detector (HOG) to
        detect faces in the given image.
        """
        if self._detector is None:
            return None

        rects = self._detector(image, 2)
        return rects

class DetectorCNN(Detector):
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

    def detect(self, image):
        """The dlib CNN face detector.
        """
        if self._detector is None:
            return None

        dets = self._detector(image, 2)

        # It is also possible to pass a list of images to the
        # detector - like this:
        #   dets = detector([image # list], upsample_num, batch_size = 128)
        # In this case it will return a mmod_rectangless object. This object
        # behaves just like a list of lists and can be iterated over.

        # d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(),
        # d.confidence

        rects = [d.rect for d in dets]
        confs = [d.confidence for d in dets]
        return rects
