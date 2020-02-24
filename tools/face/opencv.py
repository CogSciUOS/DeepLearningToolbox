
import os
import cv2
import numpy as np

from .detector import Detector


class DetectorHaar(Detector):
    """ The OpenCV Haar cascade face detector.
    
    _model_file: str
        Absolute path to the model file.
    """

    _model_file: str = None
    _detector = None

    def __init__(self, model_file='haarcascade_frontalface_default.xml',
                 *args, **kwargs):
        """The OpenCV Haar cascade face detector.

        Arguments
        ---------
        model_file: str
            The path to the pretrained model file.
        """
        super().__init__(*args, **kwargs)
        self.set_model_file(model_file)

    def _get_cascade_dir(self):
        """Get the default directory in which the files for the
        cascade classifiers are located."""
        cascade_path = cv2.__file__
        for _ in range(4):
            cascade_path, _ = os.path.split(cascade_path)
        if os.path.basename(cascade_path) == 'lib':
            cascade_path, _ = os.path.split(cascade_path)
    
        if os.path.isdir(os.path.join(cascade_path, 'share', 'OpenCV',
                                      'haarcascades')):
            cascade_path = os.path.join(cascade_path, 'share', 'OpenCV',
                                        'haarcascades')
        elif os.path.isdir(os.path.join(cascade_path, 'share', 'opencv4',
                                        'haarcascades')):
            cascade_path = os.path.join(cascade_path, 'share', 'opencv4',
                                        'haarcascades')
        return cascade_path

    def set_model_file(self, model_file):       
        if not os.path.isabs(model_file):
            opencv_data = self._get_cascade_dir()
            model_file = os.path.join(opencv_data, model_file)

        if model_file != self._model_file:
            self._model_file = model_file
            self._add_requirement('model_file', 'file', model_file)

    def _prepare(self):
        self._detector = cv2.CascadeClassifier(self._model_file)

        if self._detector.empty():
            self._detector = None
            raise RuntimeError("Haar detector is empty")

    def prepared(self):
        return self._detector is not None

    def detect(self, image: np.ndarray):
        if not self.prepared():
            raise RuntimeError("Running unprepared detector.")
        
        if image.ndim == 3 and image.shape[2] == 3:
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        rects = self._detector.detectMultiScale(gray)
        return rects


class DetectorSSD(Detector):
    """The OpenCV Single Shot MultiBox Face Detector (SSD).
        
    This model was included in OpenCV from version 3.3.
    It uses ResNet-10 Architecture as backbone.

    OpenCV provides 2 models for this face detector.
    1. Floating point 16 version of the original caffe implementation
       (5.4 MB)
    2. 8 bit quantized version using Tensorflow
       (2.7 MB)

    The required files can be fount in the opencv_extra repository
        #
        #    git clone https://github.com/opencv/opencv_extra.git
        #
        # You can find the data in the directory testdata/dnn.
        # The functions readNetFrom...() will look for relative filenames

        # in the current working directory (where the toolbox was started).
    """
    
    _detector = None
    _canvas = None
    _duration = None
    _rects = None

    _dnn: str = None
    _model_file: str = None
    _config_file: str = None

    def __init__(self, dnn='TF', *args, **kwargs):
        """The OpenCV Single Shot MultiBox Detector (SSD).

        Arguments
        ---------
        dnn: str
            The model to use. There are currently two models available:
            'CAFFE' is the original 16-bit floating point model trained
            with Caffe, and 'TF' is a 8-bit quantized version for TensorFlow.
        """
        super().__init__(*args, **kwargs)
        self._dnn = dnn
        self.set_model_file()
        self._add_requirement('cv2', 'module', 'cv2')

    def set_model_file(self, model_file: str=None, config_file: str=None):
        if self._dnn == 'CAFFE':
            model_file = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
            config_file = "deploy.prototxt"
        else:
            model_file = "opencv_face_detector_uint8.pb"
            config_file = "opencv_face_detector.pbtxt"

        if model_file != self._model_file or config_file != self._config_file:
            self._model_file = model_file
            self._config_file = config_file
            self._add_requirement('model_file', 'file', model_file)
            self._add_requirement('config_file', 'file', config_file)

    def _prepare(self):
        if self._dnn == 'CAFFE':
            constructor = cv2.dnn.readNetFromCaffe
        else:
            constructor = cv2.dnn.readNetFromTensorflow

        self._detector = constructor(self._model_file, self._config_file)

    def prepared(self):
        return self._detector is not None

    def detect(self, image):
        """Use the OpenCV
        """

        # The detector expects 3-channel images (BGR!)
        if image.ndim < 3:
            image = np.repeat(image[:,:,np.newaxis], 3, axis=2)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        conf_threshold = 1 # FIXME[hack]
        # the image is converted to a blob
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                     [104, 117, 123], False, False)

        # the blob passed through the network using the forward() function.
        self._detector.setInput(blob)
        detections = self._detector.forward()

        # The output detections is a 4-D matrix, where
        #  * the 3rd dimension iterates over the detected faces.
        #    (i is the iterator over the number of faces)
        #  * the fourth dimension contains information about the
        #    bounding box and score for each face.
        #    - detections[0,0,i,2] gives the confidence score
        #    - detections[0,0,i,3:6] give the bounding box
        #
        # The output coordinates of the bounding box are normalized
        # between [0,1]. Thus the coordinates should be multiplied by
        # the height and width of the original image to get the
        # correct bounding box on the image.
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
        return bboxes
