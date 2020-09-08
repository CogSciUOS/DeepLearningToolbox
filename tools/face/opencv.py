"""Interface to access face detectors from the OpenCV library.
"""

# standard imports
import os

# third party imports
import cv2
import numpy as np

# toolbox imports
from dltb.tool.detector import ImageDetector as Detector
from datasource import Metadata
from util.image import BoundingBox


class DetectorHaar(Detector):
    # pylint: disable=too-many-ancestors
    """ The OpenCV Haar cascade face detector.

    _model_file: str
        Absolute path to the model file.
    _detector: cv2.CascadeClassifier
        The OpenCV CascadeClassifier
    """

    def __init__(self, *args,
                 model_file='haarcascade_frontalface_default.xml',
                 **kwargs) -> None:
        """The OpenCV Haar cascade face detector.

        Arguments
        ---------
        model_file: str
            The path to the pretrained model file.
        """
        super().__init__(*args, **kwargs)
        self._detector = None
        self._model_file = None
        self.set_model_file(model_file)

    @staticmethod
    def _get_cascade_dir() -> str:
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

    def set_model_file(self, model_file) -> None:
        """Set the model file to be used with this :py:class:`DetectorHaar`.
        """
        if not os.path.isabs(model_file):
            opencv_data = self._get_cascade_dir()
            model_file = os.path.join(opencv_data, model_file)

        if model_file != self._model_file:
            self._model_file = model_file
            self._add_requirement('model_file', 'file', model_file)

    def _prepare(self, **kwargs) -> None:
        """Prepare this :py:class:`DetectorHaar`.
        """
        super()._prepare(**kwargs)
        self._detector = cv2.CascadeClassifier(self._model_file)

        if self._detector.empty():
            self._detector = None
            raise RuntimeError("Haar detector is empty")

    def _unprepare(self) -> None:
        """Free resources occupied by this :py:class:`DetectorHaar`.
        """
        self._detector = None
        super()._unprepare()

    def _prepared(self) -> bool:
        """Check if this :py:class:`DetectorHaar` is prepared.
        """
        return (self._detector is not None) and super()._prepared()

    #
    # Detection
    #

    def _detect(self, image: np.ndarray, **kwargs) -> Metadata:
        """Detect faces using this :py:class:`DetectorHaar` detector.
        """
        # The OpenCV CascadeClassifier expects the input image to
        # be of type CV_8U (meaning unsigned 8-bit, gray scale image
        # with pixel values from 0 to 255). If not given as in this
        # format, we will try to convert it.
        if issubclass(image.dtype.type, np.float):
            image = (image*256).astype(np.uint8)

        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.ndim == 3 and image.shape[2] == 1:
            gray = image[:, :, 0].copy()
        elif image.ndim == 2:
            gray = image.copy()
        else:
            raise ValueError("The image provided has an illegal format: "
                             f"shape={image.shape}, dtype={image.dtype}")

        # There seem to be two Python interfaces to the CascadeClassifier:
        #
        # detectMultiScale(image[,
        #                  scaleFactor[, minNeighbors[, flags[,
        #                  minSize[, maxSize]]]]])
        # detectMultiScale(image, rejectLevels, levelWeights[,
        #                  scaleFactor[, minNeighbors[, flags[,
        #                  minSize[, maxSize[, outputRejectLevels]]]]]])
        #
        # The meaning of the arguments are:
        #
        # scaleFactor – Parameter specifying how much the image size
        #               is reduced at each image scale.
        #
        # minNeighbors – Parameter specifying how many neighbors each
        #                candidate rectangle should have to retain it.
        #
        # flags (old) – Parameter with the same meaning for an old cascade
        #               as in the function cvHaarDetectObjects.
        #               It is not used for a new cascade.
        #
        # minSize - Minimum possible object size.
        #           Objects smaller than that are ignored.
        # maxSize - Maximum possible object size.
        #           Objects larger than that are ignored.
        #
        # rejectLevels, levelWeights, outputRejectLevels: ?
        rects = self._detector.detectMultiScale(gray)

        detections = Metadata(
            description='Detections by the OpenCV CascadeClassifier')
        for rect in rects:
            detections.add_region(BoundingBox(x=rect[0], y=rect[1],
                                              width=rect[2], height=rect[3]))
        return detections


class DetectorSSD(Detector):
    # pylint: disable=too-many-ancestors
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

    _dnn: str = None
    _model_file: str = None
    _config_file: str = None

    def __init__(self, *args, dnn='TF', **kwargs) -> None:
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

    def set_model_file(self, model_file: str = None,
                       config_file: str = None) -> None:
        """Set the model file to be used by this :py:class:`DetectorSSD`.
        """
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

    def _prepare(self, **kwargs) -> None:
        """Prepare this :py:class:`DetectorSSD`.
        This will load the model data from file.
        """
        super()._prepare(**kwargs)
        if self._dnn == 'CAFFE':
            constructor = cv2.dnn.readNetFromCaffe
        else:
            constructor = cv2.dnn.readNetFromTensorflow

        self._detector = constructor(self._model_file, self._config_file)

    def _unprepare(self) -> None:
        """Release resources acquired by this :py:class:`DetectorSSD`.
        This will delete the actual OpenCV model.
        """
        self._detector = None
        super()._unprepare()

    def _prepared(self) -> bool:
        """Check if this :py:class:`DetectorSSD` is prepared.
        """
        return (self._detector is not None) and super()._prepared()

    #
    # Detection
    #

    def _detect(self, image: np.ndarray, **kwargs) -> Metadata:
        """Detect faces with this :py:class:`DetectorSSD`.
        """

        # The detector expects 3-channel images (BGR!)
        if image.ndim < 3:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        # the image is converted to a blob
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                     [104, 117, 123], False, False)

        # the blob passed through the network using the forward() function.
        self._detector.setInput(blob)
        result = self._detector.forward()
        # The result is a 4-D matrix, shape = (1, 1, detections, 7), where
        #  * the 3rd dimension iterates over the detected faces.
        #    (i is the iterator over the number of faces)
        #  * the fourth dimension contains information about the
        #    bounding box and score for each face.
        #    - detections[0,0,i,0] ?
        #    - detections[0,0,i,1] ?
        #    - detections[0,0,i,2] gives the confidence score
        #    - detections[0,0,i,3:6] give the bounding box
        #
        # The output coordinates of the bounding box are normalized
        # between [0,1]. Thus the coordinates should be multiplied by
        # the height and width of the original image to get the
        # correct bounding box on the image.
        #
        # The results are sorted according to the confidence score,
        # that is once we reached our threshold we can stop searching
        # for more detections.

        conf_threshold = .1  # FIXME[hack]
        #frameWidth, frameHeight = 300, 300
        frame_width, frame_height = image.shape[1], image.shape[0]
        # print(image.shape)

        detections = Metadata(
            description='Detections by the OpenCV Deep Neural Network')

        for i in range(result.shape[2]):
            #print(f"{i}: {result[0, 0, i, :]}")
            confidence = result[0, 0, i, 2]
            if confidence < conf_threshold:
                break
            pos_x1 = int(result[0, 0, i, 3] * frame_width)
            pos_y1 = int(result[0, 0, i, 4] * frame_height)
            pos_x2 = int(result[0, 0, i, 5] * frame_width)
            pos_y2 = int(result[0, 0, i, 6] * frame_height)
            detections.add_region(BoundingBox(x1=pos_x1, y1=pos_y1,
                                              x2=pos_x2, y2=pos_y2),
                                  confidence=confidence)
        return detections
