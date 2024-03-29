"""Interface to access face detectors from the dlib library.
"""

# standard imports
import os
import sys
import logging

# third party imports
import numpy as np
import dlib

# toolbox imports
from ..base.metadata import Metadata
from ..base.image import BoundingBox
from ..base.install import Installable
from ..tool.face.detector import Detector as FaceDetector
from ..tool.face.landmarks import (Detector as LandmarkDetector,
                                   FacialLandmarks68)

# logging
LOG = logging.getLogger(__name__)


class DetectorHOG(FaceDetector, Installable):
    # pylint: disable=too-many-ancestors
    """The dlib HOG face detector.

    Attributes
    ----------
    _detector: dlib.fhog_object_detector
    """

    def __init__(self, **kwargs) -> None:
        """Initialize this :py:class:`DetectorHOG`.
        """
        super().__init__(**kwargs)
        self._detector = None

    def _prepare(self, **kwargs) -> None:
        """Prepare this :py:class:`DetectorHOG` by loading the model data.
        """
        super()._prepare(**kwargs)
        self._detector = dlib.get_frontal_face_detector()

    def _unprepare(self):
        """Release the resources acquired by :py:class:`DetectorHOG`.
        """
        self._detector = None
        super()._unprepare()

    def _prepared(self) -> bool:
        """The DetectorHOG is prepared, once the model data
        have been loaded.
        """
        return (self._detector is not None) and super()._prepared()

    def _detect(self, image: np.ndarray, **kwargs) -> Metadata:
        """Apply the dlib histogram of gradients detector (HOG) to
        detect faces in the given image.

        Arguments
        ---------
        image:
            An image in an appropriate format for detection with the
            DLib HOG detector. This means, an `uint8` grayscale or
            RGB image.
        """
        if self._detector is None:
            return None

        LOG.debug("Calling dlib detector with %s of shape %s of type %s",
                  type(image), image.shape, image.dtype)
        # FIXME[hack]: make sure image is in correct format
        # (should be done) by preprocessing ...
        if image.dtype is not np.uint8:
            image = image.astype(np.uint8)

        # dlib: image must be 8bit gray or RGB image.

        # Remarks:
        # 1. Dlib's face detector has no GPU support
        # 2. Detector does not use color information - greyscale images
        #    will work faster
        # 3. For best performance you can downscale images, because dlib
        #    works with small 80x80 faces for detection and use
        #    scanner.set_max_pyramid_levels(1) to exclude scanning of
        #    large faces. And also face detector has 5 cascades inside -
        #    they can be splitted if you need only frontal-looking faces
        #    for example; Unfortunately, scanner.set_max_pyramid_levels is not
        #    available for python API. Splitting cascades can be done in C++
        # 4. GCC-compiled code works much faster than any MSVC.
        #    MSVC 2015 works much faster than MSVC 2013
        # 5. OpenCV's function cv::equalizeHist will make it find more faces

        rects = self._detector(image, 2)

        detections = Metadata(
            description='Detections by the DLib HOG detector')
        for rect in rects:
            detections.add_region(BoundingBox(x=rect.left(), y=rect.top(),
                                              width=rect.width(),
                                              height=rect.height()))
        return detections


class DetectorCNN(FaceDetector, Installable):
    # pylint: disable=too-many-ancestors
    """The dlib CNN detector.
    _model_file: str
    _detector: dlib.cnn_face_detection_model_v1
    """

    def __init__(self, *args, model_file='mmod_human_face_detector.dat',
                 **kwargs) -> None:
        """The OpenCV Single Shot MultiBox Detector (SSD).

        Arguments
        ---------
        dnn: str
            The model to use. There are currently two models available:
            'CAFFE' is the original 16-bit floating point model trained
            with Caffe, and 'TF' is a 8-bit quantized version for TensorFlow.
        """
        super().__init__(*args, **kwargs)
        self._detector = None
        self._model_file = None
        self.set_model_file(model_file)

    def set_model_file(self, model_file) -> None:
        """Set the model file for this :py:class:`DetectorCNN`.
        """
        if not os.path.isabs(model_file):
            dlib_model_directory = os.environ.get('DLIB_MODELS', '.')
            model_file = os.path.join(dlib_model_directory, model_file)

        if model_file != self._model_file:
            self._model_file = model_file
            self._add_requirement('model', 'file', model_file)

    def _prepare(self, **kwargs) -> None:
        """Prepare this :py:class:`DetectorCNN`.
        """
        super()._prepare(**kwargs)
        self._detector = dlib.cnn_face_detection_model_v1(self._model_file)

    def _unprepare(self) -> None:
        """Release resources acquired by this :py:class:`DetectorCNN`.
        """
        self._detector = None
        super()._unprepare()

    def _prepared(self) -> bool:
        """Release resources acquired by this :py:class:`DetectorCNN`.
        """
        return (self._detector is not None) and super()._prepared()

    def _detect(self, image: np.ndarray, **kwargs) -> Metadata:
        """The dlib CNN face detector.
        """
        # The dlib detector expects images to be 8-bit RGB or grayscale
        if image.dtype != np.uint8:
            # FIXME[todo]: better solution: make sure that images are
            # provided in correct format or provide preprocessing ...
            LOG.warning("dlib CNN face detector: image is %s (min=%d, max=%d) "
                        "but should be %s!",
                        image.dtype, image.min(), image.max(), np.uint8)
            image = image.astype(np.uint8)

        # It is also possible to pass a list of images to the
        # detector - like this:
        #   dets = detector([image # list], upsample_num, batch_size = 128)
        # In this case it will return a mmod_rectangless object. This object
        # behaves just like a list of lists and can be iterated over.

        # Remark: in my conda dlib (19.21.0) installation, this is incredible
        # slow, taking 5.3 seconds on a (480, 640, 3) image with 0 upsamples
        # and 83.130 seconds on the same image with 0 upsamples
        # According to the Dlib documentation
        # [http://dlib.net/faq.html#Whyisdlibslow], it is crucial
        # to compile Dlib correctly, that is not in debug mode and
        # with compile optimizations enabled and either
        # SSE4 or AVX instruction turned on.
        #
        # The speed of face detection depends on the the resolution of
        # the image because with smaller resolution images, you look
        # for a smaller range of face sizes. The downside is that you
        # will miss out smaller faces. An easy way to speed up face
        # detection is to resize the frame.
        #
        # You should either install the Intel MKL (or OpenBLAS) to run
        # on the CPU or install CUDA. (Installing OpenBLAS can speed
        # up the process by a factor of 3 or 4)
        #
        # CUDA:
        # conda install -c zeroae dlib-cuda
        #
        # sudo apt-get install libopenblas-dev
        # # + install cudnn (from nvidia)
        # conda create -n faces python=3.6 cudatoolkit cudnn cmake numpy ipython
        # conda activate faces
        # git clone https://github.com/davisking/dlib.git
        # cd dlib
        # mkdir build
        # cd build
        # cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
        # cmake --build .
        # cd ..
        # python setup.py install --set DLIB_USE_CUDA=1

        upsample_num = 0
        import time  # FIXME[hack]
        start = time.time()
        LOG.debug("dlib CNN face detector: detected faces "
                  "in image of shape %s with %d upsamples",
                  image.shape, upsample_num)
        detections = self._detector(image, upsample_num)
        LOG.info("dlib CNN face detector: detected %d faces in %.3f seconds",
                 len(detections), time.time() - start)

        # The result is of type dlib.mmod_rectangles, which is
        # basically a list of rectangles annotated with conficence
        # values.  For an individual detection d (of type
        # dlib.mmode_rectangle), the information can be accessed by
        # d.rect and d.confidence.

        result = Metadata(
            description='Detections by the dlib CNN face detector')
        for detection in detections:
            rect = detection.rect
            result.add_region(self._rectangle_to_bounding_box(rect),
                              confidence=detection.confidence)
        return result

    @staticmethod
    def _rectangle_to_bounding_box(rect: dlib.rectangle) -> BoundingBox:
        return BoundingBox(x=rect.left(), y=rect.top(),
                           width=rect.width(), height=rect.height())


class FacialLandmarkDetector(LandmarkDetector):
    # pylint: disable=too-many-ancestors
    # FIXME[concept]: this class first applies a face detector to find faces
    # in a large image and then applies the landmark detector. This seems
    # to be a commmon situation for which we should provide an API.
    """A facial landmark detector based on dlib.


    Attributes
    ----------
    _detector: dlib.fhog_object_detector
    _predictor: dlib.shape_predictor
    """


    def __init__(self, model_file: str = None, **kwargs):
        # shape_predictor_5_face_landmarks.dat
        # shape_predictor_68_face_landmarks.dat
        super().__init__(**kwargs)
        self._predictor = None
        self._detector = None

        # FIXME[question]: what is going on here?
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

    def _prepare(self, **kwargs) -> None:
        """Prepare this DetectorHOG by loading the model data.
        """
        super()._prepare(**kwargs)
        self._detector = dlib.get_frontal_face_detector()

    def _unprepare(self):
        """Unprepare this DetectorHOG by releasing acquired resources.
        """
        self._detector = None
        super()._unprepare()

    def _prepared(self) -> bool:
        """The DetectorHOG is prepared, once the model data
        have been loaded.
        """
        return (self._detector is not None) and super()._prepared()

    @staticmethod
    def _detection_to_landmarks(detection: dlib.full_object_detection):
        # from imutils.face_utils.shape_to_np:
        # initialize the list of (x, y)-coordinates
        points = np.ndarray((detection.num_parts, 2))  # dtype=dtype

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, detection.num_parts):
            points[i] = (detection.part(i).x, detection.part(i).y)

        # Construct a Metdata object holding the detected landmarks
        return FacialLandmarks68(points)

    #
    # Detection
    #

    def _detect(self, image: np.ndarray, box: BoundingBox = None,
                **kwargs) -> Metadata:
        # pylint: disable=arguments-differ
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
        rect = (dlib.rectangle(0, 0, image.shape[1], image.shape[0])
                if box is None else
                dlib.rectangle(box.x, box.y, box.width, box.height))

        detection = self._predictor(image, rect)

        # detection is of type dlib.full_object_detection, which basically
        # is a list of N points. These can be transformed into
        # a two dimensional array of shape (N,2) for further processing.
        metadata = Metadata(
            description='Facial landmarks detectec by the dlib detctor')
        metadata.add_region(self._detection_landmarks(detection))

        return metadata
