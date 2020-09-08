"""A MTCCN face detector based on the `mtcnn` module.
"""

# standard imports
from typing import Callable
import logging

# third party imports
import numpy as np
import tensorflow as tf
import mtcnn

# toolbox imports
from base import Preparable
from util.image import BoundingBox
from datasource import Metadata
from dltb.tool.detector import Detector as FaceDetector
from dltb.tool.landmarks import Detector as LandmarkDetector, FacialLandmarks

# logging
LOG = logging.getLogger(__name__)


class FacialLandmarksMTCNN(FacialLandmarks):
    """Landmarks annotation scheme used by the MTCNN detector.
    """

    keypoint_names = ('nose', 'mouth_right', 'right_eye', 'left_eye',
                      'mouth_left')

    def __init__(self, keypoints=None, points=None, **kwargs):
        if keypoints is not None:
            points = np.ndarray((len(self.keypoint_names), 2))
            for i, name in enumerate(self.keypoint_names):
                points[i] = keypoints[name]
        super().__init__(points, **kwargs)


class KerasTensorflowModel(Preparable):
    """A base class for Keras/TensorFlow models. An instance of this class
    represents a dedicated TensorFlow environment consisting of a
    TensorFlow Graph and a TensorFlow Session. The method
    :py:meth:`keras_run` can be used to run a python function in that
    context.

    There are two motivation for establishing a specific session:

    (1) We have to make sure that a TensorFlow model is always used
    with the same tensorflow graph and the same tensorflow session
    in which it was initialized.  Hence we create these beforehand
    (during :py:meth:`prepare`) and store them for later use
    (in model initialization and prediction).

    (2) The aggressive default GPU memory allocation policy of
    TensorFlow: per default, TensorFlow simply grabs (almost) all
    of the available GPU memory. This can cause problems when
    other Tools would also require some GPU memory.

    Private Attributes
    ------------------
    _tf_graph: tf.Graph
    _tf_session: tf.Session

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tf_graph = None
        self._tf_session = None

    def _prepare(self):
        """Initialize the Keras sesssion in which the MTCNN detector is
        executed.


        This method should be called before the first Keras Model is
        created, that is before the MTCNN detector is initialized.

        """
        super()._prepare()

        # The TensorFlow GPU memory allocation policy can be
        # controlled by session parameters. There exist different
        # options, for example:
        #   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        #   config = tf.ConfigProto(gpu_options=gpu_options)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self._tf_graph = tf.Graph()
        with self._tf_graph.as_default():
            self._tf_session = tf.Session(config=config)

        # One may try to set this session as Keras' backend session,
        # but this may be problematic if there is more than one Keras
        # model, each using its own Session. Hence it seems more
        # reliable to explicityl set default graph and default session,
        # as it is done by the method `keras_run()`.
        # backend.set_session(self._tf_session)

    def _unprepare(self) -> None:
        if self._tf_graph is not None:
            with self._tf_graph.as_default():
                if tf._session is not None:
                    self._tf_session.close()
                tf.reset_default_graph()
        self._tf_graph = None
        self._tf_session = None
        super()._unprepare()

    def _prepared(self) -> bool:
        return ((self._tf_graph and self._tf_session) is not None
                and super()._prepared())

    def keras_run(self, function: Callable, *args, **kwargs):
        """Run a python function in the context of the TensorFlow graph and
        session represented by this :py:class:`KerasTensorflowModel`.

        """
        with self._tf_graph.as_default():
            with self._tf_session.as_default():
                result = function(*args, **kwargs)

        return result


class DetectorMTCNN(FaceDetector, LandmarkDetector, KerasTensorflowModel):
    # pylint: disable=too-many-ancestors
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

    Note: in some configurations, there seem to arise issues if
    instantiation of the detector (detector = mtcnn.MTCNN()) and
    invocation (faces = detector.detect_faces(images)) are invoked in
    different Python :py:class:`threading.Thread`s. In fact it seems
    that Keras is only thread-safe if used correctly: initialize your
    model in the same graph and session where you want to do inference
    - then you can run it in multiple/different threads [4].

    [1] https://pypi.org/project/mtcnn/
    [2] https://github.com/ipazc/mtcnn
    [3] https://towardsdatascience.com/how-does-a-face-detection-program-work-using-neural-networks-17896df8e6ff
    [4] https://blog.victormeunier.com/posts/keras_multithread/

    Attributes
    ----------
    _detector: mtcnn.MTCNN
        The actual detector, an instance of the class mtcnn.MTCNN.
    """

    def __init__(self, detect_boxes: bool = True,
                 detect_landmarks: bool = False, **kwargs) -> None:
        """Initialize the :py:class:`DetectorMTCNN` object.  This is a slim
        constructor that allows for quick execution is guaranteed to
        raise no exception. The actual initialization of the detector
        is postpoined to the :py:meth:`prepare` method.
        """
        super().__init__(**kwargs)
        self.detect_boxes = detect_boxes
        self.detect_landmarks = detect_landmarks
        self._detector = None

    def _prepare(self, **kwargs) -> None:
        """Prepare this :py:class:`DetectorMTCNN` face detector. This includes
        setting parameters for the Keras/TensorFlow backend,
        instantiating the :py:class:`mtcnn.MTCNN` class, loading the
        model data, and running a dummy image through it.

        """
        # FIXME[todo]: The actual GPU memory requirement of MTCNN
        # seems to be below 1G. Communicate this to the _prepare() method of
        # the KerasTensorflowModel.
        super()._prepare(**kwargs)
        self.keras_run(self._prepare_detector)

    def _prepare_detector(self):
        """Prepare the MTCNN detector.
        This function should be invoked from a suitable Keras context
        (with controlled TensorFlow Graph and Session), that is
        usually it will be called via :py:meth:`keras_run`.
        """
        # Initialize the MTCNN detector
        detector = mtcnn.MTCNN()

        # The last part of the preparation process of MTCNN is to
        # create the models' (P-net, R-net and O-net) predict
        # functions. It may be possible to achieve this by calling
        # model._make_predict_function() for each of them, but this is
        # discourageds as it uses a private Keras API.  The recomended
        # way to create the predict function is to call predict, as
        # the predict function will be automatically compiled on first
        # invocation.  Hence we provide some dummy image and invoke
        # predict for all three networks by calling
        # detector.detect_faces().
        image = np.random.randint(0, 255, (200, 200, 3), np.uint8)
        _ = detector.detect_faces(image)
        self._detector = detector

    def _prepared(self) -> bool:
        """The DetectorMTCNN is prepared, once the model data
        have been loaded.
        """
        return (self._detector is not None) and super()._prepared()

    def _detect(self, image: np.ndarray, **kwargs) -> Metadata:
        """Apply the MTCNN detector to detect faces in the given image.

        Arguments
        ---------
        image:
            The image to detect faces in. Expected is a RGB image
            with np.uint8 data.

        Returns
        ------
        metadata: Metadata
            A Metadata structure in which BoundingBoxes and
            FacialLandmarks are provided, annotated with a numeric 'id'
            and a 'confidence' value.
        """
        #
        # (1) Run the MTCNN detector
        #
        LOG.info("MTCNN: detecting facess ...")
        faces = self.keras_run(self._detector.detect_faces, image)
        LOG.info("MTCNN: ... found %d faces.", len(faces))

        #
        # (2) Create Metadata
        #
        detections = Metadata(
            description='Detections by the DLib HOG detector')
        for face_id, face in enumerate(faces):
            confidence = face['confidence']

            if self.detect_boxes:
                pos_x, pos_y, width, height = face['box']
                detections.add_region(BoundingBox(x=pos_x, y=pos_y,
                                                  width=width, height=height),
                                      confidence=confidence, id=face_id)

            if self.detect_landmarks:
                detections.add_region(FacialLandmarksMTCNN(face['keypoints']),
                                      confidence=confidence, id=face_id)

        return detections
