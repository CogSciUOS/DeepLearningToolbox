"""A MTCCN face detector based on the `mtcnn` module.

The `mtcnn` module can be installed with pip

  pip install mtcnn

or with conda

  conda install -c conda-forge mtcnn

The `mtcnn` implementation is based on TensorFlow.

"""

# standard imports
import logging

# third party imports
import numpy as np
import mtcnn

# toolbox imports
from dltb.tool.face.mtcnn import Detector as BaseDetector, Landmarks
from dltb.base.meta import Metadata
from dltb.base.image import BoundingBox
from .tensorflow.keras import KerasTensorflowModel

# logging
LOG = logging.getLogger(__name__)


class Detector(BaseDetector, KerasTensorflowModel):
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
    [3] https://towardsdatascience.com/
      how-does-a-face-detection-program-work-using-neural-networks-17896df8e6ff
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
        self.run_tensorflow(self._prepare_detector)

    def _prepare_detector(self):
        """Prepare the MTCNN detector.
        This function should be invoked from a suitable Keras context
        (with controlled TensorFlow Graph and Session), that is
        usually it will be called via :py:meth:`run_tensorflow`.
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
        faces = self.run_tensorflow(self._detector.detect_faces, image)
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
                detections.add_region(Landmarks(face['keypoints']),
                                      confidence=confidence, id=face_id)

        return detections
