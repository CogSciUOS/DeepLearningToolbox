
import os
import numpy as np

import tensorflow as tf
from keras import backend
import mtcnn

from .detector import Detector as FaceDetector
from .landmarks import Detector as LandmarkDetector, FacialLandmarks
from datasource import Metadata
from util.image import BoundingBox
from util.error import handle_exception

class FacialLandmarksMTCNN(FacialLandmarks):

    keypoint_names = ('nose', 'mouth_right', 'right_eye', 'left_eye',
                      'mouth_left')
    
    def __init__(self, keypoints=None, points=None, **kwargs):
        if keypoints is not None:
            points = np.ndarray((len(self.keypoint_names),2))
            for i, name in enumerate(self.keypoint_names):
                points[i] = keypoints[name]
        super().__init__(points, **kwargs)

class KerasTensorflowModel:
    
    def keras_prepare(self):
        """Initialize the Keras sesssion in which the MTCNN detector is
        executed.

        There are two motivation for establishing a specific session:
        
        (1) We have to make sure that the MTCNN model is always used
        with the same tensorflow graph and the same tensorflow session
        in which it was initialized.  Hence we create these beforehand
        and store them for later use (in initialization and
        prediction).

        (2) The aggressive default GPU memory allocation policy of
        TensorFlow: per default, TensorFlow simply grabs (almost) all
        of the available GPU memory. This can cause problems when
        other Tools would also require some GPU memory.  The actual
        GPU memory requirement of MTCNN seems to be below 1G.

        This method should be called before the first Keras Model is
        created, that is before the MTCNN detector is initialized.

        """
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
            print("SETTING self._tf_session")

        # One may try to set this session as Keras' backend session,
        # but this may be problematic if there are more than one Keras
        # model, each using its own Session. Hence it seems more
        # reliable to explicityl set default graph and default session,
        # as it is done by the method `keras_run()`.
        #backend.set_session(self._tf_session)

    def keras_run(self, function, *args, **kwargs):
        with self._tf_graph.as_default():
            with self._tf_session.as_default():
                result = function(*args, **kwargs)
        
        return result

class DetectorMTCNN(FaceDetector, LandmarkDetector, KerasTensorflowModel):
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

    def __init__(self, detect_boxes: bool=True, detect_landmarks: bool=False,
                 **kwargs) -> None:
        """Initialize the :py:class:`DetectorMTCNN` object.  This is a slim
        constructor that allows for quick execution is guaranteed to
        raise no exception. The actual initialization of the detector
        is postpoined to the :py:meth:`prepare` method.
        """
        super().__init__(**kwargs)
        self.detect_boxes = detect_boxes
        self.detect_landmarks = detect_landmarks
        self._detector = None

    def _prepare(self) -> None:
        """Prepare this DetectorMTCNN 
        """
        """Prepare the MTCNN face detector. This includes setting parameters
        for the Keras/TensorFlow backend, instantiating the
        :py:class:`mtcnn.MTCNN` class, loading the model data, and
        running a dummy image through it.
        """
        self.keras_prepare()
        self.keras_run(self._prepareDetector)

    def _prepareDetector(self):
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
        image = np.random.randint(0, 255, (200 ,200, 3), np.uint8)
        ignore = detector.detect_faces(image)
        self._detector = detector
        
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
            The image to detect faces in. Expected is a RGB image
            with np.uint8 data.

        Result
        ------
        metadata: Metadata
            A Metadata structure in which BoundingBoxes and
            FacialLandmarks are provided, annotated with a numeric 'id'
            and a 'confidence' value.
        """
        #
        # (1) Run the MTCNN detector
        #
        try:
            faces = self.keras_run(self._detector.detect_faces, image)
        except BaseException as exception:  # InternalError
            print(f"MTCNN[{threading.currentThread().getName()}]: _detect:"
                  "  ... error during detection.")
            handle_exception(exception)
            raise RuntimeError("Detecting faces with the MTCNN detector failed.")

        #
        # (2) Create Metadata
        #
        detections = Metadata(description=
                              'Detections by the DLib HOG detector')
        for id, face in enumerate(faces):
            confidence = face['confidence']

            if self.detect_boxes:
                x, y, w, h = face['box']
                detections.add_region(BoundingBox(x=x, y=y, width=w, height=h),
                                      confidence=confidence, id=id)

            if self.detect_landmarks:
                detections.add_region(FacialLandmarksMTCNN(face['keypoints']),
                                      confidence=confidence, id=id)        

        return detections
