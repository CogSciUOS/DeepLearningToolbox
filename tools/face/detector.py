import sys
import numpy as np

# FIXME[todo]: import only if needed to speed up import of this module
import dlib
import cv2
import time

class Detector:

    _requirements = None
    
    def __init__(self):
        self._requirements = {}

    def _add_requirement(self, name, what, *data):
        self._requirements[name] = (what,) + data

    def _remove_requirement(self, name):
        self._requirements.pop(name, None)


    def available(self, verbose=True):
        """Check if required resources are available.
        """
        for name, requirement in self._requirements.items():
            if requirement[0] == 'file':
                if not os.path.exists(requirement[1]):
                    if verbose:
                        print(type(self).__name__ +
                              f": File '{requirement[1]}' not found")
                    return False
            if requirement[0] == 'module':
                if requirement[1] in sys.modules:
                    continue
                spec = importlib.util.find_spec(requirement[1])
                if spec is None:
                    print(type(self).__name__ +
                          f": Module '{requirement[1]}' not found")
                    return False
        return True
        
    def install(self):
        """Install the resources required for this module.
        """
        raise NotImplementedError("Installation of resources for '" +
                                  type(self).__name__ +
                                  "' is not implemented (yet).")

    def prepare(self, install: bool=False):
        """Load the required resources.
        """
        if self.prepared():
            return

        # FIXME[concept]:
        # In some situations, one requirement has to be prepared in
        # order to check for other requirements.
        # Example: checking the availability of an OpenCV data file
        # may require the 'cv2' module to be loaded in order to construct
        # the full path to that file.

        for name, requirement in self._requirements.items():
            if requirement[0] == 'module' and requirement[1] not in globals():
                globals()[requirement[1]] = \
                    importlib.import_module(requirement[1])

        if not self.available(verbose=True):
            if install:
                self.install()
            else:
                raise RuntimeError("Resources required to prepare '" +
                                   type(self).__name__ +
                                   "' are not installed.")
        self._prepare()

    def _prepare(self):
        pass

    def prepared(self):
        return True

    def detect(self, image: np.ndarray):
        raise NotImplementedError("Face detection for class '" +
                                  type(self).__name__ +
                                  "' is not implemented (yet).")

    def process(self, gray: np.ndarray, canvas: np.ndarray=None):
        start = time.time()
        rects = self.detect(gray)
        end = time.time()
        self._duration = end - start
        if canvas is not None:
            self.paint_detect(canvas, rects)

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
        if len(rects) > 0:
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

        
class DetectorOpencvHaar(Detector):
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
        if self._detector is None:
            return None
        
        rects = self._detector.detectMultiScale(image)
        return rects


class DetectorOpencvSSD(Detector):
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
        # the image is converted to a blob
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300),
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

class DetectorDlibHOG(Detector):
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

class DetectorDlibCNN(Detector):
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

### Old



import dlib
import cv2
import os
import imutils
import imutils.face_utils
import numpy as np

from base.observer import Observable, change

from tools.face.detector import Detector



class FaceDetector(Observable, method='face_changed',
                   changes=['hog_changed', 'cnn_changed', 'haar_changed',
                            'predict_changed', 'image_changed']):
    """The face detector is an Observable that will inform the method
    face_changed.


    Attributes:

    _hog_detector:
    _cnn_detector:
    _haar_detector:
    _ssd_detector:
    _predictor:
    """

    _hog_detector = None
    _cnn_detector = None
    _ssd_detector = None
    _predictor = None

    _canvas_detect_cnn = None
    _canvas_detect_hog = None
    _canvas_detect_haar = None
    _canvas_predict = None

    _detectors: dict = {}

    _image: np.ndarray = None
    _next_image: np.ndarray = None

    def __init__(self) -> None:
        super().__init__()

        self.activate('haar')
        #self.activate('ssd')
        #self.activate('hog')
        #self.activate('cnn')

        for detector in self._detectors.values():
            detector.prepare()
            
    def activate(self, name, state=True):
        if not state:
            self.deactivate(name)
        if name in self._detectors:
            return
        if name == 'haar': 
            self._detectors[name] = DetectorOpencvHaar()
        elif name == 'ssd':
            self._detectors[name] = DetectorOpencvSSD()
        elif name == 'hog':
            self._detectors[name] = DetectorDlibHOG()
        elif name == 'cnn':
            self._detectors[name] = DetectorDlibCNN()
        else:
            raise ValueError(f"Invalid detector name: '{name}'."
                             "Known detectors are: " + 
                             ', '.join('haar', 'ssd', 'hog', 'cnn'))

    def deactivate(self, name, state=True):
        self._detectors.pop(name, None)

    def busy(self):
        return self._image is not None

    def queue(self, image):
        self._next_image = image


    def process(self, image):
        print("FaceDetector.process")
        self._image = image

        while self._image is not None:

            image = imutils.resize(image, width=400)
        
            if image.ndim == 3 and image.shape[2] == 3:
                # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            for name, detector in self._detectors.items():
                canvas = image.copy()
                if name == 'hog':
                    self._canvas_detect_hog = canvas
                    detector.process(gray, canvas=canvas)
                    self.change(hog_changed=True)
                elif name == 'cnn':
                    self._canvas_detect_cnn = canvas
                    detector.process(gray, canvas=canvas)
                    self.change(cnn_changed=True)
                elif name == 'haar':
                    self._canvas_detect_haar = canvas
                    detector.process(gray, canvas=canvas)
                    self.change(haar_changed=True)

            self._image = self._next_image
            self._next_image = None

        print("FaceDetector.process [end]")

    #
    # OLD: predictors
    #
    _predictors: dict = None
    
    def __init__old__(self, predictor_name: str=None):
        # shape_predictor_5_face_landmarks.dat
        # shape_predictor_68_face_landmarks.dat
        self._predictors = {}
        
        #
        # The dlib facial landmark detector
        #
        if not os.path.exists(predictor_name):
            predictor_name = os.path.join(os.environ.get('DLIB_MODELS', '.'),
                                          predictor_name)
        if os.path.exists(predictor_name):
            self._predictor = dlib.shape_predictor(predictor_name)
            self.add_predictor('predict')
        else:
            print(f"FIXME: not found '{predictor_name}'")

    def add_predictor(self, name):
        if name in self._predictors:
            self._predictors[name] += 1
        else:
            self._predictors[name] = 1

    def remove_predictor(self, name):
        if name in self._predictors:
            if self._predictors[name] > 1:
                self._predictors[name] -= 1
            else:
                del self._predictors[name]

    def process_predictor(self):

        # Predictors require detectors to be run first ...
        predictors = list(self._predictors.keys())
        for predictor in predictors:
            if predictor == 'predict':
                self._canvas_predict = image.copy()
                self.predict(gray, self._cnn_rects)  # includes painting
                self.change(predict_changed=True)

    def predict(self, image, rects):
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

from base import View as BaseView, Controller as BaseController, run

class View(BaseView, view_type=FaceDetector):
    """Viewer for :py:class:`Engine`.

    Attributes
    ----------
    _engine: Engine
        The engine controlled by this Controller

    """

    def __init__(self, engine: FaceDetector=None, **kwargs):
        super().__init__(observable=engine, **kwargs)


class FaceController(View, BaseController):
    """Controller for :py:class:`Engine`.
    This class contains callbacks for all kinds of events which are
    effected by the user in the ``MaximizationPanel``.

    Attributes
    ----------
    _engine: Engine
        The engine controlled by this Controller
    """

    def __init__(self, engine: FaceDetector, **kwargs) -> None:
        """
        Parameters
        ----------
        engine: Engine
        """
        super().__init__(engine=engine, **kwargs)

    def process(self, image):
        """Process the given image.

        """
        print(f"FaceController.process [start]: {self._facedetector.busy()}")
        if self._facedetector.busy():
            self._facedetector.queue(image)
        else:
            self._process(image)
        print("FaceController.process [end]")

    @run
    def _process(self, image):
        self._facedetector.process(image)
        

