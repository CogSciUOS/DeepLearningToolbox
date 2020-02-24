import os
import sys
import time
import imutils
import imutils.face_utils
import numpy as np

from base.observer import Observable, change


class Detector(Observable, method='detector_changed',
               changes=['image_changed', 'detection_finished']):
    """

    image_changed:
        The input image given to the detector has changed.

    detection_finished:
        The detection has finished. This is reported when the
        detector has finished its work.

    """

    _requirements = None
    
    def __init__(self):
        super().__init__()
        self._requirements = {}
        self._image = None
        self._canvas = None
        self._duration = 0
        self._rects = []

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
        """Do the actual detection.
        """
        raise NotImplementedError("Face detection for class '" +
                                  type(self).__name__ +
                                  "' is not implemented (yet).")


    def process(self, gray: np.ndarray, canvas: np.ndarray=None):
        self._image = gray
        self.change(image_changed=True)

        start = time.time()
        self._rects = self.detect(gray)
        print(self._rects)
        end = time.time()
        self._duration = end - start

        #if canvas is not None:
        #    self.paint_detect(canvas, self._rects)

        self._canvas = canvas
        self.change(detection_finished=True)

    @property
    def image(self):
        return self._image
        
    @property
    def canvas(self):
        return self._canvas

    @property
    def rects(self):
        """Bounding boxes are stored as quadruples
        (x,y,w,h).

        FIXME[question]: is this int or do we allow float for subpixel accuracy?
        FIXME[todo]: currently they are also sometimes stored as
        dlib.rectangle
        """
        return self._rects
    
    @property
    def duration(self):
        return self._duration

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



from base import View as BaseView, Controller as BaseController, run

class DetectorView(BaseView, view_type=Detector):
    """Viewer for :py:class:`Engine`.

    Attributes
    ----------
    _engine: Engine
        The engine controlled by this Controller

    """

    def __init__(self, engine: Detector=None, **kwargs):
        super().__init__(observable=engine, **kwargs)


class DetectorController(DetectorView, BaseController):
    """Controller for :py:class:`Engine`.
    This class contains callbacks for all kinds of events which are
    effected by the user in the ``MaximizationPanel``.

    Attributes
    ----------
    _engine: Engine
        The engine controlled by this Controller
    """
    
    def __init__(self, engine: Detector, **kwargs) -> None:
        """
        Parameters
        ----------
        engine: Engine
        """
        super().__init__(engine=engine, **kwargs)
        self._busy = False
        self._next_image = None

    def process(self, image):
        """Process the given image.

        """
        self._next_image = image
        if not self._busy:
            self._process()

    @run
    def _process(self):
        self._busy = True
        while self._next_image is not None:
            image = imutils.resize(self._next_image, width=400)
            self._next_image = None
            self._detector.process(image, image)

        self._busy = False


def create_detector(name: str, prepare: bool=True):
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
        from .dlib import DetectorHOG
        detector = DetectorHOG()
    else:
        raise ValueError("face.create_detector: "
                         f"unknown detector name '{name}'.")

    if prepare:
        detector.prepare()
    return detector




### Old





import cv2
import dlib

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
        return self._next_image is not None

    def queue(self, image):
        self._next_image = image


    def process(self, image):
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
        

