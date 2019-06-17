"""
File: face.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
"""


import dlib
import cv2
import os
import imutils
import imutils.face_utils
import numpy as np

from base.observer import Observable, change

class FaceDetector(Observable, method='face_changed',
                   changes=['hog_changed', 'cnn_changed', 'haar_changed',
                            'predict_changed', 'image_changed']):

    _hog_detector = None
    _cnn_detector = None
    _haar_detector = None
    _ssd_detector = None
    _predictor = None

    _canvas_detect_cnn = None
    _canvas_detect_hog = None
    _canvas_detect_haar = None
    _canvas_predict = None

    _hog_duration = 0
    _cnn_duration = 0
    _haar_duration = 0

    _hog_rects = None
    _cnn_rects = None
    _haar_rects = None
    _shapes = None

    _detectors: dict = None
    _predictors: dict = None


    _image: np.ndarray = None
    _next_image: np.ndarray = None

    # shape_predictor_5_face_landmarks.dat
    # shape_predictor_68_face_landmarks.dat
    def __init__(self, predictor_name: str=None) -> None:
        super().__init__()

        self._detectors = {}
        self._predictors = {}

        #
        # The dlib HOG detector
        #      
        self._hog_detector = dlib.get_frontal_face_detector()
        self.add_detector('hog')

        #
        # The dlib CNN detector
        #
        cnn_detector_name = 'mmod_human_face_detector.dat'
        detector_model = os.path.join(os.environ.get('DLIB_MODELS', '.'),
                                      cnn_detector_name)
        if os.path.exists(detector_model):
            self._cnn_detector = \
                dlib.cnn_face_detection_model_v1(detector_model)
            self.add_detector('cnn')
        else:
            print(f"FIXME: not found '{detector_model}'")

        #
        # The OpenCV Haar cascade face detector
        #
        # FIXME[hack]: make this more flexible ...
        faceCascadePath = '/space/home/ulf/distro/bionic/software/opencv4/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        if os.path.exists(faceCascadePath):
            self._haar_detector = cv2.CascadeClassifier(faceCascadePath)
            print(self._haar_detector.empty())
            if self._haar_detector.empty():
                self._haar_detector = None
            else:
                self.add_detector('haar')


        #
        # The OpenCV Single Shot MultiBox Detector (SSD)
        DNN = "TF"
        if DNN == "CAFFE":
            modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
            configFile = "deploy.prototxt"
            self._ssd_detector = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        else:
            modelFile = "opencv_face_detector_uint8.pb"
            configFile = "opencv_face_detector.pbtxt"
            self._ssd_detector = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

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


    def add_detector(self, name):
        if name in self._detectors:
            self._detectors[name] += 1
        else:
            self._detectors[name] = 1

    def remove_detector(self, name):
        if name in self._detectors:
            if self._detectors[name] > 1:
                self._detectors[name] -= 1
            else:
                del self._detectors[name]

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

    def detect_haar(self, image):
        if self._haar_detector is None:
            return None
        
        rects = self._haar_detector.detectMultiScale(image)
        return rects

    def detect_hog(self, image):
        """Apply the dlib historgram of gradients detector (HOG) to
        detect faces in the given image.
        """
        if self._hog_detector is None:
            return None

        rects = self._hog_detector(image, 2)
        return rects

    def detect_cnn(self, image):
        """The dlib CNN face detector.
        """
        if self._cnn_detector is None:
            return None

        dets = self._cnn_detector(image, 2)

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

    def detect_ssd(self, image):
        """Use the OpenCV
        """
        # the image is converted to a blob
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300),
                                     [104, 117, 123], False, False)

        # the blob passed through the network using the forward() function.
        self._ssd_detector.setInput(blob)
        detections = self._ssd_detector.forward()

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

    def busy(self):
        return self._image is not None

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

            detectors = list(self._detectors.keys())
            for detector in detectors:
                if detector == 'hog':
                    start = time.time()
                    self._hog_rects = self.detect_hog(gray)
                    end = time.time()
                    self._hog_duration = end - start
                    self._canvas_detect_hog = image.copy()
                    self.paint_detect(self._canvas_detect_hog, self._hog_rects)
                    self.change(hog_changed=True)
                elif detector == 'cnn':
                    start = time.time()
                    self._cnn_rects = self.detect_cnn(gray)
                    end = time.time()
                    self._cnn_duration = end - start
                    self._canvas_detect_cnn = image.copy()
                    self.paint_detect(self._canvas_detect_cnn, self._cnn_rects)
                    self.change(cnn_changed=True)
                elif detector == 'haar':
                    start = time.time()
                    self._haar_rects = self.detect_haar(gray)
                    end = time.time()
                    self._haar_duration = end - start
                    self._canvas_detect_haar = image.copy()
                    self.paint_detect(self._canvas_detect_haar,
                                      self._haar_rects, color=(0, 255, 0))
                    self.change(haar_changed=True)

            # Predictors require detectors to be run first ...
            predictors = list(self._predictors.keys())
            for predictor in predictors:
                if predictor == 'predict':
                    self._canvas_predict = image.copy()
                    self.predict(gray, self._cnn_rects)  # includes painting
                    self.change(predict_changed=True)

            self._image = self._next_image
            self._next_image = None

    def paint_detect(self, canvas, rects, color=(255, 0, 0)):
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
                    (bX, bY, bW, bH) = imutils.face_utils.rect_to_bb(rect)
                    # rect is of type <class 'dlib.rectangle'>
                else:
                    (bX, bY, bW, bH) = rect
                cv2.rectangle(canvas, (bX, bY), (bX + bW, bY + bH), color, 1)

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
        if self._facedetector.busy():
            self._facedetector.queue(image)
        else:
            self._process(image)

    @run
    def _process(self, image):
        self._facedetector.process(image)





from toolbox import Toolbox, Controller as ToolboxController
from datasources import Controller as DatasourceController

from .panel import Panel
from ..utils import QImageView, QObserver
from ..widgets import QModelImageView
from ..widgets.datasource import QLoopButton

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QResizeEvent, QHideEvent, QShowEvent
from PyQt5.QtWidgets import (QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QGroupBox)



class FacePanel(Panel, QObserver, Toolbox.Observer, FaceDetector.Observer):
    """The FacePanel provides access to different face recognition
    technologies. This includes
    * face detection
    * face landmarking
    * face alignment
    * face recogntion

    """
    _faceDetector = None

    _toolboxController: ToolboxController = None
    _faceController: FaceController = None

    _inputView: QModelImageView = None
    _detectorView1: QImageView = None
    _detectorView2: QImageView = None
    _predictorView: QImageView = None

    _inputCounter: QLabel = None
    _processCounter: QLabel = None
    _loopButton: QPushButton = None
   

    def __init__(self, toolbox: ToolboxController=None,
                 datasource: DatasourceController=None, parent=None):
        """Initialization of the FacePanel.

        Parameters
        ----------
        toolbox: ToolboxController
            The toolbox controller provides input data.
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        """
        super().__init__(parent)

        name = 'shape_predictor_5_face_landmarks.dat'
        name = 'shape_predictor_68_face_landmarks.dat'  # FIXME[hack]
        self._faceDetector = FaceDetector(name)   # FIXME[hack]
        faceController = FaceController(self._faceDetector)   # FIXME[hack]
        if toolbox is not None:
            faceController.runner = toolbox.runner  # FIXME[hack]

        self._initUI()
        self._layoutUI()
        self.setToolboxController(toolbox)
        self.setDatasourceController(datasource)
        self.setFaceController(faceController)

        self._counter = 0  # FIXME[hack]

    def _initUI(self):
        #
        # Input data
        #

        # QModelImageView: a widget to display the input data
        self._inputView = QModelImageView()

        self._inputCounter = QLabel("0")
        self._processCounter = QLabel("0")

        self._loopButton = QLoopButton('Loop')

        self._detectorView1 = QImageView()
        self._detectorView2 = QImageView()
        self._detectorView3 = QImageView()
        self._predictorView = QImageView()

    def _layoutUI(self):
        layout = QHBoxLayout()

        layout2 = QVBoxLayout()
        layout2.addWidget(self._inputView)
        row = QHBoxLayout()
        row.addWidget(self._processCounter)
        row.addWidget(QLabel("/"))
        row.addWidget(self._inputCounter)
        row.addStretch()
        row.addWidget(self._loopButton)
        layout2.addLayout(row)
        layout2.addStretch()
        layout.addLayout(layout2)

        layout2 = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(self._detectorView1)
        row.addWidget(self._detectorView2)
        layout2.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self._detectorView3)
        row.addWidget(self._predictorView)
        row.addStretch()
        layout2.addLayout(row)
        layout2.addStretch()
        layout.addLayout(layout2)

        self.setLayout(layout)

    def setToolboxController(self, toolbox: ToolboxController) -> None:
        interests = Toolbox.Change('input_changed')
        self._exchangeView('_toolboxController', toolbox, interests=interests)
        if not self.isVisible(): # FIXME[hack]: we have to consistently integrate the visibility into the notification logic
            self._deactivateView('_toolboxController')
        self._inputView.setToolboxView(toolbox)

    def setDatasourceController(self, datasource: DatasourceController) -> None:
        self._loopButton.setDatasourceController(datasource)

    def setFaceController(self, face: FaceController) -> None:
        interests = Toolbox.Change('hog_changed', 'cnn_changed',
                                   'haar_changed', 'predict_changed')
        self._exchangeView('_faceController', face, interests=interests)

    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        """The FacePanel is a Toolbox.Observer. It is interested
        in input changes and will react with applying face recognition
        to a new input image.
        """
        if change.input_changed and self._faceController is not None:
            image = toolbox.input_data if toolbox is not None else None
            self._faceController.process(image)
            self._inputCounter.setText(str(int(self._inputCounter.text())+1))

            
    def face_changed(self, detector: FaceDetector,
                     change: FaceDetector.Change) -> None:
        if change.hog_changed:
            # FIXME[hack]: private variables!
            self._detectorView1.setImage(detector._canvas_detect_hog)
        if change.cnn_changed:
            # FIXME[hack]: private variables!
            self._detectorView2.setImage(detector._canvas_detect_cnn)
        if change.haar_changed:
            # FIXME[hack]: private variables!
            self._detectorView3.setImage(detector._canvas_detect_haar)
        if change.predict_changed:
            # FIXME[hack]: private variables!
            self._predictorView.setImage(detector._canvas_predict)
            self._processCounter.setText(str(int(self._processCounter.text())+1))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)


    def hideEvent(self, event: QHideEvent) -> None:
        self._deactivateView('_toolboxController')
        print("FacePanel is now invisible.")

    def showEvent(self, event: QShowEvent) -> None:
        interests = Toolbox.Change('input_changed')
        self._activateView('_toolboxController', interests)
        print("FacePanel is now visible.")
