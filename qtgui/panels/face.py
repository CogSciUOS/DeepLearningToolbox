"""
File: face.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
"""
from toolbox import Toolbox, Controller as ToolboxController

from .panel import Panel
from ..utils import QImageView, QObserver
from ..widgets import QModelImageView

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QResizeEvent
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox


class FacePanel(Panel, QObserver, Toolbox.Observer):
    _faceDetector = None
    _toolboxController: ToolboxController = None

    _inputView: QModelImageView = None
    _detectorView: QImageView = None
    _predictorView: QImageView = None

    
    def __init__(self, toolbox: ToolboxController=None, parent=None):
        """Initialization of the AdversarialExamplePanel.

        Parameters
        ----------
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        """
        super().__init__(parent)

        name = 'shape_predictor_5_face_landmarks.dat'
        name = '/space/home/ulf/distro/bionic/lib/python3.6/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat'  # FIXME[hack]
        self._faceDetector = FaceDetector(name)

        self._initUI()
        self._layoutUI()
        self.setToolboxController(toolbox)


    def _initUI(self):
        #
        # Input data
        #

        # QModelImageView: a widget to display the input data
        self._inputView = QModelImageView()
        self._detectorView = QImageView()
        self._predictorView = QImageView()

    def _layoutUI(self):
        layout = QHBoxLayout()
        layout.addWidget(self._inputView)
        layout.addWidget(self._detectorView)
        layout.addWidget(self._predictorView)
        #layout.addStretch()
        self.setLayout(layout)

    def setToolboxController(self, toolbox: ToolboxController) -> None:
        self._exchangeView('_toolboxController', toolbox,
                           interests=Toolbox.Change('input_changed'))
        self._inputView.setToolboxView(toolbox)

    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        if change.input_changed:
            image = toolbox.input_data if toolbox is not None else None
            if image is not None:
                self._faceDetector.process(image)
                self._detectorView.setImage(self._faceDetector._canvas_detect)
                self._predictorView.setImage(self._faceDetector._canvas_predict)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)

import dlib
import cv2
import os
import imutils
import imutils.face_utils


class FaceDetector:

    _detector = None
    _predictor = None

    _canvas_detect = None
    _canvas_predict = None

    _rects = None
    _shapes = None

    # shape_predictor_5_face_landmarks.dat
    # shape_predictor_68_face_landmarks.dat
    def __init__(self, predictor_name: str=None) -> None:
        
        self._hog_detector = dlib.get_frontal_face_detector()
        self._cnn_detector = None

        if not os.path.exists(predictor_name):
            predictor_name = os.path.join(os.environ.get('DLIB_MODELS', '.'),
                                          predictor_name)
        if os.path.exists(predictor_name):
            self._predictor = dlib.shape_predictor(predictor_name)

    def detect(self, image):
        # detect faces in the grayscale image
        self._rects = self._hog_detector(image, 2)
        color = (255, 0, 0)
 
        # check to see if a face was detected, and if so, draw the total
        # number of faces on the image
        if len(self._rects) > 0:
            text = "{} face(s) found".format(len(self._rects))
            if self._canvas_detect is not None:
                cv2.putText(self._canvas_detect, text,
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
        # loop over the face detections
        if self._canvas_detect is not None:
            for rect in self._rects:
                # rect is of type <class 'dlib.rectangle'>
        
                # compute the bounding box of the face and draw it on the
                # image
                (bX, bY, bW, bH) = imutils.face_utils.rect_to_bb(rect)
                cv2.rectangle(self._canvas_detect,
                              (bX, bY), (bX + bW, bY + bH), color, 1)

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

    def process(self, image):
        image = imutils.resize(image, width=400)
        
        if image.ndim == 3 and image.shape[2] == 3:
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        self._canvas_detect = image.copy()
        self.detect(gray)

        self._canvas_predict = image.copy()
        self.predict(gray, self._rects)

