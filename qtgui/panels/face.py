"""
File: face.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de

Graphical interface for face detection and recognition.
"""

from toolbox import Toolbox, Controller as ToolboxController
from datasources import Datasource, Controller as DatasourceController
from tools.face.detector import Detector, FaceDetector, FaceController

from .panel import Panel
from ..utils import QImageView, QObserver
from ..widgets import QModelImageView
from ..widgets.datasource import QLoopButton

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QResizeEvent, QHideEvent, QShowEvent
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QGroupBox,
                             QVBoxLayout, QHBoxLayout, QGridLayout)



class FacePanel(Panel, QObserver, Toolbox.Observer, FaceDetector.Observer):
    """The FacePanel provides access to different face recognition
    technologies. This includes
    * face detection
    * face landmarking
    * face alignment
    * face recogntion

    Face detection
    --------------
    * Apply face detector to some data source
    * Compare multiple face detectors
    * Evaluate face detectors
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
        self._faceDetector = FaceDetector()   # FIXME[hack]
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
        """Initialize the user interface

        The user interface contains the following elements:
        * the input view: depicting the current input image
        * a loop button: allowing to start and stop loop data sources
        * an input counter:
        * a process counter:
        * up to four detector views: depicting faces located in the input image
        """
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
        """Initialize the user interface
        """
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

        grid = QGridLayout()
        grid.addWidget(self._detectorWidget('Haar Cascade',
                                            self._detectorView1), 0,0)
        grid.addWidget(self._detectorWidget('HOG',
                                            self._detectorView2), 0, 1)
        grid.addWidget(self._detectorWidget('DNN',
                                            self._detectorView3), 1, 0)
        grid.addWidget(self._detectorWidget('Predictor',
                                            self._predictorView), 1, 1)
        layout.addLayout(grid)

        self.setLayout(layout)

    def _detectorWidget(self, name: str, widget: QWidget):
        layout = QVBoxLayout()
        layout.addWidget(widget)
        layout.addWidget(QLabel(name))
        groupBox = QGroupBox(name)
        groupBox.setLayout(layout)
        groupBox.setCheckable(True)
        return groupBox

    def setToolboxController(self, toolbox: ToolboxController) -> None:
        """Set a new ToolboxController.
        We are only interested in changes of the input data.
        """
        # FIXME[concept]: do we need a ToolboxController or would
        # a ToolboxView be sufficient?
        interests = Toolbox.Change('input_changed')
        self._exchangeView('_toolboxController', toolbox, interests=interests)
        if not self.isVisible(): # FIXME[hack]: we have to consistently integrate the visibility into the notification logic
            self._deactivateView('_toolboxController')
        self._inputView.setToolboxView(toolbox)

    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        """The FacePanel is a Toolbox.Observer. It is interested
        in input changes and will react with applying face recognition
        to a new input image.
        """
        print("Toolbox changed:", change)
        if change.input_changed and self._faceController is not None:
            image = toolbox.input_data if toolbox is not None else None
            self._faceController.process(image)
            self._inputCounter.setText(str(int(self._inputCounter.text())+1))

    def setDatasourceController(self, datasource: DatasourceController) -> None:
        """Set a :py:class:`DatasourceController` for this
        :py:class:`FacePanel`. We are not really interested in the
        Datasource but only forward this to subcomponents.
        """
        self._loopButton.setDatasourceController(datasource)

    def setFaceController(self, face: FaceController) -> None:
        """Set a new :py:class:`FaceController`.
        The face Controller will inform us whenever new faces where
        detected by one of the detectors.
        """
        interests = Toolbox.Change('hog_changed', 'cnn_changed',
                                   'haar_changed', 'predict_changed')
        self._exchangeView('_faceController', face, interests=interests)

    def face_changed(self, detector: FaceDetector,
                     change: FaceDetector.Change) -> None:
        print("face changed:", change)
        if change.haar_changed:
            # FIXME[hack]: private variables!
            self._detectorView1.setImage(detector._canvas_detect_haar)
            if detector._canvas_detect_haar is not None:
                print(detector._canvas_detect_haar.shape)
        if change.hog_changed:
            # FIXME[hack]: private variables!
            self._detectorView2.setImage(detector._canvas_detect_hog)
        if change.cnn_changed:
            # FIXME[hack]: private variables!
            self._detectorView2.setImage(detector._canvas_detect_cnn)
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
