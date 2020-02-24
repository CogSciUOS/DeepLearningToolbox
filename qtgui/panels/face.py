"""
File: face.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de

Graphical interface for face detection and recognition.
"""

from tools.face.detector import (Detector, DetectorController)
from ..utils import QImageView, QObserver

import numpy as np

from PyQt5.QtWidgets import (QGroupBox, QWidget, QLabel, QVBoxLayout)

class DetectorWidget(QGroupBox, QObserver, Detector.Observer):
    """A detector widget displays the output of a Detector.
    """

    _detectorController: DetectorController = None
    _view: QImageView = None
    _label: QLabel = None

    def __init__(self, detector: DetectorController=None, **kwargs):
        """Initialization of the FacePanel.

        Parameters
        ----------
        decector: DetectorController
            The face detector providing data.
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        """
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()
        self.setDetectorController(detector)

    def _initUI(self):
        """Initialize the user interface

        The user interface contains the following elements:
        * the input view: depicting the current input image
        * a loop button: allowing to start and stop loop data sources
        * an input counter:
        * a process counter:
        * up to four detector views: depicting faces located in the input image
        """
        self._view = QImageView()
        self._label = QLabel()

    def _layoutUI(self):
        layout = QVBoxLayout()
        layout.addWidget(self._view)
        layout.addWidget(self._label)
        layout.addStretch(3)
        self.setLayout(layout)
        self.setCheckable(True)

    def setDetectorController(self, detector: DetectorController) -> None:
        """Set a new :py:class:`DetectorController`.
        The face Controller will inform us whenever new faces where
        detected by one of the detectors.
        """
        interests = Detector.Change('detection_finished')
        self._exchangeView('_detectorController', detector,
                           interests=interests)
        self.setTitle("None" if self._detectorController is None else
                      (self._detectorController._detector.__class__.__name__
                       if self._detectorController else "Off"))  # FIXME[hack]

    def detector_changed(self, detector: Detector,
                         change: Detector.Change) -> None:
        if change.detection_finished:
            self._view.setImage(detector.canvas)
            self._label.setText(f"{detector.duration:.3f}s")

    def setImage(self, image: np.ndarray):
        if self._detectorController is None or not self._detectorController:
            self._view.setImage(None)
        else:
            self._detectorController.process(image)

from toolbox import Toolbox, Controller as ToolboxController
from datasources import Datasource, Controller as DatasourceController
from tools.face.detector import Detector, create_detector

from .panel import Panel
from ..utils import QImageView, QObserver
from ..widgets import QModelImageView
from ..widgets.datasource import QLoopButton, QSnapshotButton, QRandomButton

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QResizeEvent, QHideEvent, QShowEvent
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QGroupBox,
                             QVBoxLayout, QHBoxLayout, QGridLayout)


class FacePanel(Panel, QObserver, Toolbox.Observer):
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
    _toolboxController: ToolboxController = None

    _detectors: list = None
    _detectorViews: list = None
    _inputView: QModelImageView = None

    _inputCounter: QLabel = None
    _processCounter: QLabel = None
    _loopButton: QLoopButton = None
    _snapshotButton: QSnapshotButton = None
    _randomButton: QRandomButton = None
   

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

        self._detectors = []
        self._detectorViews = []
        for name in 'haar', 'ssd':
            detector = create_detector(name, prepare=True)
            controller = DetectorController(engine=detector)
            if toolbox is not None:
                controller.runner = toolbox.runner  # FIXME[hack]
            self._detectors.append(controller)

        self._initUI()
        self._layoutUI()
        self.setToolboxController(toolbox)
        self.setDatasourceController(datasource)

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
        self._snapshotButton = QSnapshotButton('Snapshot')
        self._randomButton = QRandomButton('Random')

        for detector in self._detectors:
            self._detectorViews.append(DetectorWidget(detector=detector))

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
        row.addWidget(self._randomButton)
        row.addWidget(self._snapshotButton)
        row.addWidget(self._loopButton)
        layout2.addLayout(row)
        layout2.addStretch(1)
        layout.addLayout(layout2)
        layout.setStretchFactor(layout2, 1)

        grid = QGridLayout()
        for i, view in enumerate(self._detectorViews):
            grid.addWidget(view, i//2, i%2)
        layout.addLayout(grid)
        layout.setStretchFactor(grid, 1)

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
        if change.input_changed:
            image = toolbox.input_data if toolbox is not None else None
            for detectorView in self._detectorViews:
                if detectorView.isChecked():
                    detectorView.setImage(image)
            self._inputCounter.setText(str(int(self._inputCounter.text())+1))

    def setDatasourceController(self, datasource: DatasourceController) -> None:
        """Set a :py:class:`DatasourceController` for this
        :py:class:`FacePanel`. We are not really interested in the
        Datasource but only forward this to subcomponents.
        """
        self._loopButton.setDatasourceController(datasource)
        self._snapshotButton.setDatasourceController(datasource)
        self._randomButton.setDatasourceController(datasource)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)

    def hideEvent(self, event: QHideEvent) -> None:
        self._deactivateView('_toolboxController')
        print("qtgui.panel.FacePanel: FacePanel is now invisible.")

    def showEvent(self, event: QShowEvent) -> None:
        interests = Toolbox.Change('input_changed')
        self._activateView('_toolboxController', interests)
        print("qtgui.panel.FacePanel: is now visible.")
