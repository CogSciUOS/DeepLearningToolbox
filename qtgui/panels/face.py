"""
File: face.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de

Graphical interface for face detection and recognition.
"""

from tools.face.detector import (Detector as FaceDetector,
                                 Controller as FaceController)
from tools.face.landmarks import (Detector as LandmarkDetector,
                                  Controller as LandmarkController)
from datasources import Metadata
from ..utils import QImageView, QObserver, QBusyWidget, protect

import numpy as np

from PyQt5.QtWidgets import (QGroupBox, QWidget, QLabel, QVBoxLayout)

class DetectorWidget(QGroupBox, QObserver, FaceDetector.Observer):
    """A detector widget displays the output of a Detector.
    """

    _faceController: FaceController = None
    _view: QImageView = None
    _label: QLabel = None
    _trueMetadata: Metadata = None

    def __init__(self, detector: FaceController=None, **kwargs):
        """Initialization of the FacePanel.

        Parameters
        ----------
        decector: FaceController
            The face detector providing data.
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        """
        super().__init__(**kwargs)
        self._trueMetadata = None
        self._initUI()
        self._layoutUI()
        self.setFaceController(detector)
        self.toggled.connect(self.onToggled)

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
        self._busy = QBusyWidget()

    def _layoutUI(self):
        layout = QVBoxLayout()
        layout.addWidget(self._view)
        layout.addWidget(self._label)
        layout.addWidget(self._busy)
        layout.addStretch(3)
        self.setLayout(layout)
        self.setCheckable(True)

    def setFaceController(self, detector: FaceController) -> None:
        """Set a new :py:class:`FaceController`.
        The face Controller will inform us whenever new faces where
        detected by one of the detectors.
        """
        interests = FaceDetector.Change('detection_finished')
        self._exchangeView('_faceController', detector,
                           interests=interests)
        self._busy.setView(detector)
        self.setTitle("None" if self._faceController is None else
                      (self._faceController._detector.__class__.__name__
                       if self._faceController else "Off"))  # FIXME[hack]

    def detector_changed(self, detector: FaceDetector,
                         change: FaceDetector.Change) -> None:
        if change.detection_finished:
            self.update()

    def setImage(self, image: np.ndarray, metadata: Metadata=None):
        self._trueMetadata = metadata
        if self._faceController is not None and bool(self._faceController):
            self._faceController.process(image)        
        self.update()

    def update(self):
        if (self._faceController is None or not self._faceController or
            not self.isChecked()):
            self._view.setImage(None)
            self._view.setMetadata(None)
            self._label.setText("Off.")
        else:
            image = self._faceController.image
            detections = self._faceController.detections

            if detections is not None:
                self._view.setImage(image)            
                self._view.setMetadata(self._trueMetadata, detections)
                if detections.has_regions():
                    self._label.setText(f"{len(detections.regions)} detected "
                                        f"in {detections.duration:.3f}s")
                else:
                    self._label.setText(f"Nothing detected"
                                        f"in {detections.duration:.3f}s")
            else:
                self._label.setText("No detections.")

    @protect
    def onToggled(self, on: bool) -> None:
        """We want to update this DetectorWidget when it gets
        (de)activated.
        """
        self.update()


from toolbox import Toolbox, Controller as ToolboxController
from datasources import Datasource, Controller as DatasourceController
from tools.face.detector import Detector as FaceDetector

from .panel import Panel
from ..utils import QImageView, QMetadataView, QObserver
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
    _metadataView: QMetadataView = None

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
        for name in 'haar', 'ssd', 'hog': # , 'cnn'
            # FIXME[todo]: 'cnn' runs really slow on CPU and
            # blocks the GUI! - we may think of doing
            # real multiprocessing!
            # https://stackoverflow.com/questions/7542957/is-python-capable-of-running-on-multiple-cores
            # https://stackoverflow.com/questions/47368904/control-python-code-to-run-on-different-core
            # https://docs.python.org/3/library/multiprocessing.html
            # https://stackoverflow.com/questions/10721915/shared-memory-objects-in-multiprocessing
            detector = FaceDetector.create(name, prepare=False)
            controller = FaceController(engine=detector)
            if toolbox is not None:
                controller.runner = toolbox.runner  # FIXME[hack]
            controller.prepare()
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
        self._metadataView = QMetadataView()
        self._metadataView.addAttribute('directory')
        self._metadataView.addAttribute('regions')
        self._metadataView.addAttribute('image')

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
        layout2.addWidget(self._metadataView)
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
            metadata = toolbox.input_metadata if toolbox is not None else None
            self._inputView.setMetadata(metadata)
            self._metadataView.setMetadata(metadata)
            for detectorView in self._detectorViews:
                if detectorView.isChecked():
                    detectorView.setImage(image, metadata)
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
