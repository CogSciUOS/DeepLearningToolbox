"""
File: face.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de

Graphical interface for face detection and recognition.
"""
# pylint --method-naming-style=camelCase --attr-naming-style=camelCase qtgui.panels.face

# standard imports
import logging

# third party imports
import numpy as np

# Qt imports
from PyQt5.QtWidgets import (QGroupBox, QWidget, QLabel, QVBoxLayout)
from PyQt5.QtWidgets import (QWidget, QLabel, QGroupBox,
                             QVBoxLayout, QHBoxLayout, QGridLayout)
from PyQt5.QtGui import QResizeEvent

# toolbox imports
from tools.face.detector import Detector as FaceDetector
from tools.face.landmarks import Detector as LandmarkDetector
from datasource import Data
from toolbox import Toolbox

# GUI imports
from ..utils import QObserver, QBusyWidget, protect
from ..widgets.datasource import QDatasourceNavigator
from ..widgets.image import QImageView
from ..widgets.data import QDataView
from .panel import Panel



# logging
LOG = logging.getLogger(__name__)


class QDetectorWidget(QGroupBox, QObserver, qobservables={
        FaceDetector: {'detection_finished'}}):
    """A detector widget displays the output of a Detector.

    _faceDetector: FaceDetector
    _view: QImageView
    _label: QLabel
    _busy: QBusyWidget
    _trueMetadata
    """

    def __init__(self, detector: FaceDetector = None, **kwargs):
        """Initialization of the FacePanel.

        Parameters
        ----------
        decector: FaceDetector
            The face detector providing data.
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        """
        super().__init__(**kwargs)
        self._trueMetadata = None
        self._faceDetector = None
        self._initUI()
        self._layoutUI()
        self.setFaceDetector(detector)
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

    def setFaceDetector(self, detector: FaceDetector) -> None:
        """Set a new :py:class:`FaceDetector`.
        The face detector will inform us whenever new faces where
        detected.
        """
        self.setDetector(detector)
        self._busy.setView(detector)
        self._faceDetector = detector
        self.setTitle("None" if self._faceDetector is None else
                      self._faceDetector.__class__.__name__)

    def detector_changed(self, detector: FaceDetector,
                         change: FaceDetector.Change) -> None:
        LOG.debug("QDetectorWidget.detector_changed(%s, %s)", detector, change)
        if change.detection_finished:
            self.update()

    def setData(self, data: Data) -> None:
        self.setImage(None if not data else data.data, data)

    def setImage(self, image: np.ndarray, data: Data = None):
        """Set the image to be processed by the underlying detector.
        """
        print("QDetectorWidget.setImage"
              f"({None if image is None else image.shape})")
        self._trueMetadata = data
        if self._faceDetector is not None:
            print(f"QDetectorWidget.process({data})")
            self._faceDetector.process(data)
        self.update()

    def update(self):
        """Update the display of this :py:class:`QDetectorWidget`.
        """
        if self._faceDetector is None or not self.isChecked():
            self._view.setData(None)
            self._label.setText("Off.")
            return
        
        data = self._faceDetector.data
        detections = self._faceDetector.detections
        LOG.debug("QDetectorWidget.update(): %s", detections)

        self._view.setData(data)
        if detections is None:
            self._label.setText("No detections.")
            return
    
        # FIXME[old/todo]
        # self._view.showAnnotations(self._trueMetadata, detections)
        self._view.setMetadata(detections)
        if detections.has_regions():
            self._label.setText(f"{len(detections.regions)} detected "
                                f"in {detections.duration:.3f}s")
        else:
            self._label.setText(f"Nothing detected"
                                f"in {detections.duration:.3f}s")

    @protect
    def onToggled(self, on: bool) -> None:
        """We want to update this QDetectorWidget when it gets
        (de)activated.
        """
        self.update()


class FacePanel(Panel, QObserver, qobservables={
        Toolbox: {'input_changed', 'datasource_changed'}}):
    """The FacePanel provides access to different face recognition
    technologies. This includes
    * face detection
    * face landmarking
    * face alignment
    * face recogntion

    Face detection:
    * Apply face detector to some data source
    * Compare multiple face detectors
    * Evaluate face detectors

    
    _toolbox: Toolbox = None

    _detectors: list = None
    _detectorViews: list = None
    _inputView: QImageView = None
    _dataView: QDataView = None

    _inputCounter: QLabel = None
    _processCounter: QLabel = None
    _datasourceNavigator: QDatasourceNavigator = None
    """

    def __init__(self, toolbox: Toolbox = None,
                 datasource: Datasource = None, parent=None):
        """Initialization of the FacePanel.

        Parameters
        ----------
        toolbox: Toolbox
            The toolbox provides input data.
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        """
        super().__init__(parent)

        name = 'shape_predictor_5_face_landmarks.dat'
        name = 'shape_predictor_68_face_landmarks.dat'  # FIXME[hack]

        self._detectors = []
        self._detectorViews = []
        for name in 'haar', :  # 'mtcnn', 'ssd', 'hog', 'cnn'
            # FIXME[todo]: 'cnn' runs really slow on CPU and
            # blocks the GUI! - we may think of doing
            # real multiprocessing!
            # https://stackoverflow.com/questions/7542957/is-python-capable-of-running-on-multiple-cores
            # https://stackoverflow.com/questions/47368904/control-python-code-to-run-on-different-core
            # https://docs.python.org/3/library/multiprocessing.html
            # https://stackoverflow.com/questions/10721915/shared-memory-objects-in-multiprocessing
            # detector = FaceDetector[name] # .create(name, prepare=False)
            LOG.info("FacePanel: Initializing detector '%s'", name)
            detector = FaceDetector.register_initialize_key(
                name)  # FIXME[todo], busy_async=False)
            if toolbox is not None:
                detector.runner = toolbox.runner  # FIXME[hack]
            LOG.info("FacePanel: Preparing detector '%s'", name)
            detector.prepare(busy_async=False)
            self._detectors.append(detector)

        self._initUI()
        self._layoutUI()
        self.setToolbox(toolbox)
        self.setDatasource(datasource)

        self._counter = 0  # FIXME[hack]

    def _initUI(self):
        """Initialize the user interface

        The user interface contains the following elements:
        * the input view: depicting the current input image
        * a datasource navigator allowing to select new inputs from
        * a datasource, an input counter and a process counter:
        * up to four detector views: depicting faces located in the input image
        """
        #
        # Input data
        #

        # QImageView: a widget to display the input data
        self._inputView = QImageView()
        self._dataView = QDataView()
        self._dataView.addAttribute('filename')
        self._dataView.addAttribute('basename')
        self._dataView.addAttribute('directory')
        self._dataView.addAttribute('path')
        self._dataView.addAttribute('regions')
        self._dataView.addAttribute('image')

        self._inputCounter = QLabel("0")
        self._processCounter = QLabel("0")

        self._datasourceNavigator = QDatasourceNavigator()

        for detector in self._detectors:
            LOG.info("FacePanel._initUI(): add detector %s", detector)
            self._detectorViews.append(QDetectorWidget(detector=detector))

    def _layoutUI(self):
        """Initialize the user interface
        """
        layout = QHBoxLayout()

        layout2 = QVBoxLayout()
        layout2.addWidget(self._inputView)
        layout2.addWidget(self._dataView)
        row = QHBoxLayout()
        row.addWidget(self._processCounter)
        row.addWidget(QLabel("/"))
        row.addWidget(self._inputCounter)
        row.addStretch()
        # FIXME[todo]: here we could add a datasource selector ...
        layout2.addLayout(row)
        layout2.addWidget(self._datasourceNavigator)
        layout2.addStretch(1)
        layout.addLayout(layout2)
        layout.setStretchFactor(layout2, 1)

        grid = QGridLayout()
        for i, view in enumerate(self._detectorViews):
            grid.addWidget(view, i//2, i % 2)
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

    def setData(self, data: Data) -> None:
        """Set the data to be processed by this :py:class:`FacePanel`.
        """
        self._inputView.setData(data)
        self._dataView.setData(data)
        for detectorView in self._detectorViews:
            if detectorView.isChecked():
                detectorView.setData(data)
        self._inputCounter.setText(str(int(self._inputCounter.text())+1))


    def setToolbox(self, toolbox: Toolbox) -> None:
        """Set a new Toolbox.
        We are only interested in changes of the input data.
        """
        self._inputView.setToolbox(toolbox)
        self.setDatasource(toolbox.datasource if toolbox is not None else None)
        self.setData(toolbox.input_data if toolbox is not None else None)

    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        """The FacePanel is a Toolbox.Observer. It is interested
        in input changes and will react with applying face recognition
        to a new input image.
        """
        print("FacePanel.toolbox_changed({toolbox}, {change})")
        if change.datasource_changed:
            self.setDatasource(toolbox.datasource)
        if change.input_changed:
            self.setData(toolbox.input_data)

    def setDatasource(self, datasource: Datasource) -> None:
        """Set a :py:class:`Datasource` for this :py:class:`FacePanel`.
        A datasource other the `None` is only allowed if no
        :py:class:`Toolbox` is assigned to this :py:class:`FacePanel`.
        As soon as a :py:class:`Toolbox` is assigned, its current
        datasource will be use.
        """
        print(f"\nFacePanel.setDatasource(datasource)\n")
        self._datasourceNavigator.setDatasource(datasource)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
