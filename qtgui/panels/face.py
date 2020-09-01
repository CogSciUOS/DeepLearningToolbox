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
from PyQt5.QtWidgets import (QGroupBox, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QGridLayout)
from PyQt5.QtGui import QResizeEvent

# toolbox imports
from datasource import Data
from tools.face.detector import Detector as FaceDetector
from toolbox import Toolbox
from dltb.base.image import Image, Imagelike

# GUI imports
from ..utils import QObserver, QBusyWidget, protect
from ..widgets.image import QImageView
from ..widgets.data import QDataSelector
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
        # pylint: disable=invalid-name
        """React to changes in the observed :py:class:`FaceDetector`.
        """
        LOG.debug("QDetectorWidget.detector_changed(%s, %s)", detector, change)
        if change.detection_finished:
            self.update()

    def setData(self, data: Data) -> None:
        """Set a new :py:class:`Data` object to be displayed by this
        :py:class:`QDetectorWidget`. The data is expected to an image.

        """
        self.setImage(None if not data else data.data, data)

    def setImage(self, image: np.ndarray, data: Data = None):
        """Set the image to be processed by the underlying detector.
        """
        self._trueMetadata = data
        if self._faceDetector is not None:
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
    def onToggled(self, _state: bool) -> None:
        """We want to update this QDetectorWidget when it gets
        (de)activated.
        """
        self.update()


class FacePanel(Panel, QObserver, qobservables={Toolbox: {'input_changed'}}):
    # pylint: disable=too-many-instance-attributes
    """The :py:class:`FacePanel` provides access to different
    face recognition technologies. This includes
    * face detection
    * face landmarking
    * face alignment
    * face recogntion

    The panel allows to independently select these components (if
    possible - some implementations combine individutal steps).

    The :py:class:`FacePanel` can be assigned an image to process
    using the :py:meth:`setImage`. This will trigger the processing
    steps, updating the display(s) accordingly. Alternatively, if
    a full data object is available, including image data and
    metadata like ground truth annotations, this can be set using
    the :py:class:`setData` method (which will internally call
    :py:class:`setImage`).

    A :py:class:`FacePanel` is associated with a :py:class:`Toolbox`.
    It will use the toolbox' input and the `QDataselector` can be
    used to change this input.

    Face detection
    --------------
    * Apply face detector to some data source
    * Compare multiple face detectors
    * Evaluate face detectors


    Properties
    ----------
    _toolbox: Toolbox = None

    _detectors: list = None
    _detectorViews: list = None
    _dataView: QDataView = None

    _inputCounter: QLabel = None
    _processCounter: QLabel = None
    _dataSelector: QDataSelector = None

    """

    def __init__(self, toolbox: Toolbox = None, parent=None):
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
        for name in ('haar',):  # 'mtcnn', 'ssd', 'hog', 'cnn'
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

        self._counter = 0  # FIXME[hack]

    def _initUI(self):
        """Initialize the user interface

        The user interface contains the following elements:
        * the data selector: depicting the current input image
          and allowing to select new inputs from a datasource
        * an input counter and a process counter:
        * up to four detector views: depicting faces located in
          the input image
        """
        #
        # Input data
        #

        # QImageView: a widget to display the input data
        self._dataSelector = QDataSelector()
        self._dataView = self._dataSelector.dataView()
        self._dataView.addAttribute('filename')
        self._dataView.addAttribute('basename')
        self._dataView.addAttribute('directory')
        self._dataView.addAttribute('path')
        self._dataView.addAttribute('regions')
        self._dataView.addAttribute('image')

        self._inputCounter = QLabel("0")
        self._processCounter = QLabel("0")

        for detector in self._detectors:
            LOG.info("FacePanel._initUI(): add detector %s", detector)
            self._detectorViews.append(QDetectorWidget(detector=detector))

    def _layoutUI(self):
        """Initialize the user interface of this :py:class:`FacePanel`.
        """
        # The big picture:
        #
        #  +--------------------+----------------------------------------+
        #  |+------------------+|                                        |
        #  ||dataSelector      ||                                        |
        #  ||[view]            ||                                        |
        #  ||                  ||                                        |
        #  ||                  ||                                        |
        #  ||                  ||                                        |
        #  ||                  ||                                        |
        #  ||                  ||                                        |
        #  ||[navigator]       ||                                        |
        #  ||                  ||                                        |
        #  ||                  ||                                        |
        #  |+------------------+|                                        |
        #  +--------------------+----------------------------------------+
        layout = QHBoxLayout()

        layout2 = QVBoxLayout()
        layout2.addWidget(self._dataSelector)
        row = QHBoxLayout()
        row.addWidget(self._processCounter)
        row.addWidget(QLabel("/"))
        row.addWidget(self._inputCounter)
        row.addStretch()
        layout2.addLayout(row)
        layout2.addStretch(1)
        layout.addLayout(layout2)
        layout.setStretchFactor(layout2, 1)

        grid = QGridLayout()
        for i, view in enumerate(self._detectorViews):
            grid.addWidget(view, i//2, i % 2)
        layout.addLayout(grid)
        layout.setStretchFactor(grid, 1)

        self.setLayout(layout)

    @staticmethod
    def _detectorWidget(name: str, widget: QWidget):
        layout = QVBoxLayout()
        layout.addWidget(widget)
        layout.addWidget(QLabel(name))
        groupBox = QGroupBox(name)
        groupBox.setLayout(layout)
        groupBox.setCheckable(True)
        return groupBox

    def setImage(self, image: Imagelike) -> None:
        """Set the image for this :py:class:`FacePanel`. This
        will initiate the processing of this image using the
        current tools.
        """
        self.setData(Image.as_data(image))

    def setData(self, data: Data) -> None:
        """Set the data to be processed by this :py:class:`FacePanel`.
        """
        # set data for the dataView - this is redundant if data is set
        # from the toolbox (as the dataView also observes the toolbox),
        # but it is necessary, if setData is called independently.
        self._dataView.setData(data)

        # now feed the new data to the detecotors
        for detectorView in self._detectorViews:
            if detectorView.isChecked():
                detectorView.setData(data)

        # increase the data counter
        self._inputCounter.setText(str(int(self._inputCounter.text())+1))

    def setToolbox(self, toolbox: Toolbox) -> None:
        """Set a new Toolbox.
        We are only interested in changes of the input data.
        """
        self._dataSelector.setToolbox(toolbox)
        self._dataView.setToolbox(toolbox)
        self.setData(toolbox.input_data if toolbox is not None else None)

    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        # pylint: disable=invalid-name
        """The FacePanel is a Toolbox.Observer. It is interested
        in input changes and will react with applying face recognition
        to a new input image.
        """
        if change.input_changed:
            self.setData(toolbox.input_data)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """React to a resizing of this :py:class:`FacePanel`
        """
        LOG.debug("Resize event")
        super().resizeEvent(event)
