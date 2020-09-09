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
from toolbox import Toolbox
from dltb.base.image import Image, Imagelike
from dltb.tool import Tool
from dltb.tool.face.detector import Detector as FaceDetector
from dltb.tool.processor import Processor

# GUI imports
from ..utils import QObserver, QBusyWidget, protect
from ..widgets.image import QImageView
from ..widgets.data import QDataSelector
from ..widgets.tools import QToolSelector
from .panel import Panel

# logging
LOG = logging.getLogger(__name__)


class QDetectorWidget(QGroupBox, QObserver, qobservables={
        Processor: {'tool_changed', 'process_finished'}}):
    """A detector widget displays the output of a Detector.

    _processor: Processor
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
        self._initUI()
        self._layoutUI()
        self.setProcessor(Processor(detector))
        self.toggled.connect(self.onToggled)
        LOG.info("New QDetectorWidget[%s] initialized: detector=%s",
                 type(self), detector)

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

    def faceDetector(self) -> FaceDetector:
        """Get the detector currently applied by this
        :py:class:`QDetectorWidget`.

        Result
        ------
        detector: FaceDetector
            The face detector on `None` if no detector is set.
        """
        return self._processor.tool

    def setFaceDetector(self, detector: FaceDetector) -> None:
        """Set a new :py:class:`FaceDetector`.
        The face detector will inform us whenever new faces where
        detected.
        """
        detector.timer = True
        # setting the tool in the processor will indirectly trigger update()
        # in the main event loop thread.
        self._processor.tool = detector
        self._busy.setView(detector)
        if detector is not None:
            detector.prepare()

    def processor_changed(self, processor: Processor,
                          change: Processor.Change) -> None:
        # pylint: disable=invalid-name
        """React to changes in the observed :py:class:`FaceDetector`.
        """
        LOG.debug("QDetectorWidget.processor_changed(%s, %s)",
                  processor.tool, change)
        if change.tool_changed:
            detector = processor.tool
            self.setTitle("None" if detector is None else
                          type(detector).__name__)
        self.update()

    def setData(self, data: Data) -> None:
        """Set a new :py:class:`Data` object to be displayed by this
        :py:class:`QDetectorWidget`. The data is expected to an image.

        """
        self.setImage(None if not data else data.array, data)

    def setImage(self, image: np.ndarray, data: Data = None):
        """Set the image to be processed by the underlying detector.
        """
        self._trueMetadata = data
        if self._processor.ready:
            self._processor.process(data)
        self.update()

    def update(self):
        """Update the display of this :py:class:`QDetectorWidget`.
        """
        if self._processor.tool is None or not self.isChecked():
            self._view.setData(None)
            self._label.setText("Off.")
            return

        detector = self._processor.tool
        data = self._processor.data
        detections = detector.detections(data)
        LOG.debug("QDetectorWidget.update(): data = %s", data)
        LOG.debug("QDetectorWidget.update(): detections = %s", detections)

        self._view.setData(data)
        if detections is None:
            self._label.setText("No detections.")
            return

        # FIXME[old/todo]
        # self._view.showAnnotations(self._trueMetadata, detections)
        self._view.setMetadata(detections)
        duration = detector.duration(data) or -1.0
        if detections.has_regions():
            self._label.setText(f"{len(detections.regions)} faces detected "
                                f"in {duration:.3f}s")
        else:
            self._label.setText(f"Nothing detected in {duration:.3f}s")

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

        # name = 'shape_predictor_5_face_landmarks.dat'
        name = 'shape_predictor_68_face_landmarks.dat'  # FIXME[hack]

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

        self._toolSelector = QToolSelector()
        self.addAttributePropagation(Toolbox, self._toolSelector)
        self._toolSelector.toolSelected.connect(self.onToolSelected)

        self._detectorViews = []
        for detector in range(2):
            LOG.info("FacePanel._initUI(): add detector view %s", detector)
            self._detectorViews.append(QDetectorWidget())

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
        row.addWidget(self._toolSelector)
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

    def setFaceDetector(self, detector: FaceDetector) -> None:
        """Set the face detector to be used in this :py:class:`FacePanel`.
        """
        if detector is not self._detectorViews[0].faceDetector:
            self._detectorViews[0].setFaceDetector(detector)
            self._toolSelector.setCurrentTool(detector)

    def setToolbox(self, toolbox: Toolbox) -> None:
        """Set a new Toolbox.
        We are only interested in changes of the input data.
        """

        if toolbox is not None:
            print("FacePanel: new toolbox contains the following detectors:")
            detector = None
            for key in 'haar', 'mtcnn', 'ssd', 'hog', 'cnn':
                print(f" - {key}: {toolbox.contains_tool(key)}")
                if toolbox.contains_tool(key):
                    detector = toolbox.get_tool(key)
            if detector is not None:
                self.setFaceDetector(detector)

        self._dataSelector.setToolbox(toolbox)
        # self._dataView.setToolbox(toolbox)
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

    @protect
    def onToolSelected(self, tool: Tool) -> None:
        print(f"Tool selected: {tool}")
        self.setFaceDetector(tool)
