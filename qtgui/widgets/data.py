"""
.. moduleauthor:: Ulf Krumnack

.. module:: qtgui.widgets.data

This module provides widgets for viewing data and metadata.
"""
# pylint --method-naming-style=camelCase --attr-naming-style=camelCase qtgui.widgets.data

# standard imports
from typing import Any
import logging

# third party imports
import numpy as np

# Qt imports
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy

# toolbox imports
from toolbox import Toolbox
from datasource import Data, Metadata, ClassIdentifier, Datasource
from datasource import View as DatasourceView
from util.image import Region, PointsBasedLocation
from tools.activation import Engine as ActivationEngine

# GUI imports
from ..utils import QObserver, protect
from .image import QImageView
from .navigation import QIndexControls

# logging
LOG = logging.getLogger(__name__)


class QDataInfoBox(QWidget, QObserver, qobservables={
        Datasource: {'data_changed'},
        Toolbox: {'input_changed'},
        ActivationEngine: {'input_changed'}}):
    # FIXME[concept]: should we forbid simultanous observation of
    # datasource and toolbox?
    """A :py:class:`QDataInfoBox` displays information on a piece of
    :py:class:`Data`.

    The :py:class:`QDataInfoBox` displays three graphical elements:
    * _metaLabel: a multiline label showing metadata on the current
        :py:class:`Data`.
    * _dataLabel: a multiline :py:class:`QLabel` showing statistical
        information on the current :py:class:`Data`.
    * _button: a checkable :py:class:`QPushButton` labeled `'Statistics'`.
        If the button is checked, additional statistical information will
        be displayed in the _dataLabel (otherwise it will be empty)

    Data can be provided in different ways:
    (1) By calling the method :py:meth:`setData`.
    (2) From a :py:class:`Toolbox`, using the input data
    (3) From a :py:class:`Datasource`, using the current data

    _toolbox: Toolbox = None
    """
    _datasourceView: DatasourceView = None
    _activation: ActivationEngine = None
    _processed: bool = False

    def __init__(self, toolbox: Toolbox = None,
                 datasource: DatasourceView = None,
                 activation: ActivationEngine = None, parent=None):
        '''Create a new QDataInfoBox.

        parent: QWidget
            Parent widget
        '''
        super().__init__(parent)
        self._initUI()
        self._layoutUI()
        self._showInfo()
        self.setToolbox(toolbox)
        self.setDatasourceView(datasource)
        self.setActivationEngine(activation)

    def _initUI(self):
        '''Initialise the UI'''
        self._metaLabel = QLabel()
        self._dataLabel = QLabel()
        self._button = QPushButton('Statistics')
        self._button.setCheckable(True)
        self._button.toggled.connect(self.update)
        self._button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def _layoutUI(self):
        self._metaLabel.setWordWrap(True)
        self._dataLabel.setWordWrap(True)

        layout1 = QHBoxLayout()
        layout1.addWidget(self._metaLabel)
        layout1.addWidget(self._button)

        layout = QVBoxLayout()
        layout.addLayout(layout1)
        layout.addWidget(self._dataLabel)
        self.setLayout(layout)

    #
    # Toolbox.Observer
    #

    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        LOG.debug("QDataView.toolbox_changed(%s, %s)", toolbox, change)
        if change.input_changed:
            if not self._toolbox:
                data = None
            elif self._processed:
                data = self._toolbox.input_data
            else:
                data = self._toolbox.input_data
            label = getattr(data, 'label', None)
            description = getattr(data, 'description', None)
            self._showInfo(data=data, description=description)

    #
    # Datasource.Observer
    #

    def setDatasource(self, datasource: Datasource) -> None:
        LOG.debug("QDataView.setDatasource: %s -> %s",
                  self._datasource, datasource)

    def setDatasourceView(self, datasource: DatasourceView) -> None:
        self._exchangeView('_datasourceView', datasource,
                           interests=Datasource.Change('data_changed'))

    def datasource_changed(self, datasource, change):
        LOG.debug("QDataInfoBox.datasource_changed(%s, %s)",
                  datasource, change)
        if change.data_changed:
            # the data provided by the Datasource has changed
            if self._datasource and self._datasource.fetched:
                self.setData(datasource.data)
            else:
                self.setData(None)

    #
    # ActivationEngine.Observer
    #

    def activation_changed(self, engine: ActivationEngine, info):
        if info.input_changed:
            self._updateInfo()

    def _updateInfo(self):
        if self._activation is None:
            data = self._toolbox.input_data if self._toolbox else None
        elif self._processed:
            data = self._activation.input_data
        else:
            data = self._activation.raw_input_data
        label = getattr(data, 'label', None)
        description = getattr(data, 'description', "No description"
                              if self._toolbox else "No toolbox")
        self._showInfo(data=data, label=label, description=description)

    # FIXME[old]
    def setController(self, controller) -> None:
        # FIXME[hack]: we need a more reliable way to observe multiple
        # observable!
        self.observe(controller.get_observable(), interests=None)

    @pyqtSlot(bool)
    @protect
    def onModeChanged(self, processed: bool):
        """The display mode was changed.

        Arguments
        ---------
        processed: bool
            The new display mode (False=raw, True=processed).
        """
        self.setMode(processed)

    def setMode(self, processed):
        if processed != self._processed:
            self._processed = processed
            self._updateInfo()

    def _showInfo(self, data: np.ndarray = None, label=None,
                  description: str = ''):
        '''Show info for the given (image) data.
        '''
        self._metaText = '<b>Input image:</b><br>\n'
        self._metaText += f'Description: {description}<br>\n'
        if label is not None:
            self._metaText += f'Label: {label}<br>\n'

        self._dataText = ('<b>Preprocessed input:</b><br>\n'
                          if self._processed else
                          '<b>Raw input:</b><br>\n')
        if data is not None:
            self._dataText += (f'Input shape: {data.shape}, '
                               f'dtype={data.dtype}<br>\n')
            self._dataText += ('min = {}, max={}, mean={:5.2f}, '
                               'std={:5.2f}\n'.
                               format(data.min(), data.max(),
                                      data.mean(), data.std()))
        self.update()

    def update(self):
        self._metaLabel.setText(self._metaText)
        self._dataLabel.setText(
            self._dataText if self._button.isChecked() else '')

    def _attributeValue(self, data: Data, attribute: str) -> Any:
        if attribute == 'data':
            return data.data.shape
        value = getattr(data, attribute)
        if isinstance(value, ClassIdentifier):
            if value.has_label('text'):
                value = f"{value.label('text')} ({value})"
        elif isinstance(value, Region):
            points = (len(value.location)
                      if isinstance(value.location, PointsBasedLocation)
                      else 'no')
            attributes = len(value._attributes)  # FIXME[hack]: privte property
            value = (f"Region[{type(value.location).__name__}]: "
                     f"{points} points, {attributes} attributes")
        elif isinstance(value, list):
            value = (f"list[{len(value)}]: "
                     f"{type(value[0]) if value else 'empty'}")
        elif isinstance(value, np.ndarray):
            value = f"array[{value.dtype}]: {value.shape}"
        return value

    def setData(self, data: Data, index: int = None) -> None:
        """Set the data to be displayed in the :py:class:`QDataInfoBox`.

        Arguments
        ---------
        data: Data
            The data to be displayed in this :py:class:`QDataInfoBox`.
        index: int
            The index of the batch item to be displayed (in case that
            `data` is a batch).
        """
        LOG.debug("QDataInfoBox.setData[%s]: %s", index, data)
        self._metaText = "<b>Input Data:</b><br>\n"
        if data:
            for attribute in data.attributes(batch=False):
                value = self._attributeValue(data, attribute)
                self._metaText += f"{attribute}: {value}<br>\n"
            if data.is_batch:
                if index is not None:
                    self._metaText += \
                        f"<b>Batch Data ({index}/{len(data)}):</b><br>\n"
                    data = data[index]
                else:
                    self._metaText += \
                        f"<b>Batch Data ({len(data)}):</b><br>\n"
            for attribute in data.attributes(batch=True):
                value = ('*' if data.is_batch else
                         self._attributeValue(data, attribute))
                self._metaText += f"{attribute}[batch]: {value}<br>\n"
        else:
            self._metaText += f'No data!<br>\n'
        self.update()


class QDataView(QWidget, QObserver, qobservables={
        Datasource: {'data_changed'}, Toolbox: {'input_changed'}}):
    """Display data.

    The display is split into two parts:
    * _imageView: a :py:class:`QImageView` for displaying image data.
    * _inputInfo: a :py:class:`QDataInfoBox` for displaying the
        :py:class:`Data` object.

    In case the :py:class:`Data` object represents a batch of data,
    a :py:class:`QBatchNavigator` will be included to allow navigating
    through the batch.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = None
        self._attributes = []
        self._initUI()
        self._layoutUI()

    def _initUI(self):
        """Initialize the user interface of this :py:class:`QDataView`.
        """
        # QImageView: a widget to display the input data
        self._imageView = QImageView()
        self.addAttributePropagation(Toolbox, self._imageView)

        self._dataInfo = QDataInfoBox()
        self.addAttributePropagation(Toolbox, self._dataInfo)
        self.addAttributePropagation(Datasource, self._dataInfo)
        self._imageView.modeChanged.connect(self._dataInfo.onModeChanged)

        self._batchNavigator = QBatchNavigator()
        self._batchNavigator.indexChanged.connect(self.onIndexChanged)

    def _layoutUI(self):
        """Initialize the user interface of this :py:class:`QDataView`.
        """
        # FIXME[layout]
        # layout.setSpacing(0)
        # layout.setContentsMargins(0, 0, 0, 0)
        self._dataInfo.setMinimumWidth(200)
        # keep image view square (FIXME[question]: does this make
        # sense for every input?)
        self._imageView.heightForWidth = lambda w: w
        self._imageView.hasHeightForWidth = lambda: True

        # FIXME[todo]: make this span the whole width
        #self.setMinimumWidth(1200)

        dataInfo = QVBoxLayout()
        dataInfo.addWidget(self._dataInfo)
        dataInfo.addWidget(self._batchNavigator)

        if True:
            layout = QVBoxLayout()
            row = QHBoxLayout()
            row.addStretch()
            row.addWidget(self._imageView)
            row.addStretch()
            layout.addLayout(row)
            layout.addLayout(dataInfo)
        else:
            layout = QHBoxLayout()
            self._imageView.setMinimumSize(400, 400)
            layout.addWidget(self._imageView)
            layout.addLayout(dataInfo)

        self.setLayout(layout)

    def setToolbox(self, toolbox: Toolbox) -> None:
        """Set a Toolbox for this :py:class:`QDataView`.
        If a toolbox is set, the :py:calls:`QDataView` will display
        the current input of the :py:class:`Toolbox`.
        """
        if toolbox is not None:
            self.setDatasource(None)

    def setDatasource(self, datasource: Datasource) -> None:
        """Set a Datasource for this :py:class:`QDataView`.
        If a datasource is set, the :py:calls:`QDataView` will display
        the currently selected data item of that :py:class:`Datasource`.
        """
        if datasource is not None:
            self.setToolbox(None)

    @protect
    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        """React to a input change of the :py:class:`Toolbox`.
        """
        self.setData(toolbox.input_data)

    @protect
    def datasource_changed(self, datasource: Datasource,
                           change: Datasource.Change) -> None:
        """React to a fetching a data item in the :py:class:`Datasource`.
        """
        LOG.debug("QDataView.datasource_changed(%s, %s)", datasource, change)
        if change.data_changed:
            self.setData(datasource.data)

    def setData(self, data: Data) -> None:
        """Set the :py:class:`Data` to be viewed by this
        :py:class:`QDataView`.
        """
        LOG.debug("QDataView.setData: %s", data)
        self._data = data
        is_batch = bool(data) and data.is_batch
        self._batchNavigator.setVisible(is_batch)
        if is_batch:
            self._batchNavigator.setData(data)
        self.update()

    def addAttribute(self, name: str) -> None:
        """
        """
        self._attributes.append(name)

    def update(self):
        if not self._data:
            self._imageView.setImage(None)
            self._imageView.setMetadata(None)
            self._dataInfo.setData(None)
            return

        data = self._data
        if data.is_batch:
            self._dataInfo.setData(data, self._batchNavigator.index())
            data = data[self._batchNavigator.index()]
        else:
            self._dataInfo.setData(data)

        if not data.is_image:
            self._imageView.setData(None)
            return

        # we have an image
        self._imageView.setImage(data.data)
        for attribute in data.attributes(batch=False):
            value = getattr(data, attribute)
            if isinstance(value, Region):
                self._imageView.addRegion(value)
            elif isinstance(value, list):
                for val in value:
                    if isinstance(val, Region):
                        self._imageView.addRegion(val)
            # self._imageView.setMetadata(self._datasource.data)

    @protect
    def onIndexChanged(self, index: int) -> None:
        """Slot to react to changes of the batch index.
        """
        self.update()


class QBatchNavigator(QIndexControls):
    """A widget allowing to navigate through a batch of py:class:`Data`.
    """

    def setData(self, data: Data) -> None:
        """Set the :py:class:`Data`.
        The number of elements will be adapted to the batch size.
        If `data` does not represent a batch, navigation will be disabled.

        Parameters
        ----------
        data: Data
            The (batch) data in which to navigate.
        """
        if data is not None and data.is_batch:
            self.setEnabled(True)
            self.setElements(len(data))
            self.setIndex(0)
        else:
            self.setEnabled(False)
            self.setElements(-1)
        self.update()


# FIXME[old]
class QMetadataView(QLabel):
    """A :py:class:`QWidget` to display :py:class:`Metadata` information.
    """

    _metadata: Metadata

    def __init__(self, metadata: Metadata = None, **kwargs):
        """Initialize this :py:class:`QMetadataView`.
        """
        super().__init__(**kwargs)
        self._attributes = []
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setMetadata(metadata)

    def addAttribute(self, attribute):
        """Add an attribute to be displayed.
        """
        self._attributes.append(attribute)

    def setMetadata(self, metadata: Metadata) -> None:
        """Set the metadata.
        """
        self._metadata = metadata
        if metadata is None:
            self.setText(None)
        else:
            text = ("No description" if hasattr(metadata, 'description')
                    else metadata.description)
            for attribute in self._attributes:
                if not metadata.has_attribute(attribute):
                    text += f"\n{attribute}: None"
                elif attribute == 'regions':
                    text += f"\n{len(metadata.regions)} regions"
                elif attribute == 'image':
                    text += (f"\n{metadata.image.shape}, "
                             f"dtype={metadata.image.dtype}")
                else:
                    value = metadata.get_attribute(attribute)
                    text += f"\n{attribute}: {value}"
            self.setText(text)
