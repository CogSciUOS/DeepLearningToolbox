"""
.. moduleauthor:: Ulf Krumnack

.. module:: qtgui.widgets.data

This module provides widgets for viewing data and metadata.
"""
# pylint --method-naming-style=camelCase --attr-naming-style=camelCase --variable-naming-style=camelCase qtgui.widgets.data

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
from dltb.base.data import Data, ClassIdentifier
from toolbox import Toolbox
from datasource import Metadata
from datasource import Datasource, Datafetcher
from util.image import Region, PointsBasedLocation
from tools.activation import Engine as ActivationEngine

# GUI imports
from ..utils import QObserver, protect
from .image import QImageView
from .navigation import QIndexControls
from .datasource import QDatasourceNavigator

# logging
LOG = logging.getLogger(__name__)


class QDataInfoBox(QWidget, QObserver, qobservables={
        Datafetcher: {'data_changed'},
        Toolbox: {'input_changed'},
        ActivationEngine: {'input_changed'}}):
    # FIXME[concept]: should we forbid simultanous observation of
    # datafetcher and toolbox?
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
    (3) From a :py:class:`Datafetcher`, using the current data
    Setting a Toolbox will invalidate the Datafetcher and vice versa.

    _toolbox: Toolbox = None
    _datafetcher: Datafetcher = None
    """
    _activation: ActivationEngine = None  # FIXME[why]?
    _processed: bool = False

    def __init__(self, toolbox: Toolbox = None,
                 datafetcher: Datafetcher = None,
                 activation: ActivationEngine = None, **kwargs):
        '''Create a new QDataInfoBox.

        parent: QWidget
            Parent widget
        '''
        super().__init__(**kwargs)
        self._data = None
        self._metaText = ""
        self._initUI()
        self._layoutUI()
        self._showInfo()
        self.setToolbox(toolbox)
        self.setDatafetcher(datafetcher)
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

    def setToolbox(self, toolbox: Toolbox) -> None:
        """Set a Toolbox for this :py:class:`QDataInfoBox`.
        If a toolbox is set, the :py:class:`QDataInfoBox` will display
        the current input of the :py:class:`Toolbox`.
        """
        LOG.debug("QDataInfoBox.setToolbox: %s -> %s",
                  self._toolbox, toolbox)
        if toolbox is not None:
            self.setDatafetcher(None)

    @protect
    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        # pylint: disable=invalid-name
        """React to a input change of the :py:class:`Toolbox`.
        """
        LOG.debug("QDataInfoBox.toolbox_changed(%s, %s)", toolbox, change)
        self.setData(toolbox.input_data)

    #
    # Datafetcher.Observer
    #

    def setDatafetcher(self, datafetcher: Datafetcher) -> None:
        """Set a Datafetcher for this :py:class:`QDataInfoBox`.
        If a datafetcher is set, the :py:class:`QDataInfoBox` will display
        the currently selected data item of that :py:class:`Datafetcher`.
        """
        LOG.debug("QDataView.setDatafetcher: %s -> %s",
                  self._datafetcher, datafetcher)
        if datafetcher is not None:
            self.setToolbox(None)

    @protect
    def datafetcher_changed(self, datafetcher: Datafetcher,
                            change: Datafetcher.Change) -> None:
        # pylint: disable=invalid-name
        """React to newly fetched data item in the :py:class:`Datafetcher`.
        """
        LOG.debug("QDataInfoBox.datafetcher_changed(%s, %s)",
                  datafetcher, change)
        self.setData(datafetcher.data)

    #
    # ActivationEngine.Observer
    #

    # FIXME[hack]: what are we trying to achieve here?
    def activation_changed(self, _engine: ActivationEngine, info):
        # pylint: disable=invalid-name
        """React to changes of the activation engine.
        """
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

    #
    # Data
    #

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

        if data is self._data:
            return  # nothing changed

        # label = getattr(data, 'label', None)
        # description = getattr(data, 'description', None)
        # self._showInfo(data=data, description=description)

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
            self._metaText += "No data!<br>\n"
        self.update()

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
        """Set the display mode for this :py:class:`QDataInfoBox`.
        """
        if processed != self._processed:
            self._processed = processed
            self._updateInfo()

    def _showInfo(self, data: Data = None, label=None,
                  description: str = ''):
        """Show info for the given (image) data.
        """
        self._metaText = '<b>Input image:</b><br>\n'
        self._metaText += f'Description: {description}<br>\n'
        if label is not None:
            self._metaText += f'Label: {label}<br>\n'

        self._dataText = ('<b>Preprocessed input:</b><br>\n'
                          if self._processed else
                          '<b>Raw input:</b><br>\n')
        if data is not None:
            array = data.array
            self._dataText += (f'Input shape: {array.shape}, '
                               f'dtype={array.dtype}<br>\n')
            self._dataText += ('min = {}, max={}, mean={:5.2f}, '
                               'std={:5.2f}\n'.
                               format(array.min(), array.max(),
                                      array.mean(), array.std()))
        self.update()

    @staticmethod
    def _attributeValue(data: Data, attribute: str) -> Any:
        """Return the attribute of the current data object in
        a form suitable to be displayed in this :py:class:`QDataInfoBox`.
        """
        if attribute == 'data':
            return data.array.shape
        value = getattr(data, attribute)
        if isinstance(value, ClassIdentifier):
            if value.has_label('text'):
                value = f"{value.label('text')} ({value})"
        elif isinstance(value, Region):
            points = (len(value.location)
                      if isinstance(value.location, PointsBasedLocation)
                      else 'no')
            attributes = len(value._attributes)  # FIXME[hack]: private property
            value = (f"Region[{type(value.location).__name__}]: "
                     f"{points} points, {attributes} attributes")
        elif isinstance(value, list):
            value = (f"list[{len(value)}]: "
                     f"{type(value[0]) if value else 'empty'}")
        elif isinstance(value, np.ndarray):
            value = f"array[{value.dtype}]: {value.shape}"
        return value

    def update(self):
        """Update the information displayed by this :py:class:`QDataInfoBox`.
        """
        self._metaLabel.setText(self._metaText +
                                f"Toolbox: {self._toolbox is not None}, "
                                f"Datfetcher: {self._datafetcher is not None}")
        self._dataLabel.setText(
            self._dataText if self._button.isChecked() else '')


class QDataView(QWidget, QObserver, qobservables={
        Datafetcher: {'data_changed'}, Toolbox: {'input_changed'}}):
    """A Display for :py:class:`Data` objects.

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
        self.addAttributePropagation(Datafetcher, self._dataInfo)
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
        self._batchNavigator.hide()

        orientation = 'vertical'
        if orientation == 'vertical':
            # vertical layout: image above info
            layout = QVBoxLayout()
            row = QHBoxLayout()
            row.addStretch()
            row.addWidget(self._imageView)
            row.addStretch()
            layout.addLayout(row)
            layout.addLayout(dataInfo)
        else:
            # horizontal layout: image left of info
            layout = QHBoxLayout()
            self._imageView.setMinimumSize(400, 400)
            layout.addWidget(self._imageView)
            layout.addLayout(dataInfo)

        self.setLayout(layout)

    def imageView(self) -> QImageView:
        """Get the :py:class:`QImageView` of this :py:class:`QDataView`.
        """
        return self._imageView

    #
    # Toolbox.Observer
    #

    def setToolbox(self, toolbox: Toolbox) -> None:
        """Set a Toolbox for this :py:class:`QDataView`.
        If a toolbox is set, the :py:class:`QDataView` will display
        the current input of the :py:class:`Toolbox`.
        """
        LOG.debug("QDataView.setToolbox: %s -> %s",
                  self._toolbox, toolbox)
        if toolbox is not None:
            self.setDatafetcher(None)

    @protect
    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        # pylint: disable=invalid-name
        """React to a input change of the :py:class:`Toolbox`.
        """
        LOG.debug("QDataView.toolbox_changed(%s, %s)", toolbox, change)
        self.setData(toolbox.input_data)

    #
    # Datafetcher.Observer
    #

    def setDatafetcher(self, datafetcher: Datafetcher) -> None:
        """Set a Datafetcher for this :py:class:`QDataView`.
        If a datafetcher is set, the :py:class:`QDataView` will display
        the currently selected data item of that :py:class:`Datafetcher`.
        """
        LOG.debug("QDataView.setDatafetcher: %s -> %s",
                  self._datafetcher, datafetcher)
        if self._toolbox is not None and datafetcher is not None:
            if datafetcher is self._toolbox.datafetcher:
                self.setDatafetcher(None)
            else:
                self.setToolbox(None)

    @protect
    def datafetcher_changed(self, datafetcher: Datafetcher,
                            change: Datafetcher.Change) -> None:
        # pylint: disable=invalid-name
        """React to newly fetched data item in the :py:class:`Datafetcher`.
        """
        LOG.debug("QDataView.datafetcher_changed(%s, %s)", datafetcher, change)
        self.setData(datafetcher.data)

    #
    # Data
    #

    def setData(self, data: Data) -> None:
        """Set the :py:class:`Data` to be viewed by this
        :py:class:`QDataView`.
        """
        LOG.debug("QDataView.setData: %s", data)
        if data is self._data:
            return  # nothing changed

        self._data = data
        isBatch = bool(data) and data.is_batch
        self._batchNavigator.setVisible(isBatch)
        if isBatch:
            self._batchNavigator.setData(data)
        self.update()

    #
    # Configuration
    #

    def addAttribute(self, name: str) -> None:
        """Add an attribute to the list of attributes to be displayed.
        """
        self._attributes.append(name)

    def update(self):
        """Update the information displayed by this :py:class:`QDataView`
        to reflect the current state of the data object.
        """
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
        self._imageView.setImage(data.array)
        for attribute in data.attributes(batch=False):
            value = getattr(data, attribute)
            if isinstance(value, Region):
                self._imageView.addRegion(value)
            elif isinstance(value, list):
                for val in value:
                    if isinstance(val, Region):
                        self._imageView.addRegion(val)

    @protect
    def onIndexChanged(self, _index: int) -> None:
        """A slot to react to changes of the batch index.
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


class QDataSelector(QWidget, QObserver, qattributes={
        Toolbox: False, Datafetcher: False}):
    """A widget for selecting and viewing data items. This is essentially
    a combination of a :py:class:`QDataView` and a
    :py:class:`QDatasourceNavigator`.

    A :py:class:`QDataSelector` can include a
    :py:class:`QDatasourceNavigator` that allows to select a
    :py:class:`Data` object from a :py:class:`Datasource`.

    A :py:class:`QDataSelector` can be associated with a
    :py:class:`Toolbox`. In this case, the datasource navigator
    (if any) will navigate the current datasource of the toolbox.
    """

    def __init__(self, toolbox: Toolbox = None,
                 datafetcher: Datafetcher = None,
                 datasource_selector = True,
                 **kwargs) -> None:
        super().__init__()
        self._layoutScheme = 2
        self._initUI(datasource_selector)
        self._layoutUI()
        self.setToolbox(toolbox)
        self.setDatafetcher(datafetcher)

    def _initUI(self, datasource_selector) -> None:
        self._dataView = QDataView()
        self.addAttributePropagation(Toolbox, self._dataView)
        self.addAttributePropagation(Datafetcher, self._dataView)

        self._datasourceNavigator = \
            QDatasourceNavigator(datasource_selector=datasource_selector)
        self.addAttributePropagation(Toolbox, self._datasourceNavigator)
        self.addAttributePropagation(Datafetcher, self._datasourceNavigator)

        self._button = QPushButton("Change Layout")
        self._button.clicked.connect(self._onButtonClicked)

    def _layoutUI(self) -> None:
        """Layout the user interface. There are different ways in which
        the components of a :py:class:`QDataSelector` can be arranged.
        """
        if self._layoutScheme == 1:
            row = QHBoxLayout()
            row.addWidget(self._dataView)
            row.addWidget(self._datasourceNavigator)
            row.addWidget(self._button)
            self.setLayout(row)
        elif self._layoutScheme == 2:
            column = QVBoxLayout()
            column.addWidget(self._dataView)
            column.addWidget(self._datasourceNavigator)
            column.addWidget(self._button)
            self.setLayout(column)

    def _updateUI(self) -> None:
        """Update the layout of the user interface.
        """
        # QLayout::removeItem() just removes the item from layout, but
        # does not hide or delete it.
        layout = self.layout()
        layout.removeItem(layout.itemAt(2))
        layout.removeItem(layout.itemAt(1))
        layout.removeItem(layout.itemAt(0))
        # self.layout().removeWidget(self._dataView)
        # self.layout().removeWidget(self._datasourceNavigator)
        # self.layout().removeWidget(self._button)
        # Reparent the current layout to a dummy widget, which will
        # be delete immedieatly as we do not store a reference.
        # https://stackoverflow.com/a/10439207
        QWidget().setLayout(layout)
        self._layoutUI()
        self.update()

    @protect
    def _onButtonClicked(self, _checked: bool) -> None:
        if self._layoutScheme == 1:
            self._layoutScheme = 2
        elif self._layoutScheme == 2:
            self._layoutScheme = 1
        self._updateUI()

    def imageView(self) -> QImageView:
        """The :py:class:`QDataView` used by this
        :py:class:`QDataSelector`.
        """
        return self._dataView.imageView()

    def dataView(self) -> QDataView:
        """The :py:class:`QDataView` used by this
        :py:class:`QDataSelector`.
        """
        return self._dataView

    def datasourceNavigator(self) -> QDatasourceNavigator:
        """The :py:class:`QDatasourceNavigator` used by this
        :py:class:`QDataSelector`.
        """
        return self._datasourceNavigator

    def showNavigator(self) -> None:
        """Show the :py:class:`QDatasourceNavigator`.
        """
        self._datasourceNavigator.show()

    def hideNavigator(self) -> None:
        """Hide the :py:class:`QDatasourceNavigator`.
        """
        self._datasourceNavigator.hide()

    def setDatasource(self, datasource: Datasource) -> None:
        """Set the datasource for the datasource navigator.
        If this :py:class:`QDataSelector` is associated with
        a :py:class:`Toolbox`, it will set the current datasource
        for that tooblox.

        Arguments
        ---------
        datasource: Datasource
            The new datasource to navigate. If None, the datasource
            navigator will be disabled.
        """
        self._datasourceNavigator.setDatasource(datasource)
