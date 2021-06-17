"""
.. moduleauthor:: Ulf Krumnack

.. module:: qtgui.widgets.data

This module provides widgets for viewing data and metadata.
"""
# pylint --method-naming-style=camelCase --attr-rgx="_[a-z]+[A-Za-z0-9]+" --attr-naming-style=camelCase --variable-naming-style=camelCase --extension-pkg-whitelist=PyQt5 qtgui.widgets.data

# standard imports
from typing import Any, Optional, Sequence, Mapping
import os
import json
import logging

# third party imports
import numpy as np

# Qt imports
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy
from PyQt5.QtWidgets import QScrollArea

# toolbox imports
from toolbox import Toolbox
from dltb.base.data import Data
from dltb.base.image import Image
from dltb.base.meta import Metadata
from dltb.base.image import Region, PointsBasedLocation, Landmarks
from dltb.datasource import Datasource, Datafetcher
from dltb.tool.classifier import ClassIdentifier

# GUI imports
from ..utils import QObserver, protect
from .image import QImageView, QMultiImageView
from .navigation import QIndexControls
from .datasource import QDatasourceNavigator
from .scroll import QOrientedScrollArea

# logging
LOG = logging.getLogger(__name__)


class QDataInfoBox(QWidget, QObserver, qobservables={
        Datafetcher: {'data_changed'},
        Toolbox: {'input_changed'}}):
    # FIXME[old]: modes / statistics
    """A :py:class:`QDataInfoBox` displays information on a piece of
    :py:class:`Data`.

    Displaying data
    ---------------

    A :py:class:`Data` object encapsulates the actual data (as some
    form of array) as well as optional metadata. The :py:class:`QDataInfoBox`
    aims at a concise textual presentation of that information.
    As the actual form of data as well as the available metadata can
    vary greatly, there is no single best mode of presentation.  Instead,
    the :py:class:`QDataInfoBox` offers different modes and configuration
    options allowing to adjust the display to specific needs.
    
    With respect to the actual data (the array), the :py:class:`QDataInfoBox`
    can display simple metadata (like shape and data type). It also
    supports displaying different statistics (like mean, standard deviation,
    etc.). Note that computing statistics can be computationally expensive
    and hence should be done in a background thread.

    When it comes to metadata, the situation is more involved, as amount
    and types of metadata can greatly vary. Here the :py:class:`QDataInfoBox`
    provides some base functionality for different common classes as
    well as extension mechanisms.

    The display of the :py:class:`QDataInfoBox` can be adapted in
    several directions:
    * a whitelist/blacklist approach allows to selectively enable or
      disable the presentation of specific metadata
    * formaters can be provided to alter the display of specific
      metadata

    The :py:class:`QDataInfoBox` displays three graphical elements:
    * _metaLabel: a multiline label showing metadata on the current
        :py:class:`Data`.
    * _dataLabel: a multiline :py:class:`QLabel` showing statistical
        information on the current :py:class:`Data`.

    Setting Data
    ------------

    Data can be provided in different ways:
    (1) Directly by calling the method :py:meth:`setData`.
    (2) Indirectly by setting a :py:class:`Toolbox`. The
        :py:class:`QDataInfoBox` will then observe this for
        `'input_changed'` notifications and update its current data
        to be the input data or the toolbox, whenever receiving
        that notification.
    (3) From a :py:class:`Datafetcher`. The :py:class:`QDataInfoBox`
        will observer the `Datafetcher` for `'data_changed'` notifications
        and update its current data whenever new data was fetched, to
        display information on that data.

    Setting a Toolbox will invalidate the Datafetcher and vice versa.
    The ratio behind this behavior is that often a `Datafetcher` will
    set the input data for a `Toolbox` and hence will result in double
    updates.


    Properties
    ----------

    mode: bool
        A flag indicating if data is shown in raw format or whether some
        preprocessing is applied.
    statistics: bool
        A flag indicating whether statistical information should be
        shown (showing statistical information may require some extra
        time for computation)

    _toolbox: Toolbox = None
    _datafetcher: Datafetcher = None

    """    
    _whitelist: Optional[Sequence] = None  # also provides an order
    _blacklist: Optional[AbstractSet] = None  # no order
    _formatter: Optional[Mapping] = None  # map name or type to formatter

    _processed: bool = False

    statisticsChanged = pyqtSignal(bool)

    def __init__(self, toolbox: Toolbox = None,
                 datafetcher: Datafetcher = None, **kwargs) -> None:
        """Initialize a new QDataInfoBox.

        parent: QWidget
            Parent widget
        """
        super().__init__(**kwargs)
        self._metaText = ""
        self._dataText = ""
        self._data = None
        self._statistics = False
        self._initUI()
        self._layoutUI()
        self.setToolbox(toolbox)
        self.setDatafetcher(datafetcher)

    def _initUI(self):
        """Initialise the UI."""
        self._metaLabel = QLabel()
        self._metaLabel.setWordWrap(True)
        self._metaLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self._dataLabel = QLabel()
        self._dataLabel.setWordWrap(True)
        self._dataLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)

    def _layoutUI(self):
        """Layout the UI.
        """
        layout = QVBoxLayout()
        layout.addWidget(self._metaLabel)
        layout.addWidget(self._dataLabel)
        layout.addStretch()
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
        LOG.debug("QDataInfoBox.setDatafetcher: %s -> %s",
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

        self._data = data
        self._updateMeta(index=index)
        self._updateStatistics()
        self.update()

    def mode(self) -> bool:
        """A flag indicating wether data is presented preprocessed (`True`)
        or in raw format (`False`)
        """
        return self._processed

    @pyqtSlot(bool)
    # @protect
    def setMode(self, processed: bool) -> None:
        """Set the display mode for this :py:class:`QDataInfoBox`.

        Arguments
        ---------
        processed: bool
            The new display mode (False=raw, True=processed).
        """
        LOG.info("QDataInfoBox.setMode(%s)", processed)
        if processed != self._processed:
            self._processed = processed
            self._updateStatistics()
            self.update()

    def statistics(self) -> bool:
        """A flag indicating if statistics are to be shown in this
        :py:class:`QDataInfoBox`.
        """
        return self._statistics

    @pyqtSlot(bool)
    # FIXME[bug]: with @protect, setMode is called instead of setStatistics,
    # when setStatistics is connected to a button.
    # @protect
    def setStatistics(self, show: bool) -> None:
        """A the flag indicating that statistics are to be shown in this
        :py:class:`QDataInfoBox`.
        """
        LOG.info("QDataInfoBox.setStatistics(%s)", show)
        if self._statistics != show:
            self._statistics = show
            self.statisticsChanged.emit(show)
            self._updateStatistics()
            self.update()

    #
    # Output
    #

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
                value = f"{value['text']} ({value})"
        elif isinstance(value, Region):
            points = (len(value.location)
                      if isinstance(value.location, PointsBasedLocation)
                      else 'no')
            value = (f"Region[{type(value.location).__name__}]: "
                     f"{points} points, {len(value)} attributes")
        elif isinstance(value, list):
            value = (f"list[{len(value)}]: "
                     f"{type(value[0]) if value else 'empty'}")
        elif isinstance(value, (bool, np.bool_)):
            color = 'green' if value else 'red'
            value = f'<font color="{color}">{value}</font>'
        elif isinstance(value, np.ndarray):
            value = f"array[{value.dtype}]: {value.shape}"
        else:
            value = f"{value} [{type(value).__name__}]"
        return value

    def _updateMeta(self, index: int = None) -> None:
        """Update the `_metaText` attribute based on the current data.
        """
        data = self._data
        self._metaText = f"<b>Input Data ({type(data).__name__}):</b><br>\n"
        if not data:
            self._metaText += "No data!<br>\n"
            return

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

    def _updateStatistics(self):
        """Update the statistic information from the current data object.
        """
        if not self._statistics:
            return  # statistics are deactivated

        if self._data is None:
            data = None
        elif True:  # FIXME[todo/old/hack]: we need a better processing concept
            data = self._data
        elif self._processed:
            data = self._data.input_data
        else:
            data = self._data.raw_input_data

        # self._metaText = f'<b>Input image ({type(data).__name__}):</b><br>\n'
        # self._metaText += f'Description: {description}<br>\n'
        # if label is not None:
        #     self._metaText += f'Label: {label}<br>\n'

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

    def update(self):
        """Update the information displayed by this :py:class:`QDataInfoBox`.
        """
        self._metaLabel.setText(self._metaText +
                                f"Toolbox: {self._toolbox is not None}, "
                                f"Datfetcher: {self._datafetcher is not None}")
        self._dataLabel.setText(self._dataText if self._statistics else '')

    #
    # Events
    #

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process :py:class:`QKeyEvent`s for this :py:class:`QDataInfoBox`.

        S: toggle the statics flag
        M: toggle the mode flag
        """
        key = event.key()
        if key == Qt.Key_S:  # toggle statistics flag
            self.setStatistics(not self.statistics())
        elif key == Qt.Key_M:  # toggle mode flag
            self.setMode(not self.mode())
        else:
            super().keyPressEvent(event)


class QDataView(QWidget, QObserver, qobservables={
        Datafetcher: {'data_changed'}, Toolbox: {'input_changed'}}):
    # pylint: disable=too-many-instance-attributes
    # The QDataView consists of multiple graphical components
    """A Display for :py:class:`Data` objects.

    The actual components and layout of this display depend on type of
    data to be displayed.

    In case of an :py:class:`Image` the display is split into (at least)
    two parts:
    * _imageView: a :py:class:`QImageView` for displaying image data.
    * _inputInfo: a :py:class:`QDataInfoBox` for displaying the
        :py:class:`Data` object.
    Additional components may be included depending of the metadata
    provided by the :py:class:`Data` object.

    In case the :py:class:`Data` object represents a batch of data,
    a :py:class:`QBatchNavigator` will be included to allow navigating
    through the batch.

    * _statisticsButton: a checkable :py:class:`QPushButton` labeled
        `'Statistics'`. If the button is checked, additional statistical
        information will be displayed in the _dataLabel
        (otherwise it will be empty)
    """

    def __init__(self, orientation: Qt.Orientation = Qt.Vertical,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._orientation = orientation
        self._data = None
        self._attributes = []
        self._regions = []
        self._autosave = True  # automatically save when changing to new Data
        self._annotationsChanged = False  # a flag indicationg if data changed
        self._initUI()
        self._layoutUI()

    def _initUI(self) -> None:
        """Initialize the user interface of this :py:class:`QDataView`.
        """
        # QImageView: a widget to display the input data
        self._imageView = QImageView()
        self.addAttributePropagation(Toolbox, self._imageView)

        # QMultiImageView for displaying multiple regions of interest
        # (this is useful for detection tasks)
        orientation = (Qt.Vertical if self._orientation == Qt.Horizontal else
                       Qt.Horizontal)
        self._multiImageView = QMultiImageView(orientation=orientation)
        self._multiImageScroller = \
            QOrientedScrollArea(orientation=orientation)
        self._multiImageScroller.setWidget(self._multiImageView)
        self._multiImageScroller.setWidgetResizable(True)

        self._multiImageView.currentImageChanged.\
            connect(self._imageView.setCurrentRegion)
        self._multiImageView.annotationsChanged.\
            connect(self._imageView.updateRegion)
        self._multiImageView.annotationsChanged.\
            connect(self.annotationsChanged)
        self._imageView.currentRegionChanged.\
            connect(self._multiImageView.setCurrentImage)
        self._imageView.currentRegionChanged.\
            connect(self.setCurrentRegion)

        self._dataInfo = QDataInfoBox()
        self.addAttributePropagation(Toolbox, self._dataInfo)
        self.addAttributePropagation(Datafetcher, self._dataInfo)

        # FIXME[old]: this does not no longer work but it should be easy
        # to repair with the new data concept
        # self._imageView.modeChanged.connect(self._dataInfo.setMode)

        self._batchNavigator = QBatchNavigator()
        self._batchNavigator.indexChanged.connect(self.onIndexChanged)

        self._statisticsButton = QPushButton('Statistics')
        self._statisticsButton.setCheckable(True)
        self._statisticsButton.toggled.\
            connect(self._dataInfo.setStatistics)
        self._dataInfo.statisticsChanged.\
            connect(self._statisticsButton.setChecked)

    def _layoutUI(self) -> None:
        # The layout consists of two main parts
        # (1) textual display of data (QDataInfoBox)
        # (2) graphical display of data (depends of data type)

        # (1) dataInfo: a vertical layout containt
        # - the QDataInfoBox object (in a QScrollarea)
        # - a QBatchNavigator (only visible if data is batch)
        dataInfoLayout = QVBoxLayout()

        # add QDataInfoBox (self._dataInfo) embedded into an QScrollArea
        scrollarea = QScrollArea()
        scrollarea.setWidget(self._dataInfo)
        scrollarea.setWidgetResizable(True)
        dataInfoLayout.addWidget(scrollarea)

        row = QHBoxLayout()
        self._statisticsButton.setSizePolicy(QSizePolicy.Fixed,
                                             QSizePolicy.Fixed)
        row.addWidget(self._statisticsButton)

        # add QBatchNavigator with reasonable size policy
        sizePolicy = self._batchNavigator.sizePolicy()
        sizePolicy.setVerticalPolicy(QSizePolicy.Fixed)
        self._batchNavigator.setSizePolicy(sizePolicy)
        self._batchNavigator.hide()
        row.addWidget(self._batchNavigator)
        row.addStretch()
        dataInfoLayout.addLayout(row)

        # (2) graphical display
        if self._orientation == Qt.Vertical:
            # vertical layout: image above info
            layout = QVBoxLayout()
            layout.addWidget(self._imageView)
            layout.addWidget(self._multiImageScroller)
            layout.addLayout(dataInfoLayout)
        else:  # self._orientation == Qt.Horizontal
            # horizontal layout: image left of info
            layout = QHBoxLayout()
            layout.addWidget(self._imageView)
            layout.addWidget(self._multiImageScroller)
            layout.addLayout(dataInfoLayout)

        self.setLayout(layout)

    def imageView(self) -> QImageView:
        """Get the :py:class:`QImageView` of this :py:class:`QDataView`.
        """
        return self._imageView

    def multiImageView(self) -> QImageView:
        """Get the :py:class:`QMultiImageView` of this :py:class:`QDataView`.
        """
        return self._multiImageView

    def setDataInfoVisible(self, visible: bool) -> None:
        """Get the :py:class:`QMultiImageView` of this :py:class:`QDataView`.
        """
        self._dataInfo.parent().setVisible(visible)
        self._statisticsButton.setVisible(visible)

    @pyqtSlot(int)
    def setCurrentRegion(self, index: int) -> None:
        """In :py:class:`Image` data with multiple regions, set
        the currently selected region.  It will be ensured that
        the relevant part of the :py:class`QMultiImageView` is
        visible.

        Arguments
        ---------
        index:
            The index of the region to become the current region.
        """
        position = self._multiImageView.imagePosition(index)
        if position is not None:
            imageSize = self._multiImageView.imageSize()
            spacing = self._multiImageView.spacing()
            xmargin = (imageSize.width() + spacing) // 2
            ymargin = (imageSize.height() + spacing) // 2
            self._multiImageScroller.ensureVisible(position.x(), position.y(),
                                                   xmargin, ymargin)

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
        LOG.debug("QDataView.setData: %s", data is not None)
        if data is self._data:
            return  # nothing changed

        if (self._data is not None and
                self._autosave and self._annotationsChanged):
            self._saveAnnotations(overwrite=True)

        self._data = data
        isBatch = bool(data) and data.is_batch
        self._batchNavigator.setVisible(isBatch)
        if isBatch:
            self._batchNavigator.setData(data)

        self.update()
        if self._data is not None:
            self._loadAnnotations()

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
            self._multiImageView.setImagesFromRegions(None, [])
            self._dataInfo.setData(None)
            return

        data = self._data
        if data.is_batch:
            self._dataInfo.setData(data, self._batchNavigator.index())
            data = data[self._batchNavigator.index()]
        else:
            self._dataInfo.setData(data)

        if not isinstance(data, Image):
            self._imageView.setData(None)
            return

        # we have an image
        self._imageView.setImage(data.array)
        regions = []
        for attribute in data.attributes(batch=False):
            value = getattr(data, attribute)
            if isinstance(value, Region):
                self._imageView.addRegion(value)
                regions.append(value)
            elif isinstance(value, Landmarks):
                self._imageView.addLandmarks(value)
            elif isinstance(value, list):
                for val in value:
                    if isinstance(val, Region):
                        self._imageView.addRegion(val)
                        regions.append(val)
        self._regions = regions
        self._multiImageView.setImagesFromRegions(data, regions)

        if self._annotationsChanged:
            self.setStyleSheet("border: 1px solid red")
        else:
            self.setStyleSheet("border: 1px solid green")


    @protect
    def onIndexChanged(self, _index: int) -> None:
        """A slot to react to changes of the batch index.
        """
        self.update()

    #
    # Events
    #

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process key events for this :py:class:`QDataView`.
        The following keys are recognized:

        B: toggle visibility of the batch navigator
        M: toggle visibility of multi image view
        Return: set the annotation changed flag
        Ctrl+Z: undo
        Ctrl+S: save
        """
        key = event.key()
        if key == Qt.Key_B:  # toggle visibility of batch navigator
            visible = self._batchNavigator.isVisible()
            self._batchNavigator.setVisible(not visible)
        elif key == Qt.Key_M:  # toggle visibility of multi image view
            visible = self._multiImageScroller.isVisible()
            self._multiImageScroller.setVisible(not visible)
        elif key == Qt.Key_Return:  # set the annotation changed flag
            self._annotationsChanged = True
            self.update()
        elif key == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            # Ctrl+Z = undo
            self._loadAnnotations()
            self.update()
        elif key == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
            # Ctrl+S = save
            self._saveAnnotations()
            self.update()
        else:
            super().keyPressEvent(event)

    #
    # Saving changes
    #

    # FIXME[todo]: work in progress

    @pyqtSlot(int)
    def annotationsChanged(self, index: int) -> None:
        """Indicate whether the annotations for the given index has
        changed.
        """
        LOG.debug("Annotations for index %d changed", index)
        self._annotationsChanged = True
        self.update()

    def _filenameForAnnotations(self) -> str:
        """Filename to the file where additional anotations are stored.
        """
        if self._data is None:
            return None
        filename = getattr(self._data, 'filename', None)
        if filename is None:
            return None

        basename = os.path.basename(filename).rsplit('.', 1)[0]
        return 'annotations' + "-" + basename + '.json'

    def _loadAnnotations(self) -> None:
        filename = self._filenameForAnnotations()
        haveAnnotations = os.path.isfile(filename)
        self._data.add_attribute('have_annotations', haveAnnotations)
        self._annotationsChanged = False
        if not haveAnnotations:
            print(f"No annotations file '{filename}' exists.")
            for region in self._regions:
                region.invalid = False
            return
        with open(filename) as infile:
            annotations = json.load(infile)
        if len(annotations) != len(self._regions):
            LOG.warning("Mismatching number of regions in '%s' (%d vs. %d).",
                        filename, len(annotations), len(self._regions))
            return
        for region, invalid in zip(self._regions, annotations):
            region.invalid = bool(invalid)
        print(f"Loaded annotations from '{filename}'")

    def _saveAnnotations(self, overwrite: bool = False) -> None:
        filename = self._filenameForAnnotations()
        if os.path.isfile(filename) and not overwrite:
            LOG.warning("Not overwriting existing annotations file '%s'",
                        filename)
            return
        with open(filename, 'w') as outfile:
            json.dump([int(getattr(region, 'invalid', False))
                       for region in self._regions], outfile)
        print(f"Saved annotations to '{filename}'")
        self._annotationsChanged = False


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


# FIXME[old]: not used anymore
# (hower, setMetadata is called in qtgui/panels/face.py and also
# defined in qtgui/widgets/image.py -> clean up all of it ...
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


class QDataInspector(QWidget, QObserver, qattributes={
        Toolbox: False, Datafetcher: False}):
    """A widget for selecting and viewing data items. This is essentially
    a combination of a :py:class:`QDataView` and a
    :py:class:`QDatasourceNavigator`.

    A :py:class:`QDataInspector` can include a
    :py:class:`QDatasourceNavigator` that allows to select a
    :py:class:`Data` object from a :py:class:`Datasource`.

    A :py:class:`QDataInspector` can be associated with a
    :py:class:`Toolbox`. In this case, the datasource navigator
    (if any) will navigate the current datasource of the toolbox.

    Arguments
    ---------
    toolbox:
        If associated with a :py:class:`Toolbox`, the
        :py:class:`QDataInspector` will select data from
        the current :py:class:`Datasource` of that :py:class:`Toolbox`.

    datafetcher:
        The :py:class:`Datafetcher` object to be used by this
        :py:class:`QDataInspector` to fetch data from the underlying
        :py:class:`Datasource`. If none is provided, the
        :py:class:`QDataInspector` will create its own instance.
        This datafetcher will be propagated to the :py:class:`QDataView`
        and to the :py:class:`QDatasourceNavigator`, so that fetching
        data (either initiated by some button from the
        :py:class:`QDatasourceNavigator` or programmatically) will
        automatically update these components.
    """

    def __init__(self, toolbox: Toolbox = None,
                 datafetcher: Datafetcher = None,
                 orientation: Qt.Orientation = Qt.Vertical,
                 datasource_selector: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._experimental = False  # FIXME[hack]
        self._orientation = orientation
        self._initUI(datasource_selector)
        self._layoutUI()
        self.setToolbox(toolbox)
        if datafetcher is None:
            datafetcher = Datafetcher()
        self.setDatafetcher(datafetcher)

    def _initUI(self, datasource_selector: bool) -> None:
        self._dataView = QDataView(orientation=self._orientation)
        self.addAttributePropagation(Toolbox, self._dataView)
        self.addAttributePropagation(Datafetcher, self._dataView)

        self._datasourceNavigator = \
            QDatasourceNavigator(datasource_selector=datasource_selector)
        self.addAttributePropagation(Toolbox, self._datasourceNavigator)
        self.addAttributePropagation(Datafetcher, self._datasourceNavigator)

        if self._experimental:
            self._button = QPushButton("Change Layout")
            self._button.clicked.connect(self._onButtonClicked)

    def _layoutUI(self) -> None:
        """Layout the user interface. There are different ways in which
        the components of a :py:class:`QDataInspector` can be arranged.
        """
        if self._orientation == Qt.Horizontal:
            row = QHBoxLayout()
            row.addWidget(self._dataView)
            row.addWidget(self._datasourceNavigator)
            if self._experimental:
                row.addWidget(self._button)
            self.setLayout(row)
        else:  # if self._orientation == Qt.Vertical:
            column = QVBoxLayout()
            column.addWidget(self._dataView)
            column.addWidget(self._datasourceNavigator)
            if self._experimental:
                column.addWidget(self._button)
            self.setLayout(column)

    def _updateUI(self) -> None:
        """Update the layout of the user interface.
        """
        # Remove the 3 items (dataView, datasourceNavigator, and button)
        # from the layout, and then create a new layout by calling
        # _layoutUI().

        # QLayout::removeItem() just removes the item from layout, but
        # does not hide or delete it.
        layout = self.layout()
        if self._experimental:
            layout.removeItem(layout.itemAt(2)) # self._button
        layout.removeItem(layout.itemAt(1))  # self._datasourceNavigator
        layout.removeItem(layout.itemAt(0))  # self._dataView
        # self.layout().removeWidget(self._dataView)
        # self.layout().removeWidget(self._datasourceNavigator)
        # self.layout().removeWidget(self._button)
        # Reparent the current layout to a dummy widget, which will
        # be delete immedieatly as we do not store a reference.
        # https://stackoverflow.com/a/10439207
        QWidget().setLayout(layout)
        self._layoutUI()
        self.update()

    def dataView(self) -> QDataView:
        """The :py:class:`QDataView` used by this
        :py:class:`QDataInspector`.
        """
        return self._dataView

    def datasourceNavigator(self) -> QDatasourceNavigator:
        """The :py:class:`QDatasourceNavigator` of
        this :py:class:`QDataInspector`.
        """
        return self._datasourceNavigator

    @protect
    def _onButtonClicked(self, _checked: bool) -> None:
        if self._orientation == Qt.Horizontal:
            self._orientation = Qt.Vertical
        elif self._orientation == Qt.Vertical:
            self._orientation = Qt.Horizontal
        self._updateUI()

    def imageView(self) -> QImageView:
        """The :py:class:`QDataView` used by this
        :py:class:`QDataInspector`.
        """
        return self._dataView.imageView()

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
        If this :py:class:`QDataInspector` is associated with
        a :py:class:`Toolbox`, it will set the current datasource
        for that tooblox.

        Arguments
        ---------
        datasource: Datasource
            The new datasource to navigate. If None, the datasource
            navigator will be disabled.
        """
        self._datasourceNavigator.setDatasource(datasource)
