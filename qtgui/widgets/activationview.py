"""Widgets for displaying network activations.
"""

# standard imports
from typing import Tuple
import math
import logging

# third party imports
import numpy as np

# Qt imports
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, QMargins, pyqtSignal
from PyQt5.QtGui import (QPainter, QImage, QPen, QColor, QBrush,
                         QMouseEvent, QPaintEvent, QResizeEvent, QKeyEvent)
from PyQt5.QtWidgets import QWidget, QToolTip

# toolbox imports
from network import Layer
from dltb.tool.activation import ActivationWorker
from dltb.util.array import DATA_FORMAT_CHANNELS_FIRST
from dltb.util.image import grayscaleNormalized
from dltb.util.error import print_exception

# GUI imports
from ..utils import QObserver, protect

# logging
LOG = logging.getLogger(__name__)

# FIXME[todo]: improve the display of the activations: check that the
# space is used not only in a good, but in an optimal way. Check that
# the aspect ratio is correct. Make it configurable to allow for
# explicitly setting different aspects of display.

# FIXME[todo]: we may display positive and negative activation in a
# two-color scheme. This probably requires some thought: currently
# activation values are converted to uint8 - for a two-color scheme,
# we should keep negative values ...

# FIXME[todo]: there is also some room for improvement with local
# contrast. At least a better documentation, but also some additional
# thoughts - it may for example be an option also to weigh the
# activation with the connection weights of a selected unit in the
# next layer - that is, we should offer a way to provide an API for
# such weighing strategies

# FIXME[todo]: a further thought might be an option to deactivate
# units or cells. This goes a bit in the direction of occlusion, so
# one should think what would be the best way to provide such
# functionality.


class QActivationView(QWidget, QObserver, qobservables={
        ActivationWorker: {'data_changed', 'work_finished'}}):
    # pylint: disable=too-many-instance-attributes
    """A widget to display the activations of a given
    :py:class:`Layer` in a :py:class:`Network`.
    Currently there are two types of layers that are
    supported: (two-dimensional) convolutional layers and dense
    (=fully connected) layers.

    The :py:class:``QActivationView`` widget allows to select an
    individual unit in the network layer by a single mouse click (this
    will either select a single unit in a dense layer, or a channel in
    a convolutional layer). The selection can be moved with the cursor
    keys and the unit can be deselected by hitting escape. The widget
    will signal such a (de)selection by emitting the 'selected'
    signal.

    In a convolution layer it is also possible to select an individual
    position in an activation map, by again clicking into the selected
    unit. Such a selection will be signaled by emitting 'positionChanged'.

    The :py:class:``QActivationView`` will try to make good use of the
    available space by arranging and scaling the units. However, the
    current implementation is still suboptimal and may be improved to
    allow for further configuration.

    Attributes
    -----------
    _activationWorker: ActivationWorker = None

    _activations: np.ndarray
        The activation values to be displayed in this
        activation view. ``None`` means that no activation
        is assigned to this :py:class:``QActivationView``
        and will result in an empty widget.

    _layer: Layer
        The network :py:class:`Layer` currently displayed
        in this :py:class:`QActivationView`.

    _units: int
        Number of units in this view. This is simply the length
        of the activations array.

    _rows: int
        Number of rows used for displaying the units. This value
        is computed automatically from the widget size and the
        number of units to display.

    _columns: int
        Number of columns used for displaying the units. This value
        is computed automatically from the widget size and the
        number of units to display.

    _unit: int
        The currently selected unit. The value None means that
        no unit is currently selected.

    _position: QPoint
        The currently selected position in the activation map of
        the current unit.

    _isConvolution: bool
        A flag indicating if the current QActivationView is
        currently in convolution mode (True) or not (False).

    _unitSize: QSize
        The size of of a unit (its activation map). This will be (1,1)
        for a fully connected layer and the shape of the activation map
        for a convolutional layer.

    _padding: int
        Padding between the individual units in this
        :py:class:``QActivationView``.

    _unitDisplaySize: QSize
        The width of a unit in pixels, including padding.

    toolTipActive : bool
        A flag indicating whether tooltips shoould be shown.

    _localContrast: bool
        A flag indicatint if activation maps (in convolutional layers)
        should be shown with local contrast. Local constrast can be
        toggled with the 'L' key.

    _colorScale: bool
        A flag indicatint if activation maps (in convolutional layers)
        are displayed using a color scale. Color scale can be
        toggled with the 'C' key.
    """

    unitChanged = pyqtSignal(int)
    positionChanged = pyqtSignal(QPoint)

    def __init__(self, worker: ActivationWorker = None, **kwargs) -> None:
        """Initialization of the QActivationView.

        Parameters
        ----------
        parent: QtWidget
            Parent widget (passed to super)
        """
        super().__init__(**kwargs)

        self._activations = None
        self._activationWorker = None
        self._layer = None
        self._units = 0
        self._rows = None
        self._columns = None
        self._isConvolution = False
        self._unit = None
        self._position = None
        self._unitSize = None
        self._padding = 2
        self._unitDisplaySize = None
        self._localContrast = False
        self._colorScale = False

        # By default, a QWidget does not accept the keyboard focus, so
        # we need to enable it explicitly: Qt.StrongFocus means to
        # get focus by 'Tab' key as well as by mouse click.
        self.setFocusPolicy(Qt.StrongFocus)
        self.setToolTip(False)
        self.setActivationWorker(worker)

    def setActivations(self, activations: np.ndarray) -> None:
        """Set the activations to be displayed in this
        :py:class:`QActivationView`.
        """
        if activations is not None:
            if activations.dtype != np.float32:
                raise ValueError('Activations must be floats.')
            if activations.ndim not in {1, 3}:
                raise ValueError(f'Unexpected shape {activations.shape}')

        if activations is not None:
            self._isConvolution = (activations.ndim == 3)

            # this is a uint8 array, globally normalized
            activations = grayscaleNormalized(activations)
            # a contiguous array is important for display with Qt
            activations = np.ascontiguousarray(activations)

        self._activations = activations
        self._updateGeometry()


    def layer(self) -> Layer:
        """The layer for which activations are currently displayed in this
        :py:class:`QActivationView`. None if no :py:class:`Layer` is
        set.
        """
        return self._layer

    def setLayer(self, layer: Layer) -> None:
        """Set the :py:class:`Layer` which activation should be
        displayed in this :py:class:`QActivationView`.

        Parameters
        ----------
        layer: Layer
            The network :py:class:`Layer`. It is assumed that this
            layer belongs th the current
            `py:meth:`ActivationWorker.tool.network`
            of the :py:class:`ActivationTool`.
        """
        LOG.info("QActivationView.setLayer(%s)", layer)
        if self._layer is not None:
            if self._activationWorker is not None:
                self._activationWorker.remove_layer(self._layer)
        self._layer = layer
        if layer is None:
            self._units = 0
            self._isConvolution = False
            self._unitSize = None
            self._unitDisplaySize = None
            self._activations = None
        else:
            if self._activationWorker is not None:
                self._activationWorker.add_layer(self._layer)
        self.setUnit(None)
        self.update()

    def unit(self) -> int:
        """The currently selected unit in this :py:class:`QActivationView`.
        None if no unit is selected.
        """
        return self._unit

    def setUnit(self, unit: int) -> None:
        """Set the current unit for this  :py:class:`QActivationView`.
        A change of unit will trigger the `unitChanged` signal.

        Parameters
        ----------
        unit: int
            The unit to select. This has to be a valid unit in the
            current layer or None to deselect the unit. Deselecting
            the unit will also clear the position.

        Raises
        ------
        ValueError:
            The given unit is not valid for the current :py:meth:`layer`.
        """
        if unit == self._unit:
            return  # nothing changed

        # if self._layer is None:
        #     raise ValueError("Cannot set a unit without a Layer.")

        # FIXME[todo]: Layer does not implement __len__
        if unit is not None and not 0 <= unit:  # < len(self._layer):
            raise ValueError(f"Invalid unit {unit}: "
                             #f"possible values are 0 to {len(self._layer)-1}"
                             )

        self._unit = unit

        # Deselecting the unit will also clear the position
        if unit is None:
            self.setPosition(None)

        # We are not allow to sent a None signal - hence we use
        # -1 to indicate that no position is set
        self.unitChanged.emit(-1 if unit is None else unit)
        self.update()

    def position(self) -> QPoint:
        """The currently selected position in a two-dimension unit in this
        :py:class:`QActivationView`.  None if no position is selected.
        """
        return self._position

    def setPosition(self, position: QPoint) -> None:
        """Select a position in a two-dimension unit in this
        :py:class:`QActivationView`.
        Currently a position can only be set for two-dimensional
        activation maps.

        Parameters
        ----------
        Position: QPoint
            The position to select. This has to be a valid position in the
            activation map of the current layer or None to select no position.

        Raises
        ------
        ValueError:
            The given position is not valid for the current :py:meth:`layer`.
        """
        if position == self._position:
            return  # nothing changed

        # if self._layer is None:
        #     raise ValueError("Cannot set a position without a Layer.")

        # FIXME[todo]: provide information on layer geometry
        #
        # if position is not None:
        #     if and not layer.dimensions == 2:
        #         raise ValueError("Cannot set a position "
        #                          f"for layer {self._layer}")
        #
        #     if not position "FIXME[todo]: is valid for layer":
        #         raise ValueError(f"Invalid position {position} "
        #                          f"for layer {self._layer} "
        #                          f"of shape {self._layer.FIXME_shape}")

        self._position = position
        # We are not allow to sent a None signal - hence we use
        # QPoint(-1, -1) to indicate no position
        self.positionChanged.emit(QPoint(-1, -1) if position is None
                                  else position)
        self.update()

    def localContrast(self) -> bool:
        """Check if this :py:class:`QActivationView` is using local
        contrast for displaying activation maps of convolutional layers.
        If True, the activation map for each feature is stretch to
        use the full range of available values for display.
        If False (the default behaviour) the same scaling is used for
        all activation maps of a layer.
        """
        return self._localContrast

    def setLocalContrast(self, flag: bool = True):
        """Set local contrast for displaying activation maps
        of convolutional layers (contrast is computed for each feature
        of that layer separately). The default behaviour is to use
        a global contrast.
        """
        self._localContrast = flag
        self.update()

    def colorScale(self) -> bool:
        """Check if this :py:class:`QActivationView` is using a color
        scale for displaying the activations.
        """
        return self._colorScale

    def setColorScale(self, flag: bool = True):
        """Set color scale activation display for
        this :py:class:`QActivationView`.
        """
        self._colorScale = flag
        self.update()

    def worker_changed(self, worker: ActivationWorker,
                       info: ActivationWorker.Change) -> None:
        # pylint: disable=invalid-name
        """Get the current activations from the ActivationWorker and
        set the activations to be displayed in this
        :py:class:`QActivationView`.
        Currently there are two possible types of activations that are
        supported by this widget: 1D, and 2D convolutional.

        """
        LOG.debug("QActivationView.worker_changed(%s)", info)

        if self._layer is None:
            return  # we are not interestend in any activation

        # get activation and update overlay only when
        # significant properties change
        if info.work_finished:
            try:
                # FIXME[bug]: in loop mode it may happen that the
                # worked data has already been replaced by new data
                # when this code is executed. In this case a
                # TypeError ('NoneType' object is not subscriptable)
                # is raised. We may adapt the worker to avoid this.

                # for convolution we want activation to be `channels_first`
                # that is of shape (output_channels, width, height)
                activations = worker.activations(self._layer, data_format=
                                                 DATA_FORMAT_CHANNELS_FIRST)
            except (ValueError, TypeError) as error:
                #LOG.warning("QActivationView.worker_changed: error=%s", error)
                print_exception(error)
                activations = None
            LOG.debug("QActivationView.worker_changed: type=%s",
                      type(activations))
            self.setActivations(activations)

        LOG.debug("QActivationView.worker_changed: units=%s/%s, "
                  "isConvolution=%s, currentPosition=%s, size/display=%s/%s",
                  self._unit, self._units, self._isConvolution,
                  self._position, self._unitSize, self._unitDisplaySize)

    def setToolTip(self, active: bool = True) -> None:
        """Turn on/off the tooltips for this Widget.
        The tooltip will display the index and value for the matrix
        at the current mouse position.

        Parameters
        ----------
        active: bool
            Will turn the tooltips on or off.
        """
        self.toolTipActive = active
        self.setMouseTracking(self.toolTipActive)

        if not self.toolTipActive:
            QToolTip.hideText()

    def _updateGeometry(self) -> None:
        """Update the private attributes containing geometric information on
        the currently selected activation map. This method should be
        called whenever this :py:class:`QActivationView` is adapted in
        a way that affects its geometric properties.
        """
        if self._activations is None:
            self._rows = None
            self._columns = None
            self._unitDisplaySize = None
        else:
            # In case of a convolutional layer, the axes of activation are:
            # (width, height, output_channels)
            # For fully connected (i.e., dense) layers, the axes are:
            # (units,)
            activations = self._activations
            self._isConvolution = (activations.ndim == 3)
            self._units = activations.shape[0]

            if self._isConvolution:
                self._unitSize = QSize(activations.shape[2],
                                       activations.shape[1])
                unitRatio = self._unitSize.width() / self._unitSize.height()
            else:
                self._unitSize = QSize(1, 1)
                unitRatio = 1

            # FIXME[todo]: implement better computation!
            # - allow for rectangular (i.e. non-quadratic) widget size
            # - allow for rectangular (i.e. non-quadratic) convolution filters
            # and maintain aspect ratio ...
            pixelsPerUnit = (self.width() * self.height()) / self._units

            unitHeight = math.floor(math.sqrt(pixelsPerUnit / unitRatio))
            self._rows = math.ceil(self.height() / unitHeight)

            unitWidth = math.floor(unitRatio * unitHeight)
            self._columns = math.ceil(self.width() / unitWidth)

            self._unitDisplaySize = \
                QSize(math.floor(self.width() / self._columns),
                      math.floor(self.height() / self._rows))
        self.update()

    def _getUnitDisplayCorner(self, unit: int, padding: int = 0) -> QPoint:
        """Get the display coordinates of the left upper corner of a unit.

        """
        if unit is None or self._unitDisplaySize is None:
            return None
        corner = \
            QPoint(self._unitDisplaySize.width() * (unit % self._columns),
                   self._unitDisplaySize.height() * (unit // self._columns))
        if padding:
            corner += QPoint(padding, padding)
        return corner

    def _getUnitDisplayRect(self, unit: int, padding: int = 0) -> QRect:
        """Get the rectangle (screen position and size) occupied by the given
        unit.

        Parameters
        ----------
        unit    :   index of the unit of interest
        padding :   padding of the unit.  If None is given, standard padding
                    value of this QActivationView will be used.

        Returns
        -------
        QRect
            The rectangle occupied by the unit, excluding the padding.
        """
        if unit is None or self._unitDisplaySize is None:
            return None
        rect = QRect(self._getUnitDisplayCorner(unit, self._padding),
                     self._unitDisplaySize)
        if padding:
            rect -= QMargins(padding, padding, padding, padding)
        return rect

    def _getPositionDisplayRect(self, unit: int, position: QPoint) -> QPoint:
        """Get the rectangle (screen position and size) occupied by the given
        position (in the given) unit.

        Parameters
        ----------
        unit: int
            Index of the unit of interest
        position: QPoint
            Position in the activation map of that unit.

        Returns
        -------
        position: QRect
            The rectangle occupied by the unit, excluding the padding.
        """
        if unit is None or position is None or self._unitDisplaySize is None:
            return None

        corner = self._getUnitDisplayCorner(unit, self._padding)

        unitDisplaySizeWithoutPadding = QSize(self._unitDisplaySize) \
            - QSize(2 * self._padding, 2 * self._padding)

        offset = QPoint((unitDisplaySizeWithoutPadding.width() *
                         position.x()) // self._unitSize.width(),
                        (unitDisplaySizeWithoutPadding.height() *
                         position.y()) // self._unitSize.height())
        corner += offset

        size = QSize(unitDisplaySizeWithoutPadding.width() //
                     self._unitSize.width(),
                     unitDisplaySizeWithoutPadding.height() //
                     self._unitSize.height())

        rect = QRect(corner, size)
        rect += QMargins(1, 1, 1, 1)
        return rect

    def _unitAtDisplayPosition(self, pos: QPoint) -> Tuple[int, QPoint]:
        """Compute the entry corresponding to some point in this widget.

        Parameters
        ----------
        pos: QPoint
            The position of the point in question (in Widget
            position).

        Returns
        -------
        unit: int
            The unit occupying that position or ``None`` if no entry
            corresponds to that position.
        position: QPoint
            The position inside the activation map of the selected unit
            or None if no valid unit is at that position.
        """

        if self._activations is None:
            return None, None
        unit = ((pos.y() // self._unitDisplaySize.height()) * self._columns +
                (pos.x() // self._unitDisplaySize.width()))
        if unit >= self._units:
            # selected something which may be on the grid, but where there's no
            # unit anymore
            unit = None

        if unit is None:
            return None, None

        row, column = (unit // self._columns), (unit % self._columns)
        corner = QPoint(self._unitDisplaySize.width() * column,
                        self._unitDisplaySize.height() * row)
        padding = 0 if self._padding is None else self._padding
        corner += QPoint(padding, padding)
        position = QPoint(((pos - corner).x() * self._unitSize.width()) /
                          (self._unitDisplaySize.width()-2*padding),
                          ((pos - corner).y() * self._unitSize.height()) /
                          (self._unitDisplaySize.height()-2*padding))
        return unit, position

    def _drawConvolution(self, painter: QPainter):
        """Draw activation values for a convolutional layer in the form of
        ``QImage``s.

        Parameters
        ----------
        painter: QPainter

        """
        for unit in range(self._units):
            rect = self._getUnitDisplayRect(unit, self._padding)
            if rect is None:
                continue

            activations = self._activations[unit]
            imageFormat = QImage.Format_Grayscale8
            bytesPerLine = self._unitSize.width()

            if self.localContrast():
                # increase local contrast for this unit
                amin, amax = activations.min(), activations.max()
            else:
                amin, amax = 0, 255

            if self.colorScale():
                mapping = np.linspace([255, 0, 0], [0, 0, 255],
                                      1+amax-amin,
                                      dtype=np.uint8, endpoint=True)
                activations = mapping[activations-amin]
                imageFormat = QImage.Format_RGB888
                bytesPerLine *= 3

            elif self.localContrast():
                mapping = np.linspace(0, 255, 1+amax-amin,
                                      dtype=np.uint8, endpoint=True)
                activations = mapping[activations-amin]

            image = QImage(activations,
                           self._unitSize.width(), self._unitSize.height(),
                           bytesPerLine, imageFormat)
            painter.drawImage(rect, image)

    def _drawDense(self, painter: QPainter):
        """Draw activation values for a dense layer in the form of rectangles.

        Parameters
        ----------
        painter: QPainter
        """
        for unit, value in enumerate(self._activations):
            painter.fillRect(self._getUnitDisplayRect(unit, self._padding),
                             QBrush(QColor(value, value, value)))

    def _drawSelection(self, painter: QPainter):
        """Mark the currently selected unit in the painter.

        Parameters
        ----------
        painter : QPainter
        """
        penWidth = 4
        penColor = Qt.red
        pen = QPen(penColor)
        pen.setWidth(penWidth)
        painter.setPen(pen)
        painter.drawRect(self._getUnitDisplayRect(self._unit, 0))

    def _drawPosition(self, painter: QPainter):
        rect = self._getPositionDisplayRect(self._unit,
                                            self._position)
        if rect is not None:
            penWidth = 1
            penColor = Qt.green
            pen = QPen(penColor)
            pen.setWidth(penWidth)
            painter.setPen(pen)
            painter.drawRect(rect)

    def _drawNone(self, painter: QPainter):
        """Draw a view when no activation values are available.

        Parameters
        ----------
        painter : QPainter
        """
        painter.drawText(self.rect(), Qt.AlignCenter, "No data,\n"
                         f"layer='{self._layer.key if self._layer else None}',"
                         f"\nworker={self._activationWorker is not None}!")

    def paintEvent(self, _: QPaintEvent) -> None:
        """Process the paint event by repainting this Widget.

        Parameters
        -----------
        event: QPaintEvent
        """
        painter = QPainter()
        painter.begin(self)
        if self._activations is None:
            self._drawNone(painter)
        elif self._isConvolution:
            self._drawConvolution(painter)
        else:
            self._drawDense(painter)
        if self._unit is not None:
            self._drawSelection(painter)
        if self._position is not None:
            self._drawPosition(painter)
        painter.end()

    def resizeEvent(self, _: QResizeEvent):
        """Adapt to a change in size. The behavior dependes on the zoom
        policy.  This event handler is called after the Widget has
        been resized.  providing the new .size() and the old
        .oldSize().

        Parameters
        ----------
        event : QResizeEvent

        """
        self._updateGeometry()

    @protect
    def mousePressEvent(self, event):
        """Process mouse event. A mouse click will select (or deselect)
        the unit under the mouse pointer.

        Parameters
        ----------
        event : QMouseEvent
        """
        unit, position = self._unitAtDisplayPosition(event.pos())
        if event.button() == Qt.LeftButton:
            if unit is None:
                self.setUnit(None)
            elif unit != self._unit:
                # selecting a new unit -> turn off position
                self.setUnit(unit)
                self.setPosition(None)
            else:
                # clicking the active unit -> toggle position
                if self._position == position:
                    position = None
                self.setPosition(position)
        elif event.button() == Qt.RightButton:
            if self._position is None:
                self.setUnit(None)
            else:
                self.setPosition(None)

    # The widget will also in addition to the double click event
    # receive mouse press (before) and mouse release events
    # (afterwards).  It is up to the developer to ensure that the
    # application interprets these events correctly.
    @protect
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Process a double click. We use double click to deselect a
        unit.

        Parameters
        ----------
        event : QMouseEvent
        """
        if event.button() == Qt.LeftButton:
            unit, position = self._unitAtDisplayPosition(event.pos())
            if unit == self._unit:
                if self._position is None or self._position == position:
                    # double click: deactivate the unit
                    self.setUnit(None)
                else:
                    self.setPosition(position)
            else:
                self.setUnit(unit)
                self.setPosition(position)

    @protect
    def mouseReleaseEvent(self, _: QMouseEvent) -> None:
        """Process mouse event.

        Parameters
        ----------
        event: QMouseEvent
        """
        # As we implement .mouseDoubleClickEvent(), we also provide
        # stubs for the other mouse events to not confuse other
        # widgets.

    # Attention: The mouseMoveEvent() is only called for regular mouse
    # movements, if mouse tracking is explicitly enabled for this
    # widget by calling self.setMouseTracking(True).  Otherwise it may
    # be called on dragging.
    @protect
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Process mouse movements.  If tooltips are active, information on
        the entry at the current mouse position are displayed.

        Parameters
        ----------
        event: QMouseEvent
            The mouse event, providing local and global coordinates.
        """
        if self.toolTipActive:
            unit, position = self._unitAtDisplayPosition(event.pos())
            if position is not None:
                text = (f"unit: {unit}, "
                        f"position=({position.x()},{position.y()})")
            elif unit is not None:
                text = f"unit: {unit}"
            else:
                text = "no unit"
            QToolTip.showText(event.globalPos(), text)

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        # pylint: disable=too-many-branches
        """Process special keys for this widget.  Allow moving selected entry
        using the cursor keys. Deselect unit using the Escape key.

        Parameters
        ----------
        event: QKeyEvent

        """
        unit = self._unit
        position = (None if self._position is None else
                    QPoint(self._position))
        key = event.key()
        shiftPressed = event.modifiers() & Qt.ShiftModifier

        # Space will toggle display of tooltips
        if key == Qt.Key_Space:
            self.setToolTip(not self.toolTipActive)
        # 'C' will toggle local contrast
        elif key == Qt.Key_C:
            self.setColorScale(not self.colorScale())
        # 'L' will toggle local contrast
        elif key == Qt.Key_L:
            self.setLocalContrast(not self.localContrast())
        # Arrow keyes will move the selected entry
        elif position is not None and not shiftPressed:
            if key == Qt.Key_Left:
                position.setX(max(0, position.x()-1))
            elif key == Qt.Key_Up:
                position.setY(max(0, position.y()-1))
            elif key == Qt.Key_Right:
                position.setX(min(self._unitSize.width()-1, position.x()+1))
            elif key == Qt.Key_Down:
                position.setY(min(self._unitSize.height()-1, position.y()+1))
            elif key == Qt.Key_Escape:
                self._position = None
            else:
                event.ignore()
            self.setPosition(position)
        elif unit is not None:
            row = self._unit // self._columns
            col = self._unit % self._columns
            if key == Qt.Key_Left:
                col = max(col-1, 0)
            elif key == Qt.Key_Up:
                row = max(row-1, 0)
            elif key == Qt.Key_Right:
                col = min(col+1, self._columns-1)
            elif key == Qt.Key_Down:
                row = min(row+1, self._rows-1)
            elif key == Qt.Key_Escape:
                unit = None
            else:
                event.ignore()
            unit = row * self._columns + col
            if unit >= self._units:
                unit = self._unit
            self.setUnit(unit)
        else:
            event.ignore()

    def debug(self):
        LOG.debug("QActivationView: debug.")
        if hasattr(super(), 'debug'):
            super().debug()
        print(f"debug: QActivationView[{type(self).__name__}]:")
        print(f"debug:   - worker: {self._activationWorker}")
