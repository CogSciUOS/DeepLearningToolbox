from controller import ActivationsController
from tools.activation import (Engine as ActivationEngine,
                              Controller as ActivationsController)

from ..utils import QObserver

from PyQt5.QtCore import Qt, QPoint, QRect, QSize, QMargins, pyqtSignal
from PyQt5.QtGui import QPainter, QImage, QPen, QColor, QBrush, QMouseEvent
from PyQt5.QtWidgets import QWidget, QToolTip

from typing import Tuple
import math
import numpy as np

# FIXME[todo]: improve the display of the activations: check that the
# space is used not only in a good, but in an optimal way. Check that
# the aspect ratio is correct. Make it configurable to allow for
# explicitly setting different aspects of display.

# FIXME[todo]: we may display positive and negative activation in a
# two-color scheme.


class QActivationView(QWidget, QObserver, ActivationEngine.Observer):
    '''A widget to display the activations of a given layer in a
    network. Currently there are two types of layers that are
    supported: (two-dimensional) convolutional layers and dense
    (=fully connected) layers.

    The :py:class:``QActivationView`` widget allows to select an
    individual unit in the network layer by a single mouse click (this
    will either select a single unit in a dense layer, or a channel in
    a convolutional layer). The selection can be moved with the cursor
    keys and the unit can be deselected by hitting escape. The widget
    will signal such a (de)selection by emitting the 'selected'
    signal.

    The :py:class:``QActivationView`` will try to make good use of the
    available space by arranging and scaling the units. However, the
    current implementation is still suboptimal and may be improved to
    allow for further configuration.

    Attributes
    -----------
    _activations: np.ndarray
        The activation values to be displayed in this
        activation view. ``None`` means that no activation
        is assigned to this :py:class:``QActivationView``
        and will result in an empty widget.

    _units: int
        Number of units in this view. This is simply the length
        of the activations array.

    _currentUnit: int
        The currently selected unit. The value None means that
        no unit is currently selected.

    _currentPosition: QPoint
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
    '''

    _activationController: ActivationsController = None

    def __init__(self, parent: QWidget=None) -> None:
        '''Initialization of the QActivationView.

        Parameters
        -----------
        parent  :   QtWidget
                    Parent widget (passed to super)
        '''
        super().__init__(parent)

        self._activations = None
        self._units = 0
        self._isConvolution = False
        self._currentUnit = None
        self._currentPosition = None
        self._unitSize = None
        self._padding = 2
        self._unitDisplaySize = None

        # By default, a QWidget does not accept the keyboard focus, so
        # we need to enable it explicitly: Qt.StrongFocus means to
        # get focus by 'Tab' key as well as by mouse click.
        self.setFocusPolicy(Qt.StrongFocus)

        self.setToolTip(False)

    def setActivationController(self, activation: ActivationsController):
        interests = ActivationEngine.Change('activation_changed',
                                            'unit_changed')
        self._exchangeView('_activationController', activation,
                           interests=interests)

    def activation_changed(self, engine: ActivationEngine,
                           info: ActivationEngine.Change) -> None:
        '''Get the current activations from the ActivationEngine and
        set the activations to be displayed in this QActivationView.
        Currently there are two possible types of activations that are
        supported by this widget: 1D, and 2D convolutional.

        '''
        if engine is None:
            self._activations = None
            self._units = 0
            self._isConvolution = False
            self._currentUnit = None
            self._currentPosition = None
            self._unitSize = None
            self._unitDisplaySize = None
            return

        # get activation and update overlay only when
        # significant properties change
        if info & {'activation_changed', 'layer_changed'}:
            # activation is also changed when a new network is selected ...
            try:
                activation = engine.get_activation()
            except ValueError:
                activation = None

            if activation is not None:
                if activation.dtype != np.float32:
                    raise ValueError('Activations must be floats.')
                if activation.ndim not in {1, 3}:
                    raise ValueError(f'Unexpected shape {activation.shape}')

            if activation is not None:
                self._isConvolution = (activation.ndim == 3)

                if self._isConvolution:
                    # for convolution we want activtation to be of shape
                    # (output_channels, width, height) but it comes in
                    # (width, height, output_channels)
                    activation = activation.transpose([2, 0, 1])

                from util.image import grayscaleNormalized
                # a contiguous array is important for display with Qt
                activation = \
                    np.ascontiguousarray(grayscaleNormalized(activation))

            self._activations = activation
            self._updateGeometry()

        if info.unit_changed:
            # unit_changed includes also a change of position
            self._currentUnit = engine.unit
            self._currentPosition = (None if engine.position is None else
                                     QPoint(engine.position[1],
                                            engine.position[0]))
            self.update()

    def setUnit(self, unit: int=None, position: QPoint=None) -> None:
        """Set the current unit for this activations view.
        """
        pos = None if position is None else (position.y(), position.x())
        self._activationController.onUnitSelected(unit, pos)

    def setToolTip(self, active: bool=True) -> None:
        '''Turn on/off the tooltips for this Widget.
        The tooltip will display the index and value for the matrix
        at the current mouse position.

        Parameters
        ----------
        active: bool
            Will turn the tooltips on or off.
        '''
        self.toolTipActive = active
        self.setMouseTracking(self.toolTipActive)

        if not self.toolTipActive:
            QToolTip.hideText()

    # FIXME[old]: should be provided by the model, not by the view ...
    def getUnitActivation(self, unit: int=None) -> np.ndarray:
        '''Get the activation mask for a given unit.

        Parameters
        -----------
        unit    :   int
                    The unit/channel index to get the activation from

        Returns
        -------
        np.ndarray or np.uint8
                The scalar unit activation for dense layers or the array of
                activations (the channel) for 2d conv layers

        '''
        if self._activations is None or unit is None:
            return None
        elif not self._isConvolution:
            return None
        else:
            if unit is None:
                unit = self._currentUnit
            return self._activations[unit]

    def _updateGeometry(self) -> None:
        '''Update the private attributes containing geometric information on
        the currently selected activation map. This method should be
        called whenever this :py:class:`QActivationView` is adapted in
        a way that affects its geometric properties.
        '''
        if self._activations is None:
            self._rows = None
            self._columns = None
            self._unitDisplaySize = None
        else:
            # In case of a convolutional layer, the axes of activation are:
            # (width, height, output_channels)
            # For fully connected (i.e., dense) layers, the axes are:
            # (units,)
            activation = self._activations
            self._isConvolution = (activation.ndim == 3)
            self._units = activation.shape[0]

            if self._isConvolution:
                self._unitSize = QSize(activation.shape[2], activation.shape[1])
                unitRatio = self._unitSize.width() / self._unitSize.height()
            else:
                self._unitSize = QSize(1, 1)
                unitRatio = 1

            # FIXME: implement better computation!
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

    def _getUnitDisplayCorner(self, unit: int, padding: int=0) -> QPoint:
        '''Get the display coordinates of the left upper corner of a unit.

        '''
        if unit is None or self._unitDisplaySize is None:
            return None
        corner = \
            QPoint(self._unitDisplaySize.width() * (unit % self._columns),
                   self._unitDisplaySize.height() * (unit // self._columns))
        if padding:
            corner += QPoint(padding, padding)
        return corner

    def _getUnitDisplayRect(self, unit: int, padding: int=0) -> QRect:
        '''Get the rectangle (screen position and size) occupied by the given
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
        '''
        if unit is None or self._unitDisplaySize is None:
            return None
        rect = QRect(self._getUnitDisplayCorner(unit, self._padding),
                     self._unitDisplaySize)
        if padding:
            rect -= QMargins(padding, padding, padding, padding)
        return rect

    def _getPositionDisplayRect(self, unit: int, position: QPoint) -> QPoint:
        '''Get the rectangle (screen position and size) occupied by the given
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
        '''
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
        '''Compute the entry corresponding to some point in this widget.

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
        '''

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

    def _drawConvolution(self, qp):
        '''Draw activation values for a convolutional layer in the form of
        ``QImage``s.

        Parameters
        ----------
        qp: QPainter

        '''
        for unit in range(self._units):
            rect = self._getUnitDisplayRect(unit, self._padding)
            if rect is not None:
                image = QImage(self._activations[unit],
                               self._unitSize.width(), self._unitSize.height(),
                               self._unitSize.width(),
                               QImage.Format_Grayscale8)
                qp.drawImage(rect, image)

    def _drawDense(self, qp):
        '''Draw activation values for a dense layer in the form of rectangles.

        Parameters
        ----------
        qp: QPainter
        '''
        for unit, value in enumerate(self._activations):
            qp.fillRect(self._getUnitDisplayRect(unit, self._padding),
                        QBrush(QColor(value, value, value)))

    def _drawSelection(self, qp):
        '''Mark the currently selected unit in the painter.

        Parameters
        ----------
        qp : QPainter
        '''
        pen_width = 4
        pen_color = Qt.red
        pen = QPen(pen_color)
        pen.setWidth(pen_width)
        qp.setPen(pen)
        qp.drawRect(self._getUnitDisplayRect(self._currentUnit, 0))

    def _drawPosition(self, qp):
        rect = self._getPositionDisplayRect(self._currentUnit,
                                            self._currentPosition)
        if rect is not None:
            pen_width = 1
            pen_color = Qt.green
            pen = QPen(pen_color)
            pen.setWidth(pen_width)
            qp.setPen(pen)
            qp.drawRect(rect)

    def _drawNone(self, qp):
        '''Draw a view when no activation values are available.

        Parameters
        ----------
        qp : QPainter
        '''
        qp.drawText(self.rect(), Qt.AlignCenter, 'No data!')

    def paintEvent(self, event):
        '''Process the paint event by repainting this Widget.

        Parameters
        -----------
        event : QPaintEvent
        '''
        qp = QPainter()
        qp.begin(self)
        if self._activations is None:
            self._drawNone(qp)
        elif self._isConvolution:
            self._drawConvolution(qp)
        else:
            self._drawDense(qp)
        if self._currentUnit is not None:
            self._drawSelection(qp)
        if self._currentPosition is not None:
            self._drawPosition(qp)
        qp.end()

    def resizeEvent(self, event):
        '''Adapt to a change in size. The behavior dependes on the zoom
        policy.  This event handler is called after the Widget has
        been resized.  providing the new .size() and the old
        .oldSize().

        Parameters
        ----------
        event : QResizeEvent

        '''
        self._updateGeometry()

    def mousePressEvent(self, event):
        '''Process mouse event. A mouse click will select (or deselect)
        the unit under the mouse pointer.

        Parameters
        ----------
        event : QMouseEvent
        '''
        unit, position = self._unitAtDisplayPosition(event.pos())
        if event.button() == Qt.LeftButton:
            if unit is not None:
                if unit == self._currentUnit:
                    # clicking the active unit -> toggle position
                    if self._currentPosition == position:
                        position = None
                else:
                    # selecting a new unit -> turn off position
                    position = None
        elif event.button() == Qt.RightButton:
            position = None
            if self._currentPosition is None:
                unit = None
        self.setUnit(unit, position)

    # The widget will also in addition to the double click event
    # receive mouse press (before) and mouse release events
    # (afterwards).  It is up to the developer to ensure that the
    # application interprets these events correctly.
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        '''Process a double click. We use double click to deselect a
        unit.

        Parameters
        ----------
        event : QMouseEvent
        '''
        if event.button() == Qt.LeftButton:
            unit, position = self._unitAtDisplayPosition(event.pos())
            if unit == self._currentUnit:
                if (self._currentPosition is None or
                        self._currentPosition == position):
                    # double click: deactivate the unit
                    position = None
                    unit = None
            self.setUnit(unit, position)

    def mouseReleaseEvent(self, event):
        '''Process mouse event.

        Parameters
        ----------
        event : QMouseEvent
        '''
        # As we implement .mouseDoubleClickEvent(), we
        # also provide stubs for the other mouse events to not confuse
        # other widgets.
        pass

    # Attention: The mouseMoveEvent() is only called for regular mouse
    # movements, if mouse tracking is explicitly enabled for this
    # widget by calling self.setMouseTracking(True).  Otherwise it may
    # be called on dragging.
    def mouseMoveEvent(self, event):
        '''Process mouse movements.  If tooltips are active, information on
        the entry at the current mouse position are displayed.

        Parameters
        ----------
        event : QMouseEvent
            The mouse event, providing local and global coordinates.
        '''
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

    def keyPressEvent(self, event):
        '''Process special keys for this widget.  Allow moving selected entry
        using the cursor keys. Deselect unit using the Escape key.

        Parameters
        ----------
        event : QKeyEvent

        '''
        unit = self._currentUnit
        position = (None if self._currentPosition is None else
                    QPoint(self._currentPosition))
        key = event.key()
        shiftPressed = event.modifiers() & Qt.ShiftModifier

        # Space will toggle display of tooltips
        if key == Qt.Key_Space:
            self.setToolTip(not self.toolTipActive)
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
                self._currentPosition = None
            else:
                event.ignore()
        elif unit is not None:
            row = self._currentUnit // self._columns
            col = self._currentUnit % self._columns
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
                unit = self._currentUnit
        else:
            event.ignore()

        if unit != self._currentUnit or position != self._currentPosition:
            self.setUnit(unit, position)
