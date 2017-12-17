import math
import numpy as np
import controller
from util import ArgumentError

from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QImage, QPen, QColor, QBrush
from PyQt5.QtWidgets import QWidget

from observer import Observer

# FIXME[todo]: improve the display of the activations: check that the
# space is used not only in a good, but in an optimal way. Check that
# the aspect ratio is correct. Make it configurable to allow for
# explicitly setting different aspects of display.

# FIXME[todo]: we may display positive and negative activation in a
# two-color scheme.


class QActivationView(QWidget, Observer):
    '''A widget to diplay the activations of a given layer in a
    network. Currently there are two types of layers that are
    supported: (two-dimensional) convolutional layers and dense
    (=fully connected) layers.

    The ``QActivationView`` widget allows to select an individual unit in
    the network layer by a single mouse click (this will either select
    a single unit in a dense layer, or a channel in a convolutional
    layer). The selection can be moved with the cursor keys and the
    unit can be deselected by hitting escape. The widget will signal
    such a (de)selection by emitting the 'selected' signal.

    The ``QActivationView`` will try to make good use of the available
    space by arranging and scaling the units. However, the current
    implementation is still suboptimal and may be improved to allow
    for further configuration.

    Attributes
    -----------
    _activation :   np.ndarray
                    The activation values to be displayed in this activation
                    view. None means that no activation is assigned to this
                    QActivationView and will result in an empty widget.

    _padding    :   int
                    Padding between the individual units in this
                    QActivationView.

    _selectedUnit   :   int
                        The currently selected unit. The value None means that
                        no unit is currently selected.

    _isConvolution  :   bool
                        A flag indicating if the current QActivationView is
                        currently in convolution mode (True) or not (False).

    _selected       :   PyQt5.QtCore.pyqtSignal
                        A signal emitted whenever a unit is (de)selected in this
                        ``QActivationView``. This will be an int (the index of
                        the selected unit) or None (if no unit is selected). We
                        have to use object not int here to allow for None
                        values.
    '''

    _padding:         int = 2
    _isConvolution:   bool = False
    _selected:        pyqtSignal = pyqtSignal(object)
    _controller:      controller.ActivationsController = None

    def __init__(self, parent: QWidget=None):
        '''Initialization of the QActivationView.

        Parameters
        -----------
        parent  :   QtWidget
                    Parent widget (passed to super)
        '''
        super().__init__(parent)

        # By default, a QWidget does not accept the keyboard focus, so
        # we need to enable it explicitly: Qt.StrongFocus means to
        # get focus by 'Tab' key as well as by mouse click.
        self.setFocusPolicy(Qt.StrongFocus)

    def setController(self, controller):
        # TODO: Disconnect before reconnecting?
        super().setController(controller)
        self._selected.connect(controller.on_unit_selected)

    def modelChanged(self, model):
        '''Get the current activations from the model and set the activations to
        be displayed in this QActivationView.  Currently there are two possible
        types of activations that are supported by this widget: 1D, and 2D
        convolutional.

        Parameters
        -----------
        activation  :   np.ndarray
                        Either a 1D or a 3D array. The latter one will be
                        displayed in the convolutional mode. The activation
                        values are expected to be float values. For display they
                        will be scaled and converted to 8-bit integers.
        '''

        activation = model._activation

        if activation is not None:
            if activation.dtype != np.float32:
                raise ArgumentError('Activations must be floats.')
            if activation.ndim not in {1, 3}:
                raise ArgumentError(f'Unexpected shape {activation.shape}')
        old_shape = None if activation is None else activation.shape

        if activation is not None:
            self._isConvolution = (activation.ndim == 3)

            # normalization (values should be between 0 and 1)
            min_value = activation.min()
            max_value = activation.max()
            value_range = max_value - min_value
            activation = (activation - min_value)
            if value_range > 0:
                activation = activation / value_range

            if self._isConvolution:
                # for convolution we want activtation to be of shape
                # (output_channels, width, height) but it comes in
                # (width, height, output_channels)
                activation = activation.transpose([2, 0, 1])

            # change dtype to uint8
            activation = np.ascontiguousarray(
                activation * 255, np.uint8
            )

        # unset selected entry if shape changed
        self._controller.on_unit_selected(None, self)

        self._computeGeometry()


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
        if self._controller._activation is None or unit is None or not self._isConvolution:
            return None
        else:
            if unit is None:
                unit = self._selectedUnit
            return self._controller._activation[unit]

    def _computeGeometry(self):
        if self._controller._activation is None:
            self._rows = None
            self._columns = None
            self._unitWidth = None
            self._unitHeight = None
        else:
            # In case of a convolutional layer, the axes of activation are:
            # (width, height, output_channels)
            # For fully connected (i.e., dense) layers, the axes are:
            # (units,)
            activation = self._controller._activation
            n = activation.shape[0]
            self._isConvolution = (activation.ndim == 3)
            if self._isConvolution:
                unitRatio = activation.shape[2] / activation.shape[1]
                # unitRatio = width/height
            else:
                unitRatio = 1

            # FIXME: implement better computation!
            # - allow for rectangular (i.e. non-quadratic) widget size
            # - allow for rectangular (i.e. non-quadratic) convolution filters
            # and maintain aspect ratio ...

            unitSize = (self.width() * self.height()) / n

            unitHeight = math.floor(math.sqrt(unitSize / unitRatio))
            self._rows = math.ceil(self.height() / unitHeight)
            self._unitHeight = math.floor(self.height() / self._rows)

            unitWidth = math.floor(unitRatio * self._unitHeight)
            self._columns = math.ceil(self.width() / unitWidth)
            self._unitWidth = math.floor(self.width() / self._columns)

    def paintEvent(self, event):
        '''Process the paint event by repainting this Widget.

        Parameters
        -----------
        event : QPaintEvent
        '''
        qp = QPainter()
        qp.begin(self)
        if self._controller._activation is None:
            self._drawNone(qp)
        elif self._isConvolution:
            self._drawConvolution(qp)
        else:
            self._drawDense(qp)
        if self._controller._selectedUnit is not None:
            self._drawSelection(qp)

        qp.end()

    def _getUnitRect(self, unit: int, padding: int=None):
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
            The rectangle occupied by the unie
        '''
        if padding is None:
            padding = self._padding
        return QRect(self._unitWidth * (unit % self._columns) + padding,
                     self._unitHeight * (unit // self._columns) + padding,
                     self._unitWidth - 2 * padding,
                     self._unitHeight - 2 * padding)

    def _unitAtPosition(self, position: QPoint):
        '''Compute the entry corresponding to some point in this widget.

        Parameters
        ----------
        position    :   QPoint
                        The position of the point in question (in Widget
                        coordinates).

        Returns
        -------
        int
            The unit occupying that position or ``None`` if no entry corresponds
            to that position.
        '''

        if self._controller._activation is None:
            return None
        unit = ((position.y() // self._unitHeight) * self._columns +
                (position.x() // self._unitWidth))
        if unit >= self._controller._activation.shape[0]:
            # selected something which may be on the grid, but where there's no
            # unit anymore
            unit = None
        return unit

    def _drawConvolution(self, qp):
        '''Draw activation values for a convolutional layer in the form of
        ``QImage``s.

        Parameters
        ----------
        qp : QPainter
        '''
        # image size: filter size (or a single pixel per neuron)
        map_height, map_width = self._controller._activation.shape[1:3]

        for unit in range(self._controller._activation.shape[0]):
            image = QImage(self._controller._activation[unit],
                           map_width, map_height,
                           map_width,
                           QImage.Format_Grayscale8)
            qp.drawImage(self._getUnitRect(unit), image)

    def _drawDense(self, qp):
        '''Draw activation values for a dense layer in the form of rectangles.

        Parameters
        ----------
        qp : QPainter
        '''
        for unit, value in enumerate(self._controller._activation):
            qp.fillRect(self._getUnitRect(unit),
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
        qp.drawRect(self._getUnitRect(self._controller._selectedUnit, 0))

    def _drawNone(self, qp):
        '''Draw a view when no activation values are available.

        Parameters
        ----------
        qp : QPainter
        '''
        qp.drawText(self.rect(), Qt.AlignCenter, 'No data!')

    def resizeEvent(self, event):
        '''Adapt to a change in size. The behavior dependes on the zoom policy.
        This event handler is called after the Widget has been resized.
        providing the new .size() and the old .oldSize().

        Parameters
        ----------
        event : QResizeEvent

        '''
        self._computeGeometry()
        self.update()

    def mousePressEvent(self, event):
        '''Process mouse event.

        Parameters
        ----------
        event : QMouseEvent
        '''
        unit = self._unitAtPosition(event.pos())
        self._controller.on_unit_selected(unit, self)

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

    def mouseDoubleClickEvent(self, event):
        '''Process a double click. We use double click to select a
        matrix entry.

        Parameters
        ----------
        event : QMouseEvent
        '''
        self.mousePressEvent(event)

    def keyPressEvent(self, event):
        '''Process special keys for this widget.  Allow moving selected entry
        using the cursor keys. Deselect unit using the Escape key.

        Parameters
        ----------
        event : QKeyEvent

        '''
        key = event.key()
        # Space will toggle display of tooltips
        if key == Qt.Key_Space:
            self.setToolTip(not self.toolTipActive)
        # Arrow keyes will move the selected entry
        elif self._selectedUnit is not None:
            row = self._selectedUnit % sender.columns
            col = self._selectedUnit // sender.columns
            if key == Qt.Key_Left:
                self._controller.on_unit_selected(self._selectedUnit - 1, self)
            elif key == Qt.Key_Up:
                self._controller.on_unit_selected(self._selectedUnit - self._columns, self)
            elif key == Qt.Key_Right:
                self._controller.on_unit_selected(self._selectedUnit + 1, self)
            elif key == Qt.Key_Down:
                self._controller.on_unit_selected(self._selectedUnit + self._columns, self)
            elif key == Qt.Key_Escape:
                self._controller.on_unit_selected(None, self)
            else:
                event.ignore()
        else:
            event.ignore()
