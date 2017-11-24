import math
import numpy as np

from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QImage, QPen, QColor, QBrush
from PyQt5.QtWidgets import QWidget, QScrollBar, QHBoxLayout


# FIXME[todo]: implementation: up to now this is mainly a stub (a copy of QActivationView)

# FIXME[todo]: we may display positive and negative activation in a
# two-color scheme.

# FIXME[todo]: add docstrings!


class QConnectionView(QWidget):
    def __init__(self, parent=None):
        '''Initialization of the QMatrixView.

        Parameters
        ----------
        parent : QWidget
            The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)

        self._inputScrollBar = QScrollBar(Qt.Vertical)
        self._inputScrollBar.setFocusPolicy(Qt.StrongFocus)
        self._inputScrollBar.setMaximum(0)
        self._inputScrollBar.setValue(0)

        self._connections = QConnectionDisplay()

        self._outputScrollBar = QScrollBar(Qt.Vertical)
        self._outputScrollBar.setFocusPolicy(Qt.StrongFocus)
        self._outputScrollBar.setMaximum(0)
        self._outputScrollBar.setValue(0)

        layout = QHBoxLayout()
        layout.addWidget(self._inputScrollBar)
        layout.addWidget(self._connections)
        layout.addWidget(self._outputScrollBar)

        self.setLayout(layout)

        self._inputScrollBar.valueChanged.connect(
            self._connections.setInputOffset)

    def setActivation(self, input: np.ndarray, output: np.ndarray) -> None:
        self._connections.setActivation(input, output)
        self._inputScrollBar.setMaximum(max(self._connections.getInputHeight()
                                            - self._connections.height(), 0))
        self._inputScrollBar.setValue(self._connections.getInputOffset())
        self._inputScrollBar.setPageStep(self._connections.height())


class QConnectionDisplay(QWidget):

    connections: np.ndarray = None

    _input: np.ndarray = None

    _output: np.ndarray = None

    selectedInput: int = None

    selectedOutput: int = None

    _inputOffset: int = 0
    _outputOffset: int = 0

    _inputOrder: list

    _outputOrder: list

    padding: int = 2
    """Padding between the individual units in a layer."""

    selected = pyqtSignal(object)
    """A signal emitted whenever a unit is (de)selected in this
    QConnectionDisplay. This will be an int (the index of the selected
    unit) or None (if no unit is selected). [We have to use object not
    int here to allow for None values.]
    """

    def __init__(self, parent=None):
        '''Initialization of the QMatrixView.

        Parameters
        ----------
        matrix : numpy.ndarray
            The matrix to be displayed in this QMatrixView.
        parent : QWidget
            The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)

        self.selectedUnit = None

        # By default, a QWidget does not accept the keyboard focus, so
        # we need to enable it explicitly: Qt.StrongFocus means to
        # get focus by "Tab" key as well as by mouse click.
        self.setFocusPolicy(Qt.StrongFocus)

    def setActivation(self, input: np.ndarray, output: np.ndarray) -> None:
        """Set the activations to be displayed in this QConnectionDisplay.
        Currently there are two possible types of activations that are
        supported by this widget: 1D, and 2D convolutional.

        Parameters
        ----------
        activation:
            Either a 1D or a 3D array. The latter one will be displayed
            in the convolutional mode.
        """

        activation = input
        old_shape = None if activation is None else activation.shape
        if activation is not None:
            self.isConvolution = (len(activation.shape) > 2)

            # normalization (values should be between 0 and 1)
            min_value = activation.min()
            max_value = activation.max()
            value_range = max_value - min_value
            activation = (activation - min_value)
            if value_range > 0:
                activation = activation / value_range

            # check the shape
            if self.isConvolution:
                # for convolution we want activtation to be of shape
                # (output_channels, width, height)
                if len(activation.shape) == 4:
                    # activation may include one axis for batches, i.e.,
                    # the axis of _activation are:
                    # (batch_size, width, height, output_channels)
                    # we do not need it - just take the first
                    # element from the batch
                    activation = activation.squeeze(axis=0)
                # (width, height, output_channels)
                #  to (output_channels, width, height)
                activation = activation.transpose([2, 0, 1])
                #self.activation = np.swapaxes(self.activation,0,3)
            else:
                if len(activation.shape) == 2:
                    # activation may include one axis for batches, i.e.,
                    # we do not need it - just take the first
                    # element from the batch
                    activation = activation[0]

            # change dtype to uint8
            # FIXME[hack]:
            self._input = np.ascontiguousarray(activation * 255, np.uint8)
            self._output = np.ascontiguousarray(activation * 255, np.uint8)

        # unset selected entry if shape changed
        if self._input is None or old_shape != self._input.shape:
            self.selectUnit()
        else:
            self.selected.emit(self.selectedUnit)

        self._computeGeometry()
        self.update()

    def getInputHeight(self):
        if self._inputUnitHeight is None:
            return 0
        return self._inputUnitHeight * self._input.shape[0]

    def getInputOffset(self):
        return 0 if self._inputOffset is None else self._inputOffset

    def setInputOffset(self, offset):
        if self._inputOffset != offset:
            self._inputOffset = offset
            self.update()

    def selectUnit(self, unit=None, input=True):
        if self._input is None:
            unit = None
        elif unit is not None and (unit < 0 or unit >= self._input.shape[0]):
            unit = None
        if self.selectedUnit != unit:
            self.selectedUnit = unit
            self.selected.emit(self.selectedUnit)
            self.update()

    def getUnitActivation(self, unit=None, input=True) -> np.ndarray:
        """Get the activation mask for a given unit.
        """
        if unit is None:
            unit = self.selectedUnit
        if self._input is None or unit is None or not self.isConvolution:
            return None
        return self._input[unit]

    def _computeGeometry(self):
        if self._input is None:
            self._inputUnitWidth = None
            self._inputUnitHeight = None
            self._outputUnitWidth = None
            self._outputUnitHeight = None
        else:
            # In case of a convolutional layer, the axes of activation are:
            # (batch_size, width, height, output_channels)
            # For fully connected (i.e., dense) layers, the axes are:
            # (batch_size, units)
            # In both cases, batch_size should be 1!
            self.isConvolution = (len(self._input.shape) > 2)
            n = self._input.shape[0]
            if self.isConvolution:
                # unitRatio = width/height
                unitRatio = self._input.shape[1] / self._input.shape[2]
            else:
                unitRatio = 1

            # FIXME: implement better computation!
            # - allow for rectangular (i.e. non-quadratic) widget size
            # - allow for rectangular (i.e. non-quadratic) convolution filters
            # and maintain aspect ratio ...

            # unitRatio = w/h
            # unitSize = w*h
            unitSize = (self.width() * self.height()) / n

            unitHeight = math.floor(math.sqrt(unitSize / unitRatio))
            rows = math.ceil(self.height() / unitHeight)
            unitHeight = math.floor(self.height() / rows)

            unitWidth = math.floor(unitRatio * unitHeight)
            columns = math.ceil(self.width() / unitWidth)
            unitWidth = math.floor(self.width() / columns)

            self._inputUnitWidth = unitWidth
            self._inputUnitHeight = unitHeight
            self._outputUnitWidth = unitWidth
            self._outputUnitHeight = unitHeight

        self.update()

    def paintEvent(self, event):
        '''Process the paint event by repainting this Widget.

        Parameters
        ----------
        event : QPaintEvent
        '''
        qp = QPainter()
        qp.begin(self)
        if self._input is None:
            self._drawNone(qp)
        elif self.isConvolution:
            self._drawConvolution(qp)
        else:
            self._drawDense(qp)
        if self.selectedUnit is not None:
            self._drawSelection(qp)

        qp.end()

    def _getUnitRect(self, input: bool, unit: int, padding: int = None):
        '''Get the rectangle (screen position and size) occupied by the given
        unit.


        Parameters
        ----------
        unit : index of the unit of interest
        left
            A flag indicating if the unit indicates an input (True,
            left side) or an outupt (False, right side).
        padding: padding of the unit.
            If None is given, standard padding value of this QActivationView
            will be use.

        '''
        if padding is None:
            padding = self.padding
        if input:
            rect = QRect(padding,
                         self._inputUnitHeight * unit
                         + padding - self._inputOffset,
                         self._inputUnitWidth - 2 * padding,
                         self._inputUnitHeight - 2 * padding)
        else:
            rect = QRect(self.width() - self._outputUnitWidth + padding,
                         self._outputUnitHeight * unit + padding,
                         self._outputUnitWidth - 2 * padding,
                         self._outputUnitHeight - 2 * padding)
        return rect

    def _unitAtPosition(self, position: QPoint):
        '''Compute the entry corresponding to some point in this widget.

        Parameters
        ----------
        position
            The position of the point in question (in Widget coordinates).

        Returns
        -------
        The unit occupying that position of None
        if no entry corresponds to that position.
        '''

        if self._input is None:
            return None
        unit = None

        # FIXME[todo]
        # unit = ((position.y() // self.unitHeight) * self.columns +
        #        (position.x() // self.unitWidth))
        # if unit >= self.activation.shape[0]:
        #    unit = None
        return unit

    def _drawConvolution(self, qp):
        '''Draw activation values for a convolutional layer.

        Parameters
        ----------
        qp : QPainter
        '''

        r2 = self._getUnitRect(False, 0)
        p2 = r2.center()
        p2.setX(r2.left())
        pen_width = 4
        pen_color = Qt.red
        pen = QPen(pen_color)
        pen.setWidth(pen_width)

        map_width, map_height = self._input.shape[1:3]
        for unit in range(self._input.shape[0]):
            rect = self._getUnitRect(True, unit)
            if ((rect.top() + self._inputUnitHeight >= 0) and
                    (rect.top() < self.height())):
                image = QImage(self._input[unit],
                               map_width, map_height,
                               map_width,
                               QImage.Format_Grayscale8)
                qp.drawImage(rect, image)

                p1 = rect.center()
                p1.setX(rect.right())

                qp.setPen(pen)
                qp.drawLine(p1, p2)

        map_width, map_height = self._output.shape[1:3]
        for unit in range(self._output.shape[0]):
            image = QImage(self._output[unit],
                           map_width, map_height,
                           map_width,
                           QImage.Format_Grayscale8)
            qp.drawImage(self._getUnitRect(False, unit), image)

    def _drawDense(self, qp):
        '''Draw activation values for a dense layer.

        Parameters
        ----------
        qp : QPainter
        '''
        for unit, value in enumerate(self._input):
            qp.fillRect(self._getUnitRect(True, unit),
                        QBrush(QColor(value, value, value)))

        for unit, value in enumerate(self._output):
            qp.fillRect(self._getUnitRect(False, unit),
                        QBrush(QColor(value, value, value)))

    def _drawSelection(self, qp):

        pen_width = 4
        pen_color = Qt.red
        pen = QPen(pen_color)
        pen.setWidth(pen_width)
        qp.setPen(pen)
        qp.drawRect(self._getUnitRect(self.selectedUnit, 0))

    def _drawNone(self, qp):
        '''Draw a view when no activation values are available.

        Parameters
        ----------
        qp : QPainter
        '''
        qp.drawText(self.rect(), Qt.AlignCenter, "No data!")

    def resizeEvent(self, event):
        '''Adapt to a change in size. The behavior dependes on the zoom
        policy.

        Parameters
        ----------
        event : QResizeEvent

        '''
        # This event handler is called after the Widget has been resized.
        # providing the new .size() and the old .oldSize().
        self._computeGeometry()

    def mousePressEvent(self, event):
        '''Process mouse event.

        Parameters
        ----------
        event : QMouseEvent
        '''
        self.selectUnit(self._unitAtPosition(event.pos()))

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
        self.selectUnit(self._unitAtPosition(event.pos()))

    def keyPressEvent(self, event):
        '''Process special keys for this widget.
        Allow moving selected entry using the cursor key.

        Parameters
        ----------
        event : QKeyEvent
        '''
        key = event.key()
        # Space will toggle display of tooltips
        if key == Qt.Key_Space:
            self.setToolTip(not self.toolTipActive)
        # Arrow keyes will move the selected entry
        elif self.selectedUnit is not None:
            if key == Qt.Key_Up:
                # FIXME[todo]: input/output
                self.selectUnit(self.selectedUnit - 1)
            elif key == Qt.Key_Down:
                # FIXME[todo]: input/output
                self.selectUnit(self.selectedUnit + 1)
            elif key == Qt.Key_Escape:
                self.selectUnit(None)
            else:
                event.ignore()
        else:
            event.ignore()
