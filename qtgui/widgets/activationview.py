import math
import numpy as np

from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QImage, QPen, QColor, QBrush
from PyQt5.QtWidgets import QWidget

# FIXME[todo]: improve the display of the activations: check that the
# space is used not only in a good, but in an optimal way. Check that
# the aspect ratio is correct. Make it configurable to allow for
# explicitly setting different aspects of display.

# FIXME[todo]: we may display positive and negative activation in a
# two-color scheme.

class QActivationView(QWidget):
    """A widget to diplay the activations of a given layer in a
    network. Currently there are two types of layers that are
    supported: (two-dimensional) convolutional layers and dense
    (=fully connected) layers.

    The QActivationView widget allows to select an individual unit in
    the network layer by a single mouse click (this will either select
    a single unit in a dense layer, or a channel in a convolutional
    layer). The selection can be moved with the cursor keys and the
    unit can be deselected by hitting escape. The widget will signal
    such a (de)selection by emitting the "selected" signal.

    The QActivationView will try to make good use of the available
    space by arranging and scaling the units. However, the current
    implementation is still suboptimal and may be improved to allow
    for further configuration.
    """

    activation : np.ndarray = None
    """The activation values to be displayed in this activation view. None
    means that no activation is assigned to this QActivationView and
    will result in an empty widget.
    """

    padding : int = 2
    """Padding between the individual units in this QActivationView.
    """

    selectedUnit : int = None
    """The currently selected unit. The value None means that no unit is
    currently selected.
    """

    _isConvolution : bool = False
    """A flag indicating if the current QActivationView is currently in
    convolution mode (True) or not (False).
    """

    selected = pyqtSignal(object)
    """A signal emitted whenever a unit is (de)selected in this
    QActivationView. This will be an int (the index of the selected
    unit) or None (if no unit is selected). We have to use object not
    int here to allow for None values.
    """


    def __init__(self, parent : QWidget = None):
        '''Initialization of the QMatrixView.

        Arguments
        ---------
        parent
            The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)

        self.selectedUnit = None

        # By default, a QWidget does not accept the keyboard focus, so
        # we need to enable it explicitly: Qt.StrongFocus means to
        # get focus by "Tab" key as well as by mouse click.
        self.setFocusPolicy(Qt.StrongFocus)


    def setActivation(self, activation : np.ndarray) -> None:
        """Set the activations to be displayed in this QActivationView.
        Currently there are two possible types of activations that are
        supported by this widget: 1D, and 2D convolutional.

        Arguments
        ---------
        activation:
            Either a 1D or a 3D array. The latter one will be
            displayed in the convolutional mode. The activation values
            are expected to be float values. For display they
            will be scaled and converted to 8-bit integers.
        """

        old_shape = None if self.activation is None else self.activation.shape
        self.activation = activation

        if self.activation is not None:
            self._isConvolution = (len(self.activation.shape)>2)

            # normalization (values should be between 0 and 1)
            min_value = self.activation.min()
            max_value = self.activation.max()
            value_range = max_value - min_value
            self.activation = (self.activation - min_value)
            if value_range > 0:
                self.activation = self.activation/value_range

            # check the shape
            if self._isConvolution:
                # for convolution we want activtation to be of shape
                # (output_channels, width, height)
                if len(self.activation.shape) == 4:
                    # activation may include one axis for batches, i.e.,
                    # the axis of _activation are:
                    # (batch_size, width, height, output_channels)
                    # we do not need it - just take the first
                    # element from the batch
                    self.activation = self.activation.squeeze(axis=0)
                # (width, height, output_channels)
                #  to (output_channels, width, height)
                self.activation = self.activation.transpose([2,0,1])
                #self.activation = np.swapaxes(self.activation,0,3)
            else:
                if len(self.activation.shape) == 2:
                    # activation may include one axis for batches, i.e.,
                    # we do not need it - just take the first
                    # element from the batch
                    self.activation = self.activation[0]

            # change dtype to uint8
            self.activation = np.ascontiguousarray(self.activation*255, np.uint8)

        ## unset selected entry if shape changed
        if self.activation is None or old_shape != self.activation.shape:
            self.selectUnit()
        else:
            self.selected.emit(self.selectedUnit)

        self._computeGeometry()
        self.update()


    def selectUnit(self, unit : int = None):
        """(De)select a unit in this QActivationView.

        Arguments
        =========
        unit:

        """
        if self.activation is None:
            unit = None
        elif unit is not None and (unit < 0 or unit >= self.activation.shape[0]):
            unit = None
        if self.selectedUnit != unit:
            self.selectedUnit = unit
            self.selected.emit(self.selectedUnit)
            self.update()


    def getUnitActivation(self, unit = None) -> np.ndarray:
        """Get the activation mask for a given unit.
        """
        if unit is None:
            unit = self.selectedUnit
        if self.activation is None or unit is None or not self._isConvolution:
            return None
        return self.activation[unit]


    def _computeGeometry(self):
        if self.activation is None:
            self.rows = None
            self.columns = None
            self.unitWidth = None
            self.unitHeight = None
        else:
            # In case of a convolutional layer, the axes of activation are:
            # (batch_size, width, height, output_channels)
            # For fully connected (i.e., dense) layers, the axes are:
            # (batch_size, units)
            # In both cases, batch_size should be 1!
            self._isConvolution = (len(self.activation.shape)>2)
            n = self.activation.shape[0]
            if self._isConvolution:
                # unitRatio = width/height
                unitRatio = self.activation.shape[2]/self.activation.shape[1]
                print(unitRatio)
            else:
                unitRatio = 1

            # FIXME: implement better computation!
            # - allow for rectangular (i.e. non-quadratic) widget size
            # - allow for rectangular (i.e. non-quadratic) convolution filters
            # and maintain aspect ratio ...

            # unitRatio = w/h
            # unitSize = w*h
            unitSize = (self.width() * self.height()) / n

            unitHeight = math.floor(math.sqrt(unitSize/unitRatio))
            self.rows = math.ceil(self.height()/unitHeight)
            self.unitHeight = math.floor(self.height()/self.rows)

            unitWidth = math.floor(unitRatio * self.unitHeight)
            self.columns = math.ceil(self.width()/unitWidth)
            self.unitWidth = math.floor(self.width()/self.columns)

        self.update()



    def paintEvent(self, event):
        '''Process the paint event by repainting this Widget.

        Arguments
        ---------
        event : QPaintEvent
        '''
        qp = QPainter()
        qp.begin(self)
        if self.activation is None:
            self._drawNone(qp)
        elif self._isConvolution:
            self._drawConvolution(qp)
        else:
            self._drawDense(qp)
        if self.selectedUnit is not None:
            self._drawSelection(qp)

        qp.end()


    def _getUnitRect(self, unit : int, padding : int = None):
        '''Get the rectangle (screen position and size) occupied by the given
        unit.


        Arguments
        ---------
        unit : index of the unit of interest
        padding: padding of the unit.
            If None is given, standard padding value of this QActivationView
            will be use.
        '''
        if padding is None:
            padding = self.padding
        return QRect(self.unitWidth * (unit % self.columns) + padding,
                     self.unitHeight * (unit // self.columns) + padding,
                     self.unitWidth - 2*padding,
                     self.unitHeight - 2*padding)


    def _unitAtPosition(self, position : QPoint):
        '''Compute the entry corresponding to some point in this widget.

        Arguments
        ---------
        position
            The position of the point in question (in Widget coordinates).

        Returns
        -------
        The unit occupying that position of None
        if no entry corresponds to that position.
        '''

        if self.activation is None:
            return None
        unit = ((position.y() // self.unitHeight) * self.columns +
                (position.x() // self.unitWidth))
        if unit >= self.activation.shape[0]:
            unit = None
        return unit


    def _drawConvolution(self, qp):
        '''Draw activation values for a convolutional layer.

        Arguments
        ---------
        qp : QPainter
        '''

        # image size: filter size (or a single pixel per neuron)
        map_height, map_width = self.activation.shape[1:3]

        for unit in range(self.activation.shape[0]):
            print(map_width)
            image = QImage(self.activation[unit],
                           map_width, map_height,
                           map_width,
                           QImage.Format_Grayscale8)
            qp.drawImage(self._getUnitRect(unit), image)



    def _drawDense(self, qp):
        '''Draw activation values for a dense layer.

        Arguments
        ---------
        qp : QPainter
        '''
        for unit, value in enumerate(self.activation):
            qp.fillRect(self._getUnitRect(unit),
                        QBrush(QColor(value,value,value)))


    def _drawSelection(self, qp):
        '''Mark the currently selected unit in the painter.

        Arguments
        ---------
        qp : QPainter
        '''
        pen_width = 4
        pen_color = Qt.red
        pen = QPen(pen_color)
        pen.setWidth(pen_width)
        qp.setPen(pen)
        qp.drawRect(self._getUnitRect(self.selectedUnit,0))


    def _drawNone(self, qp):
        '''Draw a view when no activation values are available.

        Arguments
        ---------
        qp : QPainter
        '''
        qp.drawText(self.rect(), Qt.AlignCenter, "No data!")


    def resizeEvent(self, event):
        '''Adapt to a change in size. The behavior dependes on the zoom
        policy.

        Arguments
        ---------
        event : QResizeEvent

        '''
        # This event handler is called after the Widget has been resized.
        # providing the new .size() and the old .oldSize().
        self._computeGeometry()


    def mousePressEvent(self, event):
        '''Process mouse event.

        Arguments
        ---------
        event : QMouseEvent
        '''
        self.selectUnit(self._unitAtPosition(event.pos()))


    def mouseReleaseEvent(self, event):
        '''Process mouse event.
        Arguments
        ---------
        event : QMouseEvent
        '''
        # As we implement .mouseDoubleClickEvent(), we
        # also provide stubs for the other mouse events to not confuse
        # other widgets.
        pass

    def mouseDoubleClickEvent(self, event):
        '''Process a double click. We use double click to select a
        matrix entry.

        Arguments
        ---------
        event : QMouseEvent
        '''
        self.selectUnit(self._unitAtPosition(event.pos()))


    def keyPressEvent(self, event):
        '''Process special keys for this widget.  Allow moving selected entry
        using the cursor keys. Deselect unit using the Escape key.

        Arguments
        ---------
        event : QKeyEvent

        '''
        key = event.key()
        # Space will toggle display of tooltips
        if key == Qt.Key_Space:
            self.setToolTip(not self.toolTipActive)
        # Arrow keyes will move the selected entry
        elif self.selectedUnit is not None:
            row = self.selectedUnit % self.columns
            col = self.selectedUnit // self.columns
            if key == Qt.Key_Left:
                self.selectUnit(self.selectedUnit-1)
            elif key == Qt.Key_Up:
                self.selectUnit(self.selectedUnit-self.columns)
            elif key == Qt.Key_Right:
                self.selectUnit(self.selectedUnit+1)
            elif key == Qt.Key_Down:
                self.selectUnit(self.selectedUnit+self.columns)
            elif key == Qt.Key_Escape:
                self.selectUnit(None)
            else:
                event.ignore()
        else:
            event.ignore()
