import math
import numpy as np

from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QImage, QPen, QColor, QBrush
from PyQt5.QtWidgets import QWidget


# FIXME[todo]: repair convolution view!
# FIXME[todo]: add docstrings!


class QActivationView(QWidget):


    activation : object = None

    selected = pyqtSignal(int)
    
    padding : int = 2

    selectedUnit : int = None

    def __init__(self, parent = None):
        '''Initialization of the QMatrixView.

        Arguments
        ---------
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

    def setActivation(self, activation):

        self.activation = activation

        # unset selected entry
        self.selectedUnit = None
        
        if self.activation is not None:
            self.isConvolution = (len(self.activation.shape)>2)

            # normalization (values should be between 0 and 1)
            min_value = self.activation.min()
            max_value = self.activation.max()
            value_range = max_value - min_value
            self.activation = (self.activation - min_value)
            if value_range > 0:
                self.activation = self.activation/value_range

            # check the shape
            if self.isConvolution:
                # for convolution we want activtation to be of shape
                # (output_channels, width, height)
                if len(self.activation.shape) == 4:
                    # activation may include one axis for batches, i.e.,
                    # the axis of _activation are:
                    # (batch_size, width, height, output_channels)
                    # we do not need it - just take the first
                    # element from the batch
                    self.activation = self.activation[0].transpose([2,0,1])
                    #self.activation = np.swapaxes(self.activation,0,3)
            else:
                if len(self.activation.shape) == 2:
                    # activation may include one axis for batches, i.e.,
                    # we do not need it - just take the first
                    # element from the batch
                    self.activation = self.activation[0]

            # change dtype to uint8
            self.activation = np.ascontiguousarray(self.activation*255, np.uint8)

        self._computeGeometry()
        self.update()

    def selectUnit(self, unit = None):
        if self.selectedUnit != unit:
            self.selectedUnit = unit
            self.selected.emit(self.selectedUnit)
            self.update()

    def getUnitActivation(self, unit = None):
        if unit is None:
            unit = self.selectedUnit
        if self.activation is None or unit is None:
            return None
        return self.activation[self.selectedUnit]

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
            self.isConvolution = (len(self.activation.shape)>2)
            n = self.activation.shape[0]
            if self.isConvolution:
                # unitRatio = width/height
                unitRatio = self.activation.shape[1]/self.activation.shape[2]
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
        elif self.isConvolution:
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
        filter_width, filter_height = self.activation.shape[1:3]
        
        for unit in range(self.activation.shape[0]):
            image = QImage(self.activation[unit],
                           filter_width, filter_height,
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
        '''Process special keys for this widget.
        Allow moving selected entry using the cursor key.

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
                self.setSelection((row,col-1))
            elif key == Qt.Key_Up:
                self.setSelection((row-1,col))
            elif key == Qt.Key_Right:
                self.setSelection((row,col+1))
            elif key == Qt.Key_Down:
                self.setSelection((row+1,col))
            else:
                event.ignore()
        else:
            event.ignore()
