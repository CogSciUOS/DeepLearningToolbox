import math
import numpy as np

from PyQt5.QtCore import Qt, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QImage, QPen, QColor, QBrush
from PyQt5.QtWidgets import QWidget


# FIXME[todo]: repair convolution view!
# FIXME[todo]: add docstrings!


class QActivationView(QWidget):

    activation : object = None

    selected = pyqtSignal(int)

    def setActivation(self, activation):

        self.activation = activation
        if self.activation is not None:
            self.isConvolution = (len(self.activation.shape)>2)

    def computerGeometry(self):
        if self.activation is not None:
            # In case of a convolutional layer, the axes of activation are:
            # (batch_size, width, height, output_channels)
            # For fully connected (i.e., dense) layers, the axes are:
            # (batch_size, units)
            # In both cases, batch_size should be 1!
            self.isConvolution = (len(self.activation.shape)>2)
            if self.isConvolution:
                n = self.activation.shape[3]
                # unitRatio = width/height
                unitRatio = self.activation.shape[1]/self.activation.shape[2]
            else:
                n = self.activation.shape[1]
                unitRatio = 1

            # FIXME: implement better computation!
            # - allow for rectangular (i.e. non-quadratic) widget size
            # - allow for rectangular (i.e. non-quadratic) convolution filters
            # and maintain aspect ratio ...

            self.padding = 2

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
        self.computerGeometry()
        qp = QPainter()
        qp.begin(self)
        if self.activation is None:
            self._drawNone(qp)
        elif self.isConvolution:
            self._drawConvolution(qp)
        else:
            self._drawDense(qp)
        qp.end()

    def getUnitRect(self, unit, padding = 0):
        return QRect(self.unitWidth * (unit % self.columns) + padding,
                     self.unitHeight * (unit // self.columns) + padding,
                     self.unitWidth - 2*padding,
                     self.unitHeight - 2*padding)

    def _drawConvolution(self, qp):
        '''Draw activation values for a convolutional layer.
        
        Arguments
        ---------
        qp : QPainter
        '''


        # vm = np.max(intermediate_output) if fully_connected else None

        # number of plots: plot all output channels
        number_of_plots = self.activation.shape[3]

        # image size: filter size (or a single pixel per neuron)
        filter_width, filter_height = self.activation.shape[1:3]

        # number of columns and rows (arrange as square)
        #ncolumns = nraws = math.ceil(np.sqrt(nbofplots))

        # FIXME[old]
        # the pixel map to be shown
        #output_array = np.ones([(self.unitHeight+1) * self.rows,
        #                        (self.unitWidth+1) * self.columns])*2

        # the axis of _activation are:
        # (output_channels, width, height, batch_size)
        
        print("shape={}, dtype={}, min={}, max={}".format(self.activation.shape,self.activation.dtype,self.activation.min(),self.activation.max()))
        _activation = np.swapaxes(self.activation*255,0,3).astype(np.uint8).copy()
        # FIXME[hack]: we copy here to get the activation in the
        # correct memory order, so that we can construct a QImage.
        # This may be inefficient - is there a better way?
        
        for unit in range(_activation.shape[0]):
            image = QImage(_activation[unit],
                           filter_width, filter_height,
                           QImage.Format_Grayscale8)
            qp.drawImage(self.getUnitRect(unit, self.padding), image)

        self.selectedUnit = 2
        if self.selectedUnit is not None:
            pen_width = 4
            pen_color = Qt.red
            pen = QPen(pen_color)
            pen.setWidth(pen_width)
            qp.setPen(pen)
            qp.drawRect(self.getUnitRect(self.selectedUnit))
        

        qp.fillRect(QRect(0, self.height()//2-20, self.width(), 40), QBrush(QColor(Qt.yellow)))
        qp.setPen(Qt.red);
        qp.drawText(QRect(0, 0, self.width(), self.height()), Qt.AlignCenter, "FIXME[bug]: convolution currently does not work!")


    def _drawDense(self, qp):
        '''Draw activation values for a dense layer.
        
        Arguments
        ---------
        qp : QPainter
        '''
        for i in range(self.activation.shape[1]):
            value = max(0,min(int(self.activation[0,i] * 255),255))
            qp.fillRect(self.getUnitRect(unit, self.padding),
                        QBrush(QColor(value,value,value)))


    def _drawNone(self, qp):
        '''Draw a view when no activation values are available.
        
        Arguments
        ---------
        qp : QPainter
        '''
        qp.drawText(self.rect(), Qt.AlignCenter, "No data!")



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
