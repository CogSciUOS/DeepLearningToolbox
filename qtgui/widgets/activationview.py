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
            unitRatio = 1 # FIXME[todo]
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
        _activation = np.swapaxes(self.activation,0,3).copy()
        # FIXME[hack]: we copy here to get the activation in the
        # correct memory order, so that we can construct a QImage.
        # This may be inefficient - is there a better way?

        
        for i in range(_activation.shape[0]):
            #value = max(0,min(int(self.activation[0,i] * 255),255))
            row = i % self.columns
            col = i // self.columns
            rect = QRect(self.unitWidth*row + self.padding,
                         self.unitHeight*col + self.padding,
                         self.unitWidth-2*self.padding,
                         self.unitHeight-2*self.padding)
            image = QImage(_activation[i],
                           filter_width, filter_height,
                           QImage.Format_Grayscale8)
            qp.drawImage(rect, image)
            # FIXME[old]:
            #for i in range(ncolumns-(ncolumns*ncolumns-nbofplots)//ncolumns):
            #    ishow[i*(imagesize+1):(i+1)*(imagesize+1),0:(imagesize+1)*(nbofplots-i*ncolumns)]=np.hstack(np.lib.pad(_activation[i*ncolumns:(i+1)*ncolumns,:,:,0],[(0,0),(0,1),(0,1)],'constant', constant_values=2))

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
            #qp.setPen(color);
            #qp.setBrush(Qt.NoBrush);
            row = i % self.columns
            col = i // self.columns
            x = row * self.unitWidth
            y = col * self.unitHeight
            qp.fillRect(math.floor(x)+self.padding,
                        math.floor(y)+self.padding,
                        math.ceil(self.unitWidth-2*self.padding),
                        math.ceil(self.unitHeight-2*self.padding),
                        QBrush(QColor(value,value,value)))


    def _drawNone(self, qp):
        '''Draw a view when no activation values are available.
        
        Arguments
        ---------
        qp : QPainter
        '''
        qp.drawText(QRect(0, 0, self.width(), self.height()), Qt.AlignCenter, "No data!")
