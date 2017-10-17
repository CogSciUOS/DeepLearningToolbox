import numpy as np

from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
from PyQt5.QtWidgets import QWidget

# FIXME[todo]: add docstrings!

class QImageView(QWidget):
    '''An experimental class to display images using the QPixmap
    class.  This may be more efficient than using matplotlib for
    displaying images.
    '''

    image : QImage = None

    activationMask : QImage = None

    def __init__(self, parent):
        super().__init__(parent)
        #self.setScaledContents(True)
        # an alternative may be to call
        #     pixmap.scaled(self.size(), Qt.KeepAspectRatio)
        # in the myplot method.

    def setImage(self, image : np.ndarray):
        self.image = image
        if image is not None:
            # To construct a 8-bit monochrome QImage, we need a
            # 2-dimensional, uint8 numpy array
            if image.ndim == 4:
                image = image[0]

            format = QImage.Format_Grayscale8
            byter_per_line = image.shape[1]
            if image.ndim == 3:
                if image.shape[2] == 3:
                    format = QImage.Format_RGB888
                    byter_per_line *= 3
                else:
                    image = image[:,:,0]

            if image.dtype != np.uint8:
                image = (image*255).astype(np.uint8)
            image = np.copy(image)
            
            self.image = QImage(image,
                                image.shape[1], image.shape[0],
                                byter_per_line, format)
        else:
            self.image = None

        self.update()


    def setActivationMask(self, mask, position = None):
        '''Set an (activation) mask to be displayed on top of
        the actual image. The mask should be

        Arguments
        ---------
        mask : numpy.ndarray

        '''
        if mask is None:
            self.activationMask = None
        else:
            self.activationMask = QImage(mask.shape[1], mask.shape[0],
                                         QImage.Format_ARGB32)
            self.activationMask.fill(Qt.red)

            alpha = QImage(mask, mask.shape[1], mask.shape[0],
                           mask.shape[1], QImage.Format_Alpha8)
            painter = QPainter(self.activationMask)
            painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            painter.drawImage(QPoint(),alpha)
            painter.end()
        self.update()


    def paintEvent(self, event):
        '''Process the paint event by repainting this Widget.

        Arguments
        ---------
        event : QPaintEvent
        '''
        painter = QPainter()
        painter.begin(self)
        self._drawImage(painter)
        self._drawMask(painter)
        painter.end()


    def _drawImage(self, painter : QPainter):
        if self.image is not None:
            w = self.image.width()
            h = self.image.height()
            if False:
                # Scale to full size
                painter.drawImage(self.rect(),self.image)

            elif True:
                # scale maximally while maintaining aspect ratio
                w_ratio = self.width()/w
                h_ratio = self.height()/h
                ratio = min(w_ratio, h_ratio)
                rect = QRect((self.width()-w*ratio)//2,
                             (self.height()-h*ratio)//2,
                             w*ratio,h*ratio)
                painter.drawImage(rect, self.image)

            elif False:
                # Original image size
                x = (self.width()-self.image.width()) // 2
                y = (self.height()-self.image.height()) // 2
                painter.drawImage(x,y,self.image)


    def _drawMask(self, painter : QPainter):
        '''Display the given image. Image is supposed to be a numpy array.
        '''
        if self.image is not None and self.activationMask is not None:
            scale_width = self.width() / self.image.width()
            scale_height = self.height() / self.image.height()
            delta = (self.image.size() - self.activationMask.size())/2
            target = QRect(delta.width()*scale_width,
                           delta.height()*scale_height,
                           self.activationMask.width()*scale_width,
                           self.activationMask.height()*scale_height)

            #source = QRect(QPoint(),self.activationMask.size())
            painter.drawImage(target, self.activationMask)



###
### FIXME: OLD
###


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# FIXME[todo]: add docstrings!

class QImageViewMatplotlib(FigureCanvas):
    '''A simple class to display an image, using a MatPlotLib figure.
    '''

    def __init__(self, parent=None, width=9, height=9, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.axis('off')

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.draw()


    def myplot(self, image, map = 'gray'):
        '''Plot the given image.
        '''
        self.axes.imshow(image, cmap = map)
        self.draw()
