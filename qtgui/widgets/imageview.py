import numpy as np

from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
from PyQt5.QtWidgets import QLabel

# FIXME[todo]: add docstrings!

class QImageView(QLabel):
    '''An experimental class to display images using the QPixmap
    class.  This may be more efficient than using matplotlib for
    displaying images.
    '''

    activationMask : QImage = None

    def __init__(self, parent):
        super().__init__(parent)
        self.setScaledContents(True)
        # an alternative may be to call
        #     pixmap.scaled(self.size(), Qt.KeepAspectRatio)
        # in the myplot method.



    def setActivationMask(self, mask, position = None):
        if mask is None:
            self.activationMask = None
        else:
            print("debug: mask: {} ({})".format(mask.shape, mask.dtype))
            bitmaps = np.stack([mask,
                               np.ones(mask.shape,dtype=np.uint8)*255,
                               np.zeros(mask.shape,dtype=np.uint8),
                               np.zeros(mask.shape,dtype=np.uint8)])
            bitmaps = np.transpose(bitmaps,[1,2,0]).copy()
            #self.activationMask = QImage(bitmaps,
            #                             mask.shape[1], mask.shape[0],
            #                             QImage.Format_ARGB32)
            self.activationMask = QImage(mask,mask.shape[1], mask.shape[0],
                                         QImage.Format_Grayscale8)
        self.update()


    def myplot(self, image):
        '''Display the given image. Image is supposed to be a numpy array.
        '''

        print("myplot: {} ({}) {}-{}".format(image.shape,image.dtype,image.min(),image.max()))

        # To construct a 8-bit monochrome QImage, we need a uint8
        # numpy array
        if image.dtype != np.uint8:
            image = (image*255).astype(np.uint8)

        qtimage = QImage(image, image.shape[1], image.shape[0],
                         QImage.Format_Grayscale8)
        pixmap = QPixmap(qtimage)
        
        qp = QPainter(pixmap);
        pen_width = 2
        pen_color = Qt.red
        pen = QPen(pen_color)
        pen.setWidth(pen_width)
        qp.setPen(pen)
        qp.drawRect(3,3,26,26);
        if self.activationMask is not None:
            print("HALLO: {}".format(self.activationMask.size()))
            target = QRect(QPoint(),self.activationMask.size())
            source = QRect(QPoint(),self.activationMask.size())
            qp.drawImage(target, self.activationMask, source)
            
        qp.end()

        self.setPixmap(pixmap)



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




