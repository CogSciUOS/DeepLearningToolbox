import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel

# FIXME[todo]: add docstrings!

class QImageView(QLabel):
    '''An experimental class to display images using the QPixmap
    class.  This may be more efficient than using matplotlib for
    displaying images.
    '''

    def __init__(self, parent):
        super().__init__(parent)
        self.setScaledContents(True)
        # an alternative may be to call
        #     pixmap.scaled(self.size(), Qt.KeepAspectRatio)
        # in the myplot method.



    def setMask(self, mask, position):
        self.mask = mask
        QImage.Format_ARGB32
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




