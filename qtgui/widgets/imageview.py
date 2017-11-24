import numpy as np

from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QBrush
from PyQt5.QtWidgets import QWidget

# FIXME[todo]: add docstrings!


class QImageView(QWidget):
    '''An experimental class to display images using the ``QImage``
    class.  This may be more efficient than using matplotlib for
    displaying images.

    Attributes
    ----------
    _image  :   QImage
                The image to display
    _overlay    :   QImage
                    Overlay for displaying on top of the image
    '''

    _image: QImage = None

    activationMask: QImage = None

    def __init__(self, parent):
        super().__init__(parent)
        self._imageRect = None

    def setImage(self, image: np.ndarray):
        if image is not None:
            # To construct a 8-bit monochrome QImage, we need a
            # 2-dimensional, uint8 numpy array
            if image.ndim == 4:
                image = image[0]

            img_format = QImage.Format_Grayscale8
            bytes_per_line = image.shape[1]

            if image.ndim == 3:
                # three channels -> probably rgb
                if image.shape[2] == 3:
                    img_format = QImage.Format_RGB888
                    bytes_per_line *= 3
                else:
                    image = image[:, :, 0]

            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = np.copy(image)

            self._image = QImage(image,
                                 image.shape[1], image.shape[0],
                                 bytes_per_line, img_format)
        else:
            self._image = None

        self.update()

    def setActivationMask(self, mask, position=None):
        '''Set an (activation) mask to be displayed on top of
        the actual image. The mask should be

        Parameters
        ----------
        mask : numpy.ndarray

        '''
        if mask is None:
            self._overlay = None
        else:
            self._overlay = QImage(mask.shape[1], mask.shape[0],
                                   QImage.Format_ARGB32)
            self._overlay.fill(Qt.red)

            alpha = QImage(mask, mask.shape[1], mask.shape[0],
                           mask.shape[1], QImage.Format_Alpha8)
            painter = QPainter(self._overlay)
            painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            painter.drawImage(QPoint(), alpha)
            painter.end()
        self.update()

    def paintEvent(self, event):
        '''Process the paint event by repainting this Widget.

        Parameters
        ----------
        event : QPaintEvent
        '''
        painter = QPainter()
        painter.begin(self)
        self._drawImage(painter)
        self._drawMask(painter)
        painter.end()

    def _drawImage(self, painter: QPainter):
        if self._image is not None:
            w = self._image.width()
            h = self._image.height()
            if False:
                # Scale to full size
                painter.drawImage(self.rect(), self._image)

            elif True:
                # scale maximally while maintaining aspect ratio
                w_ratio = self.width() / w
                h_ratio = self.height() / h
                ratio = min(w_ratio, h_ratio)
                rect = QRect((self.width() - w * ratio) // 2,
                             (self.height() - h * ratio) // 2,
                             w * ratio, h * ratio)
                self._imageRect = rect
                painter.drawImage(rect, self._image)

            elif False:
                # Original image size
                x = (self.width() - self._image.width()) // 2
                y = (self.height() - self._image.height()) // 2
                painter.drawImage(x, y, self._image)

    def _drawMask(self, painter: QPainter):
        '''Display the given image. Image is supposed to be a numpy array.
        '''
        if self._image is not None and self._overlay is not None:
            # __import__('ipdb').set_trace()
            # scale_width = self.width() / self._image.width()
            # scale_height = self.height() / self._image.height()
            # delta = (self._image.size() - self._overlay.size())/2
            # target = QRect(delta.width()*scale_width,
            #                delta.height()*scale_height,
            #                self._overlay.width()*scale_width,
            #                self._overlay.height()*scale_height)

            #source = QRect(QPoint(),self._overlay.size())
            painter.drawImage(self._imageRect, self._overlay)
