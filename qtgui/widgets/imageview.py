import numpy as np
from scipy.misc import imresize

from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QBrush
from PyQt5.QtWidgets import QWidget

from observer import Observer

# FIXME[todo]: add docstrings!


class QImageView(QWidget, Observer):
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

    _overlay: QImage = None

    def __init__(self, parent):
        super().__init__(parent)
        self._imageRect = None

    def setImage(self, image: np.ndarray):
        if image is not None:
            # To construct an 8-bit monochrome QImage, we need a
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

    def setActivationMask(self, mask):
        '''Set an (activation) mask to be displayed on top of
        the actual image.

        Parameters
        ----------
        mask : numpy.ndarray

        '''
        if mask is None:
            self._overlay = None
        else:
            mask = imresize(mask, (self._image.height(), self._image.width()),
                                    interp='nearest')
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
        '''Draw current image into this ``QImageView``.

        Parameters
        ----------
        painter :   QPainter
        '''
        if self._image is not None:
            w = self._image.width()
            h = self._image.height()
            # scale maximally while maintaining aspect ratio
            w_ratio = self.width() / w
            h_ratio = self.height() / h
            ratio = min(w_ratio, h_ratio)
            # the rect is created such that it is centered on the current widget
            # pane both horizontally and vertically
            self._imageRect = QRect((self.width() - w * ratio) // 2,
                                    (self.height() - h * ratio) // 2,
                                    w * ratio, h * ratio)
            painter.drawImage(self._imageRect, self._image)

    def _drawMask(self, painter: QPainter):
        '''Display the given image.

        Parameters
        ----------
        painter :   QPainter
        '''
        if self._image is not None and self._overlay is not None:
            painter.drawImage(self._imageRect, self._overlay)

    def modelChanged(self, model, info):
        from util import grayscaleNormalized
        all_activations = model._current_activation
        unit = model._unit
        # skip if dense layer
        if info.input_index_changed or info.dataset_changed:
            current_input = model.getInput(model._current_index)
            self.setImage(current_input.data)
        if all_activations is not None and unit is not None and all_activations.ndim > 1:
            activation_mask_f = all_activations[..., unit]
            # PROBLEM: Mask not properly updated
            activation_mask = np.ascontiguousarray(grayscaleNormalized(activation_mask_f), np.uint8)
            self.setActivationMask(activation_mask)
        else:
            self.setActivationMask(None)
