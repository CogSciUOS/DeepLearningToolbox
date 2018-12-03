from PyQt5.QtCore import QObject, pyqtSignal

from controller import AsyncRunner
from observer import Observable, BaseChange

class QtAsyncRunner(AsyncRunner, QObject):
    """:py:class:`AsyncRunner` subclass which knows how to update Qt widgets.

    Attributes
    ----------
    _completion_signal: pyqtSignal
        Signal emitted once the computation is done.
        The signal is connected to :py:meth:`onCompletion` which will be run
        on the main thread by the Qt magic.
    """

    # signals must be declared outside the constructor, for some weird reason
    _completion_signal = pyqtSignal(Observable, object)

    def __init__(self):
        """Connect a signal to :py:meth:`Observable.notifyObservers`."""
        super().__init__()
        self._completion_signal.connect(self._notifyObservers)

    def onCompletion(self, future):
        """Emit the completion signal to have
        :py:meth:`Observable.notifyObservers` called.

        This method is still executed in the runner Thread. It will
        emit a pyqtSignal that is received in the main Thread and
        therefore can notify the Qt GUI.
        """
        observable, info = future.result()
        import threading
        me = threading.current_thread().name
        print(f"[{me}]{self.__class__.__name__}.onCompletion():{info}")
        if isinstance(info, BaseChange):
            self._completion_signal.emit(observable, info)

    def _notifyObservers(self, observable, info):
        """The method is intended as a receiver of th pyqtSignal.
        It will be run in the main Thread and hence can notify
        the Qt GUI.
        """
        observable.notifyObservers(info)

import numpy as np
from PyQt5.QtCore import Qt, QPoint, QSize, QRect
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtWidgets import QWidget


class QImageView(QWidget):
    """An experimental class to display images using the ``QImage``
    class.  This may be more efficient than using matplotlib for
    displaying images.

    Attributes
    ----------
    _image: QImage
        The image to display
    _overlay: QImage
        Overlay for displaying on top of the image
    _show_raw: bool
        A flag indicating whether this QImageView will show
        the raw input data, or the data acutally fed to the network.
    _imageRect: 
    """
    
    def __init__(self, parent: QWidget=None):
        super().__init__(parent)

        self._raw: np.ndarray = None
        self._image: QImage = None
        self._overlay: QImage = None
        self._imageRect = None

    def getImage(self) -> np.ndarray:
        return self._raw

    def setImage(self, image: np.ndarray) -> None:
        """Set the image to display.
        """
        self._raw = image
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
            image = image.copy()

            self._image = QImage(image,
                                 image.shape[1], image.shape[0],
                                 bytes_per_line, img_format)
            self.resize(self._image.size())
        else:
            self._image = None

        self.updateGeometry()
        self.update()

    def minimumSizeHint(self):
        return QSize(-1,-1) if self._image is None else self._image.size()

    def setMask(self, mask):
        """Set a mask to be displayed on top of the actual image.

        Parameters
        ----------
        mask : numpy.ndarray

        """
        if mask is None:
            self._overlay = None
        else:
            if not mask.flags['C_CONTIGUOUS'] or mask.dtype != np.uint8:
                mask = np.ascontiguousarray(mask, np.uint8)

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
        """Process the paint event by repainting this Widget.

        Parameters
        ----------
        event : QPaintEvent
        """
        painter = QPainter()
        painter.begin(self)
        self._drawImage(painter)
        self._drawMask(painter)
        painter.end()

    def _drawImage(self, painter: QPainter):
        """Draw current image into this ``QImageView``.

        Parameters
        ----------
        painter :   QPainter
        """
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
        """Display the given image.

        Parameters
        ----------
        painter :   QPainter
        """
        if self._image is not None and self._overlay is not None:
            painter.drawImage(self._imageRect, self._overlay)
