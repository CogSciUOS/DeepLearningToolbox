from PyQt5.QtCore import QObject, pyqtSignal

from controller import AsyncRunner
from base.observer import Observable

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QtAsyncRunner(AsyncRunner, QObject):
    """:py:class:`AsyncRunner` subclass which knows how to update Qt widgets.

    The important thing about Qt is that you must work with Qt GUI
    only from the GUI thread (that is the main thread). The proper way
    to do this is to notify the main thread from worker and the code
    in the main thread will actually update the GUI.

    Another point is that there seems to be a difference between
    python threads and Qt threads. When interacting with Qt always
    use Qt threads.

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
        AsyncRunner.__init__(self)
        QObject.__init__(self)
        self._completion_signal.connect(self._notifyObservers)

    def onCompletion(self, future):
        """Emit the completion signal to have
        :py:meth:`Observable.notifyObservers` called.

        This method is still executed in the runner Thread. It will
        emit a pyqtSignal that is received in the main Thread and
        therefore can notify the Qt GUI.
        """
        result = future.result()
        if result is not None:
            observable, info = result
            import threading
            me = threading.current_thread().name
            logger.debug(f"{self.__class__.__name__}.onCompletion():{info}")
            if isinstance(info, Observable.Change):
                self._completion_signal.emit(observable, info)

    def _notifyObservers(self, observable, info):
        """The method is intended as a receiver of th pyqtSignal.
        It will be run in the main Thread and hence can notify
        the Qt GUI.
        """
        observable.notifyObservers(info)


from PyQt5.QtCore import QObject, QMetaObject, pyqtSlot, Q_ARG
from base.observer import Observable
import threading

class QObserver:
    """This as a base class for all QWidgets that shall act as
    Observers in the toolbox. It implements support for asynchronous
    message passing to Qt's main event loop.

    This class provides a convenience method :py:meth:`observe`, which
    should be used to observer some :py:class:`Observable` (instead
    of calling :py:meth:`Observable.add_observer` directly). This will
    set up the required magic.
    """
    _qObserverHelper = None

    def __init__(self):
        # FIXME[question]: do we have to call this from the main thread
        #if not threading.current_thread() is threading.main_thread():
        #    raise RuntimeError("QObserver.observe must be called from"
        #                       "the main thread, not " +
        #                       threading.current_thread().name + ".")
        if self._qObserverHelper is  None:
            self._qObserverHelper = QObserver.QObserverHelper(self)
       

    def observe(self, observable: Observable, interests=None):
        if self._qObserverHelper is  None:
            raise RuntimeError("It seems QObserver's constructor was not "
                               "called (QObserverHelper is not set)")

        observable.add_observer(self._qObserverHelper,
                                notify=self.QObserverHelper._qNotify,
                                interests=interests)

    def unobserve(self, observable):
        observable.remove_observer(self._qObserverHelper)

    def _exchangeView(self, name, new_view, interests=None):
        old_view = getattr(self, name)
        if new_view == old_view:
            return  # nothing to do
        if old_view is not None:
            old_view.remove_observer(self)
        setattr(self, name, new_view)
        if new_view is not None:
            new_view.add_observer(self, interests)

    class QObserverHelper(QObject):
        """A helper class for the :py:class:`QObserver`.

        We define the functionality in an extra class to avoid
        problems with multiple inheritance (:py:class:`QObserver` is
        intended to be inherited from in addition to some
        :py:class:`QWidget`): The main problem with QWidgets and
        multiple inheritance is, that signals and slots are only
        inherited from the first super class (which will usually be
        QWidget or one of its subclasses), but we have to define our
        own pyqtSlot here to make asynchronous notification work.

        This class has to inherit from QObject, as it defines an
        pyqtSlot.
        """
        def __init__(self, observer, parent=None):
            """Initialization of the QObserver.
            """
            super().__init__(parent)
            self._observer = observer
            self._change = None
            
        def _qNotify(observable, self, change):
            if self._change is None:
                # Currently, there is no change event pending for this
                # object. So we will queue a new one and remember the
                # change:
                self._change = change
                # This will use Qt.AutoConnection: the member is invoked
                # synchronously if obj lives in the same thread as the caller;
                # otherwise it will invoke the member asynchronously.
                QMetaObject.invokeMethod(self, '_qAsyncNotify',
                                         Q_ARG("PyQt_PyObject", observable))
                #     Q_ARG("PyQt_PyObject", change))
                #   RuntimeError: QMetaObject.invokeMethod() call failed
            else:
                # There is already one change pending in the event loop.
                # We will just update the change, but not queue another
                # event.
                self._change |= change

        @pyqtSlot(object)
        def _qAsyncNotify(self, observable):
            change = self._change
            if change is not None:
                self._change = None
                observable.notify(self._observer, change)


import numpy as np
from scipy.misc import imresize

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


from PyQt5 import QtCore
from PyQt5.QtWidgets import QPlainTextEdit
import threading
import logging

from collections import deque

class QLogHandler(QPlainTextEdit, logging.Handler):
    """A log handler that displays log messages in a QWidget.
    """

    _message_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        QPlainTextEdit.__init__(self, parent)
        logging.Handler.__init__(self)
        self.setReadOnly(True)
        self._counter = 1
        self._message_signal.connect(self.appendMessage)
        self._message_signal.emit("Log view initialized")

    def __len__(self):
        """The number of lines in this QLogHandler.
        """
        return self._counter

    def clear(self):
        """Clear this :py:class:QLogHandler.
        """
        super().clear()
        self._counter = 1
        self._message_signal.emit("Log view cleared")

    @QtCore.pyqtSlot(str)
    def appendMessage(self, message: str):
        self.appendPlainText(message)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def emit(self, record):
        """Handle a :py:class:logging.logRecord.
        """
        # Here we have to be careful: adding the text directly to the
        # widget from another thread causes problems: The program
        # crashes with the following message:
        #   QObject::connect: Cannot queue arguments of type 'QTextBlock'
        #   (Make sure 'QTextBlock' is registered using qRegisterMetaType().)
        # Hence we are doing this via a signal now.        
        self._counter += 1
        self._message_signal.emit(self.format(record))

