"""Implementation of abstract classes using the Qt library
(module `PyQt5`).
"""

# standard imports
from threading import Event

# third party imports
from PyQt5.QtCore import pyqtSlot, QThread
from PyQt5.QtWidgets import QApplication

# toolbox imports
from qtgui.widgets.image import QImageView
from ..base.image import Image, Imagelike, ImageDisplay as BaseImageDisplay

class QImageDisplay(QImageView):

    def setThread(self, thread: QThread) -> None:
        self._thread = thread
    
    def closeEvent(self, event):
        print("Close Event")
        if self._thread is not None:
            self._thread.stop()
            self._thread.wait()
            self._thread = None
            print("Waited for thread")
        print("Ok to close now.")


class ImageDisplay(BaseImageDisplay):
    """An image display that uses Qt Widgets to display an image.

    The display can be used in different modes. In standard mode,
    the widget is shown once the :py:meth:`show` method is called.
    In loop mode, the widget is shown upon construction and
    updated each time when :py:meth:`show` is called.  This is
    intended to be used in a loop, e.g., when reading images from
    a movie or a webcam.

    Attributes
    ----------
    _view: QImageView
        A widget to display the image.
    _loop: bool
        A flag indicating that the display is run in loop mode.
    """

    def __init__(self, view: QImageView = None,
                 loop: bool = False, **kwargs) -> None:
        """
        """
        super().__init__(**kwargs)
        self._application = QApplication([])
        self._view = view or QImageDisplay()
        self._loop = loop
        if loop:
            self._view.show()

    def show(self, image: Imagelike, wait_for_key: bool = False,
             timeout: float = None, **kwargs) -> None:
        """Show the given image.
        """
        self._view.setData(Image.as_data(image))
        if not self._loop:
            self._view.repaint()
            self._view.show()
        if wait_for_key:
            stopped = Event()
            thread = self.WaitForKeyThread(self._view, stopped,
                                           timeout=timeout)
            thread.finished.connect(self._application.quit)
            self._view.setThread(thread)
            thread.start()
            self._application.aboutToQuit.connect(thread.onClose)
            self._application.exec_()
            stopped.set()

    class WaitForKeyThread(QThread):
        """An auxiliary class to realize the wait for key behaviour.
        This has to be a `QThread` (not a python thread), in order
        to connect the `QImageView.keyPressed` signal
        """

        def __init__(self, view: QImageView, stopped: Event,
                     timeout: float = None, **kwargs) -> None:
            super().__init__(**kwargs)
            self._view = view
            self._stopped = stopped
            self._timeout = timeout

        def run(self):
            """The code to be run in the thread. This will wait
            for the `stopped` :py:class:`Event`, either caused
            by the `keyPressed` signal or abortion of the main
            event loop.
            """
            self._view.keyPressed.connect(self.on_key_pressed)
            self._stopped.wait(timeout=self._timeout)
            self._view.keyPressed.disconnect(self.on_key_pressed)

        @pyqtSlot(int)
        def on_key_pressed(self, _key: int) -> None:
            """React to keyPressed signals by setting the `_stopped`
            event.
            """
            self.stop()

        @pyqtSlot()
        def onClose(self):
            print("Going to close the window")
            self.stop()

        def stop(self):
            self._stopped.set()

