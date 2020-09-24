"""Implementation of abstract classes using the Qt library
(module `PyQt5`).
"""

# FIXME[bug]: closing the Qt window (e.g., by pressing the close button
# of the window manager) will cause an error on next invocation of
# ImageDisplay.show():
#   AttributeError: 'QImageDisplay' object has no attribute '_thread'

# https://stackoverflow.com/questions/12718296/is-there-a-way-to-use-qt-without-qapplicationexec

# standard imports
from typing import Callable
from threading import Event
import time

# third party imports
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSlot
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QApplication

# toolbox imports
from qtgui.widgets.image import QImageView
from ..base.image import Image, Imagelike, ImageDisplay as BaseImageDisplay


class QImageDisplay(QImageView):

    def __init__(self, application: QApplication = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._application = application
        self._thread = None
        if self._application is not None:
            self._application.aboutToQuit.connect(self.onClose)

    def setThread(self, thread: QThread) -> None:
        self._thread = thread

    def closeEvent(self, event):
        print("Close Event")
        if self._thread is not None:
            self._thread.stop()
            self._thread.wait()
            self._thread = None
            print("Waited for thread")
        if self._application is not None:
            self._application.quit()
        print("Ok to close now.")
        
    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        print("key press event: ", key)
        if key in (Qt.Key_R, Qt.Key_Escape) and self._application is not None:
            self._application.quit()

    @pyqtSlot()
    def onClose(self) -> None:
        print("Going to close the window")

    def onTimer(self) -> None:
        if self._application is not None:
            self._application.quit()


class ImageDisplay(BaseImageDisplay):
    """An image display that uses Qt Widgets to display an image.

    The :py:class:`ImageDisplay` can be used in different modes.
    In standard mode, the widget is shown once the :py:meth:`show`
    method is called. No seperate Qt event loop is spawn and if
    non such loop exists, repaint events are processed explicitly
    in the current thread. In that situation, the graphical user
    interface shows the image, but it may be unresponsive.

    There is also a :py:class:`run` method that starts the Qt main
    event loop in the current thread and spawns a background QThread
    to run a worker. The worker should can update the
    :py:class:`ImageDisplay` by calling the :py:class:`show` method.
    This mode requires a bit more effort on side of the caller
    (definition of a worker function), but it should guarantee a
    resoponsive user interface.

    Attributes
    ----------
    _view: QImageView
        A widget to display the image.
    _thread: QThread
        A thread running a background worker.
    """

    def __init__(self, view: QImageView = None,
                 worker: Callable = None, **kwargs) -> None:
        """
        """
        super().__init__(**kwargs)
        self._application = QApplication([])
        self._view = view or QImageDisplay(self._application)
        self._thread = None
        if worker is not None:
            self.run(worker=worker)

    @property
    def closed(self) -> bool:
        return self._view.isVisible()

    def show(self, image: Imagelike, wait_for_key: bool = False,
             timeout: float = None, **kwargs) -> None:
        """Show the given image.
        """
        print(f"show: wait_for_key={wait_for_key}, timeout={timeout}.")
        self._view.setData(Image.as_data(image))
        if not self._view.isVisible():
            print("Showing the view 1", self._view.isVisible())
            self._view.show()
            print("Showing the view 2", self._view.isVisible())
            if self._thread is None:
                print("Showing the view 3", self._view.isVisible())
                self._application.processEvents()
                print("Showing the view 4", self._view.isVisible())

        if self._thread is None:
            print("Updating the view")
            # process Qt events (repaint) synchronously (blocking)
            self._view.update()
            self._application.processEvents()
    
        if timeout is not None or wait_for_key:
            if self._thread is not None:
                raise RuntimeError("Only one background thread is allowed.")

            milliseconds = int(timeout * 1000)
            QTimer.singleShot(milliseconds, self._view.onTimer)
            self._application.exec_()

        elif False: # old
            # FIXME[question]: do we really have to create a new QThread
            # here to run time and key event processing, or can we do
            # this using a QTime object
            stopped = Event()
            thread = self.WaitForKeyThread(self._view, stopped,
                                           timeout=timeout)
            thread.finished.connect(self._application.quit)
            self._view.setThread(thread)
            thread.start()
            self._application.aboutToQuit.connect(thread.onClose)
            self._application.exec_()
            stopped.set()
        print("show: finished.")

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


    class WorkerThread(QThread):
        """An auxiliary class to realize the wait for key behaviour.
        This has to be a `QThread` (not a python thread), in order
        to connect the `QImageView.keyPressed` signal
        """

        def __init__(self, worker: Callable, stopped: Event, **kwargs) -> None:
            super().__init__(**kwargs)
            self._worker = worker
            self._stopped = stopped
        
        def run(self):
            """The code to be run in the thread. This will wait
            for the `stopped` :py:class:`Event`, either caused
            by the `keyPressed` signal or abortion of the main
            event loop.
            """
            print("Background QTthread starts running")
            self._worker(self._stopped)
            print("Background QTthread ended running")

        @pyqtSlot()
        def onClose(self):
            print("Going to close the window")
            self.stop()

        def stop(self):
            self._stopped.set()


    def run(self, worker: Callable) -> None:
        """
        """
        if self._thread is not None:
            raise RuntimeError("Only one background thread is allowed.")

        self._stop_worker = Event()
        self._thread = self.WorkerThread(worker, self._stop_worker)
        self._thread.finished.connect(self._application.quit)
        # self._thread.finished.connect(self._thread._onFinished)
        self._thread.start()
        self._application.aboutToQuit.connect(self._thread.onClose)
        print("Starting Qt Main Event Loop (exec_)")
        self._application.exec_()
        # FIXME[todo]: make sure that thread has finished!
        print("Qt Main Event Loop (exec_) has ended.")
        self._stop_worker.set()
        self._thread = None

