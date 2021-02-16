"""Implementation of abstract classes using the Qt library
(module `PyQt5`).
"""

# https://stackoverflow.com/questions/12718296/is-there-a-way-to-use-qt-without-qapplicationexec

# FIXME[todo]: the @protect decorator catches KeyboardInterrupt - this is
# not desirable if not run from Qt main event loop but some other
# context, if we would actually be interested in these exceptions (and
# maybe also others!)

# standard imports
from typing import Callable, Tuple
import time
import logging
import threading

# third party imports
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSlot
from PyQt5.QtGui import QKeyEvent, QCloseEvent, QHideEvent
from PyQt5.QtWidgets import QApplication

# toolbox imports
from qtgui.widgets.image import QImageView
from ..base.image import Image, Imagelike, ImageDisplay as BaseImageDisplay

# logging
LOG = logging.getLogger(__name__)


class QImageDisplay(QImageView):

    def __init__(self, application: QApplication = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._application = application
        self._thread = None
        if self._application is not None:
            self._application.aboutToQuit.connect(self.onAboutToQuit)

    def closeEvent(self, event: QCloseEvent) -> None:
        """This event handler is called with the given event when Qt receives
        a window close request for a top-level widget from the window
        system.

        By default, the event is accepted and the widget is
        closed. You can reimplement this function to change the way
        the widget responds to window close requests. For example, you
        can prevent the window from closing by calling ignore() on all
        events.

        In other words: If you do not want your widget to be hidden,
        or want some special handling, you should reimplement the
        event handler and ignore() the event.

        """
        # event.ignore()
        LOG.info("QImageDisplay.closeEvent: accepted: %s", event.isAccepted())

    def hideEvent(self, event: QHideEvent) -> None:
        """Hide events are sent to widgets immediately after they have been
        hidden.
        """
        LOG.info("QImageDisplay.hideEvent: display was hidden")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """This event handler, for event event, can be reimplemented in a
        subclass to receive key press events for the widget.

        We add handling of 'Esc' and 'Q' to close the window.
        """
        key = event.key()
        if key in (Qt.Key_Q, Qt.Key_Escape) and self._application is not None:
            self._application.quit()
        else:
            super().keyPressEvent(event)

    @pyqtSlot()
    def onAboutToQuit(self) -> None:
        """A slot to be connected to the QApplicaton.aboutToQuit signal.
        It will inform this :py:class:`QImageDisplay` that the main
        event loop of the application is about to finish.

        This will not automatically close (hide) the
        :py:class:`QImageDisplay`.
        """
        LOG.info("QImageDisplay.onAboutToQuit: application.aboutToQuit")

    @pyqtSlot()
    def onTimer(self) -> None:
        """Slot to connect the `timeout` signal of a :py:class:`QTimer`.
        If a :py:class:`QApplication` is connected with this
        :py:class:`QImageDisplay`, its main event loop will be stopped.
        """
        if self._application is not None:
            self._application.quit()


class ImageDisplay(BaseImageDisplay):
    """An image display that uses Qt Widgets to display an image.

    The :py:class:`ImageDisplay` can be used in different modes.
    In standard mode, the widget is shown once the :py:meth:`show`
    method is called. No seperate Qt event loop is spawn and if
    non such loop exists, repaint events are processed explicitly
    in the current thread. In that situation, the graphical user
    interface shows the image, but it will be unresponsive (for example,
    pressing the window close button will have no effect).

    There is also a :py:class:`run` method that starts the Qt main
    event loop in the current thread and spawns a background QThread
    to run a worker. The worker can update the :py:class:`ImageDisplay`
    by calling the :py:class:`show` method.
    This mode requires a bit more effort on side of the caller
    (definition of a worker function), but it should guarantee a
    resoponsive user interface.

    Attributes
    ----------
    _view: QImageView
        A widget to display the image.
    _thread: QThread
        A thread running either the main event loop or a background
        worker (depending ont the mode of the display).
        If this is not None, a main event loop is running.
    _application: QApplication
        A application that will be started to get an responsive user
        interface. This can only be done from the Python main thread,
        and the event loop will then occupy this thread.
    """

    def __init__(self, view: QImageView = None,
                 worker: Callable = None, **kwargs) -> None:
        """
        """
        super().__init__(**kwargs)
        self._application = QApplication([])
        self._view = view or QImageDisplay(self._application)
        if worker is not None:
            self.run(worker=worker)

    def _show(self, image: np.ndarray, title: str = None, **kwargs) -> None:
        """Show the given image.

        Arguments
        ---------
        image: Imagelike
            The image to display.
        blocking: bool
            If `True`, run event loop in main thread, blocking further
            processing. If `False`, run event loop in background thread
            and return execution of main thread immediately to caller.
            If `None`, no event loop will be started. The caller is
            responsible for keeping a responsive interface, e.g., by
            regularly updating the display.
        unblock: str
            How to proceed when finishing blocking display. Options are
            'close': close the display and end all event loops,
            'show': continue showing the display with a non-blocking event
            loop (like blocking=False), or 'freeze': continue showing
            the display without event loop (like blocking=None).
        wait_for_key: bool
            If True, blocking display will be stopped when a key is pressed
            (or the window is closed).
            FIXME[todo]: the value of the key is returned
        timeout: float
            If `True`, the display will be stopped after the given number
            of seconds.
        title: str
            A title to be displayed as image title.
        """

        # Setting the image for _view will trigger a paintEvent
        # (via _view.update()) which we have to make sure is processed
        # for the image to become visible.
        self._view.setImage(image)

        title = "Qt" if title is None else f"Qt: {title}"
        self._view.setWindowTitle(title)

    def _open(self) -> None:
        self._view.show()
        if self._blocking is None:
            # Make sure the the showEvent is processed.
            # FIXME[hack]: There seems to be some timing aspect to this:
            # just calling _application.processEvents() seems to be
            # too fast, we have to wait a bit otherwise _view.setData
            # may not have triggered the paintEvent.
            # This does only occur sometimes and I have no idea
            # how long to wait or how to check if the situation is
            # fine (interestingly it seems not to matter if we wait
            # before or after calling _application.processEvents()).
            time.sleep(0.1)
            self._process_events()
            # time.sleep(0.1)

    def _close(self) -> None:
        self._view.hide()
        if self._blocking is None:
            self._process_events()

    def _run_blocking_event_loop(self, timeout: float = None) -> None:
        """Start the main event loop for this :py:class:`ImageDisplay`.
        """
        LOG.debug("Starting Qt Main Event Loop (exec_)")
        # Run the Qt main event loop to update the display and
        # process timeout and/or key events.
        if self._event_loop is not None:
            raise RuntimeError("Only one background thread is allowed.")

        self._event_loop = QThread.currentThread()
        if timeout is not None:
            milliseconds = int(timeout * 1000)
            timer = QTimer()
            timer.setInterval(milliseconds)
            timer.setSingleShot(True)
            timer.timeout.connect(self._view.onTimer)
            timer.start()
        self._application.exec_()
        if timeout is not None:
            timer.stop()
            timer.timeout.disconnect(self._view.onTimer)
        self._event_loop = None

        LOG.debug("Qt Main Event Loop (exec_) has ended.")

    def _process_events(self) -> None:
        print(f"{threading.currentThread()}: process qt events")
        self._application.processEvents()

    # ------------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        """A flag indicating if the :py:class:`ImageDisplay` was closed.
        `True` means that currently no image is shown. The
        :py:class:`ImageDisplay` can be reopened again by calling
        :py:meth:`show` again.
        """
        return not self._view.isVisible()

    @property
    def active(self) -> bool:
        """A flag indicating that the graphical user interface of this
        :py:class:`ImageDisplay` is active, i.e., an event loop is
        running.
        """
        return self._thread is not None

    def __show(self, image: np.ndarray, blocking: bool = True,
              unblock: str = 'close',
              wait_for_key: bool = False,
              timeout: float = None, title: str = None, **kwargs) -> None:
        """Show the given image.

        Arguments
        ---------
        image: Imagelike
            The image to display.
        blocking: bool
            If `True`, run event loop in main thread, blocking further
            processing. If `False`, run event loop in background thread
            and return execution of main thread immediately to caller.
            If `None`, no event loop will be started. The caller is
            responsible for keeping a responsive interface, e.g., by
            regularly updating the display.
        unblock: str
            How to proceed when finishing blocking display. Options are
            'close': close the display and end all event loops,
            'show': continue showing the display with a non-blocking event
            loop (like blocking=False), or 'freeze': continue showing
            the display without event loop (like blocking=None).
        wait_for_key: bool
            If True, blocking display will be stopped when a key is pressed
            (or the window is closed).
            FIXME[todo]: the value of the key is returned
        timeout: float
            If `True`, the display will be stopped after the given number
            of seconds.
        title: str
            A title to be displayed as image title.
        """
        LOG.debug("show: wait_for_key=%s, timeout=%r.", wait_for_key, timeout)

        if not blocking and wait_for_key:
            raise ValueError("wait_for_key is illegal "
                             "for non-blocking display.")

        # It seems essential that the QImageView is visible (on the
        # screen) before setting the data, otherwise _view.update()
        # (and even _view.repaint()) will not trigger a paintEvent(),
        # and only a black widget is shown. Notice that we not only have
        # to call _view.show(), but have have to make sure that the
        # event is processed.
        if self.closed:
            LOG.debug("show: showing the view.")
            # showing the view seems not to trigger a repaint event
            # for that widget.
            self._view.show()
            if not self.active:
                # Make sure the the showEvent is processed.
                # FIXME[hack]: There seems to be some timing aspect to this:
                # just calling _application.processEvents() seems to be
                # too fast, we have to wait a bit otherwise _view.setData
                # may not have triggered the paintEvent.
                # This does only occur sometimes and I have no idea
                # how long to wait or how to check if the situation is
                # fine (interestingly it seems not to matter if we wait
                # before or after calling _application.processEvents()).
                time.sleep(0.1)
                self._application.processEvents()
                # time.sleep(0.1)

        # Setting the image for _view will trigger a paintEvent
        # (via _view.update()) which we have to make sure is processed
        # for the image to become visible.
        self._view.setImagelike(image)

        if title is not None:
            self._view.setWindowTitle(title)

        if blocking:
            # Run the Qt main event loop to update the display and
            # process timeout and/or key events.
            if self._thread is not None:
                raise RuntimeError("Only one background thread is allowed.")

            self._thread = QThread.currentThread()
            if timeout is not None:
                milliseconds = int(timeout * 1000)
                timer = QTimer()
                timer.setInterval(milliseconds)
                timer.setSingleShot(True)
                timer.timeout.connect(self._view.onTimer)
                timer.start()
            self._runMainEventLoop()
            if timeout is not None:
                timer.stop()
                timer.timeout.disconnect(self._view.onTimer)
                timeout = None
            self._thread = None
            if unblock == 'close':
                self.close()
            elif unblock == 'show':
                blocking = False
            elif unblock == 'freeze':
                blocking = None

        if blocking is False:
            # Run a dummy event loop in background thread
            self._runDummyEventLoop(timeout=timeout)

        elif blocking is None:
            if self.active:
                raise RuntimeError("ImageDisplay should be frozen, "
                                   "but is still active.")
            # Process Qt events (repaint) synchronously (blocking)
            LOG.debug("show: updating the view")
            self._application.processEvents()

        LOG.debug("show: finished: view.isVisible(): %r",
                  self._view.isVisible())

    def close(self) -> None:
        """Close this :py:class:ImageDisplay. This will close the display
        window and stop all background threads.
        """
        self._view.hide()

    class WorkerThread(QThread):
        """An auxiliary class to realize the wait for key behaviour.
        This has to be a `QThread` (not a python thread), in order
        to connect to the Qt event system.
        """

        def __init__(self, worker: Callable, args: Tuple = (),
                     **_kwargs) -> None:
            super().__init__(**_kwargs)
            self._worker = worker
            self._args = args

        def run(self):
            """The code to be run in the thread. This will wait
            for the `stopped` :py:class:`Event`, either caused
            by the `keyPressedEvent` or abortion of the main
            event loop.
            """
            LOG.debug("Background QTthread starts running")
            self._worker(*self._args)
            LOG.debug("Background QTthread ended running")

    def run(self, worker: Callable, args: Tuple) -> None:
        """Start the given worker in a background :py:class:`QThread`,
        while the Qt main event loop in run in the current thread
        (which has to be the main thread).
        """
        if self._thread is not None:
            raise RuntimeError("Only one background thread is allowed.")

        self._thread = self.WorkerThread(worker, self._stop_worker)
        self._thread.finished.connect(self._application.quit)
        self._thread.start()
        self._runMainEventLoop()
        # FIXME[todo]: make sure that thread has finished!
        self._thread = None

    def _runMainEventLoop(self) -> None:
        """Start the main event loop for this :py:class:`ImageDisplay`.
        """
        LOG.debug("Starting Qt Main Event Loop (exec_)")
        self._application.exec_()
        LOG.debug("Qt Main Event Loop (exec_) has ended.")
