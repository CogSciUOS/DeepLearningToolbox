"""Implementation of abstract classes using the Qt library
(module `PyQt5`).
"""

# https://stackoverflow.com/questions/12718296/is-there-a-way-to-use-qt-without-qapplicationexec

# FIXME[todo]: the @protect decorator catches KeyboardInterrupt - this is
# not desirable if not run from Qt main event loop but some other
# context, if we would actually be interested in these exceptions (and
# maybe also others!)

# standard imports
from typing import Callable, Tuple, Optional
import time
import logging
import threading

# third party imports
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSlot
from PyQt5.QtGui import QKeyEvent, QCloseEvent, QHideEvent
from PyQt5.QtWidgets import QApplication

# toolbox imports
from qtgui.widgets.image import QImageView
from dltb.base.package import Package
from dltb.base.image import Image, Imagelike, ImageDisplay as BaseImageDisplay

# logging
LOG = logging.getLogger(__name__)


class QtPackage(Package):
    """An extended :py:class:`Package` for providing specific Qt
    information.

    """
    def __init__(self, **kwargs) -> None:
        super().__init__(key='qt', **kwargs)

    @property
    def version(self) -> Optional[str]:
        return f"{QtCore.QT_VERSION_STR}/{QtCore.PYQT_VERSION_STR}"


QtPackage()


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
        self._application.quit()

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

    def __init__(self, view: QImageView = None, **kwargs) -> None:
        """
        """
        if view is None:
            application = QApplication([])
            view = QImageDisplay(application)
        elif isinstance(view, QImageView):
            application = QApplication.instance()
        else:
            raise TypeError(f"{type} is not a QImageView.")
        super().__init__(view=view, **kwargs)
        self._application = application

    @property
    def view_(self) -> BaseImageDisplay:
        """The :py:class:`ImageView` used by this `ImageDisplay`.
        """
        return self._view

    def _show(self, image: np.ndarray, title: str = None, **kwargs) -> None:
        """Show the given image.

        Arguments
        ---------
        image: Imagelike
            The image to display.
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
        LOG.info("Qt: open: show the window")
        self._view.show()
        if self._blocking is None:
            # Make sure the the showEvent is processed.
            #
            # There seems to be some timing aspect to this: just
            # calling _application.processEvents() seems to be too
            # fast, we have to wait a bit otherwise _view.setData may
            # not have triggered the paintEvent.  This does only occur
            # sometimes and I have no idea how long to wait or how to
            # check if the situation is fine (interestingly it seems
            # not to matter if we wait before or after calling
            # _application.processEvents()).
            time.sleep(0.1)
            self._process_events()
            # time.sleep(0.1)
        LOG.debug("Qt: open: done.")

    def _close(self) -> None:
        LOG.info("Qt: close: hide the window")
        self._view.hide()
        if not self._event_loop_is_running():
            self._process_events()
        LOG.debug("Qt: close: done.")

    def _run_blocking_event_loop(self, timeout: float = None) -> None:
        """Start the main event loop for this :py:class:`ImageDisplay`.
        """
        LOG.info("Running Qt Main Event Loop.")
        # Run the Qt main event loop to update the display and
        # process timeout and/or key events.
        if self._event_loop_is_running():
            raise RuntimeError("Only one background thread is allowed.")

        self._event_loop = QThread.currentThread()
        if timeout is not None:
            milliseconds = int(timeout * 1000)
            timer = QTimer()
            timer.setInterval(milliseconds)
            timer.setSingleShot(True)
            timer.timeout.connect(self._view.onTimer)
            timer.start()
        LOG.debug("Starting Qt Main Event Loop (exec_)")
        self._application.exec_()
        LOG.debug("Qt Main Event Loop (exec_) has ended.")
        if timeout is not None:
            timer.stop()
            timer.timeout.disconnect(self._view.onTimer)
        self._event_loop = None

        LOG.info("Qt Main Event Loop finished (opened=%, active=%s, "
                 "closed=%s).", self.opened, self.active, self.closed)

    def _run_nonblocking_event_loop(self) -> None:
        # FIXME[hack]: calling _process_events from a background task
        # seems to have no effect for Qt. It seems that it really has
        # to be called from the main thread!
        # Hence we do not start a background thread here but instead
        # call process_events once. This will not result in a smooth
        # interface, but at least it will show the images.
        self._process_events()

    def _process_events(self) -> None:
        """Process events for the graphical user interface of
        this :py:class:`ImageDisplay`. Pending events are processed
        in a blocking mode.

        Note: Qt requires that event processing is run in the main
        thread.
        """
        self._application.processEvents()

    # FIXME[hack]: when running without an event loop (and even when
    # setting blocking=False and using explicit event processing),
    # we seem to be not notified when the window is closed (by clicking
    # the window decoration close button).
    # Hence we adapt opened and closed to explicitly check
    # the status of the window.  A future implementation should
    # improve this in several directions:
    # - check why we are not informed (we probably need some event handler)
    # - it seems suboptimal to have to adapt both methods
    #   (opened and closed)- just changing one should be enough
    #   -> redesign the class ...
        
    @property
    def opened(self) -> bool:
        """Check if this image :py:class:`Display` is opened, meaning
        the display window is shown and an event loop is running.
        """
        return self._view.isVisible() and self._opened

    @property
    def closed(self) -> bool:
        """Check if this image :py:class:`Display` is closed, meaning
        that no window is shown (and no event loop is running).
        """
        return not self._view.isVisible() or not self._opened
