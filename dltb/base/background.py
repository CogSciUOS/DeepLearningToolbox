"""Base class for using objects in background processes.
"""

# standard imports
import multiprocessing as mp
import threading
import signal
import atexit
import logging


# toolbox imports
from ..util.error import handle_exception

# logging
LOG = logging.getLogger(__name__)


class Backgroundable:
    """Base class for objects that are "backgroundable", meaning that
    that can be executed in a background process.  This may be useful
    for objects that perform intense of time-critical operations,
    that could either disturb the operation of the main process (e.g.,
    running the event loop of a graphical user interface), or that
    are disturbed themselves by such operation, e.g., for accurate
    audio or video operations.

    Properties
    ----------
    _background_process:
        The `Process` running the background operation.  This property
        is only set in the main process but not in the background
        process itself.

    _background_loop_event
        An event intended the main loop running in the background process
        that some action is required.  This event should usually not
        be set directly.  Instead the :py:method:`trigger_background_action`
        method should be called, that will set the event.

    _background_end_event
        An event indicating that the background main loop should be ended
        and the object reset to single process (non-backgrounded) operation.
        This event should not be set directly.  Instead, the property
        :py:prop:`backgrounded` should be set to `False`.

    Subclassing
    -----------
    Subclasses should overwrite the method :py:meth:`_background_action`,
    to perform background actions.  This method will be invoked in the
    background process and hence can only access the backgrounded
    version of the object.
    """
    _background_process: mp.Process = None
    in_background: bool = False

    _background_loop_event: mp.Event = mp.Event()
    _background_end_event: mp.Event = mp.Event()
    _background_queue_event: mp.Event = mp.Event()
    _background_queue: mp.Queue = mp.Queue()

    # FIXME[todo]: foreground thread implementation not finished yet
    _foreground_thread: threading.Thread = None
    _foreground_loop_end: bool = False
    _foreground_loop_event: mp.Event = mp.Event()

    _overwrite_sigint_handler: bool = False
    _original_sigint_handler = None

    def __init__(self, backgrounded: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)

        # set backgrounded flag (only for main process)
        # if mp.parent_process() is None:  # main process (only Python 3.8)
        if backgrounded:
            self.backgrounded = backgrounded

    def __del__(self) -> None:
        if self.backgrounded:
            self.backgrounded = False
        # there may be no super().__del__ ...
        getattr(super(), '__del__', lambda: None)()

    def __repr__(self) -> str:
        info = (f", Backkgroundable[backgrounded={self.backgrounded}, "
                + ("" if self._background_process is None else
                   f"alive: {self._background_process.is_alive()}, ") +
                f"loop={self._background_loop_event.is_set()}, "
                f"queue={self._background_queue_event.is_set()}, "
                f"end={self._background_end_event.is_set()}]")
        return super().__repr__() + info

    @property
    def backgroundable(self) -> bool:
        """A flag indicating if the object can be backgrounded.

        In some situations, backgrounding is not possible, for example
        if modules have been imported in the main process (and performed
        some initialization there), they may not be usable in a background
        process.
        """
        return True

    @property
    def backgrounded(self) -> bool:
        """A flag inidicating if this object is backgrounded, meaning
        that it is performing its operation in a background process.

        The property will only be `True` in the main process.
        """
        return self._background_process is not None

    @backgrounded.setter
    def backgrounded(self, backgrounded: bool) -> None:
        if backgrounded == self.backgrounded:
            return  # nothing changed
        if backgrounded:
            self._begin_background_loop()
        else:
            self._end_background_loop()

    def _begin_background_loop(self) -> None:
        """Start the background loop. This will put the object in the
        `backgrounded` mode.

        This method must be called in the main process.  If the background
        loop is already running, the function will have no effect.
        """
        if self._background_process is not None:
            return  # backgrounded has already been started

        if not self.backgroundable:
            raise RuntimeError(f"Object {self} can (no longer) be "
                               "backgrounded.")

        self._background_loop_event.clear()
        self._background_end_event.clear()
        background_process = mp.Process(target=self._background_loop)
        background_process.start()
        self._background_process = background_process

        self._foreground_loop_event.clear()
        foreground_thread = threading.Thread(target=self._foreground_loop)
        foreground_thread.start()
        self._foreground_thread = foreground_thread

        if self._overwrite_sigint_handler:
            LOG.info("Press Ctrl+C to stop the background process")
            Backgroundable._original_sigint_handler = \
                signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self.signal_handler)

        atexit.register(self._end_background_loop)

    def _end_background_loop(self, wait: bool = False) -> None:
        """End backgrounded.

        Note: this method has to be called from the parent process.
        """
        if not self.backgrounded:
            LOG.info("Not ending background loop: object is not backgrounded.")
            return  # backgrounded has already ended

        # stop the background process
        if wait:
            LOG.debug("Waiting for process '%s' (alive=%s) to join.",
                      self._background_process.name,
                      self._background_process.is_alive())
        else:
            LOG.debug("Forcing process '%s' (alive=%s) to end.",
                      self._background_process.name,
                      self._background_process.is_alive())
            self._stop_background_action()

        self._background_process.join()  # waits unti process ended
        self._background_process = None

        # Now also stop the foreground loop thread
        if self._foreground_thread is not None:
            self._foreground_loop_end = True
            self._foreground_loop_event.set()
            self._foreground_thread.join()  # waits until thread ended
            del self._foreground_thread

        # Unset exit and signal handler
        atexit.unregister(self._end_background_loop)

        if Backgroundable._original_sigint_handler is not None:
            signal.signal(signal.SIGINT,
                          Backgroundable._original_sigint_handler)
            Backgroundable._original_sigint_handler = None
        LOG.info("Backgrounding ended.")

    def _foreground_loop(self) -> None:
        """Run a foreground loop to receive notifications from the
        background process.
        """
        LOG.info("Starting foreground loop")
        try:
            while not self._foreground_loop_end:
                LOG.debug("Foreground loop is waiting for loop event ...")
                self._foreground_loop_event.wait()
                self._foreground_loop_event.clear()
                LOG.debug(" ... loop event in foreground loop!")
                self._foreground_action()
        except KeyboardInterrupt:
            LOG.warning("KeyboardInterrupt in foreground loop")
        except BaseException as exception:  # pylint: disable=broad-except
            LOG.error("Unhandled exception in foreground loop: %s", exception)
            handle_exception(exception)
        finally:
            LOG.info("Foreground loop ended (end=%s)",
                     self._foreground_loop_end)
            # The foreground loop should only be ended in response to
            # an official request from the _end_background_loop method,
            # and that method is responsible for joining and cleaning the
            # loop thread.
            if not self._background_end_event.is_set():
                LOG.warning("Background loop ended without the end event. "
                            "Process may not be cleaned up correctly - "
                            "consider explicitly setting "
                            "obj.backgrounded=False.")

    def _foreground_action(self) -> None:
        """Perform actions in the foreground process.
        This method is intended to react to events triggered in
        the background process. It may uses means of inter process
        communication, like shared values, events, queues etc. to access
        information from the background process.

        This method should be overwrittn by subclasses to perform
        subclass specific actions.  Such an implementation should
        usually contain an invocation of `super()._foreground_action()`,
        to include default actions.
        """
        if self._background_end_event.is_set():
            self._foreground_loop_end = True

    def _background_loop(self) -> None:
        LOG.info("Starting background loop")
        type(self).in_background = True
        try:
            while not self._background_end_event.is_set():
                try:
                    LOG.debug("Background loop is waiting for loop event ...")
                    self._background_loop_event.wait()
                    self._background_loop_event.clear()
                    LOG.debug(" ... loop event in background loop!")
                    if self._background_end_event.is_set():
                        break
                    self._background_action()
                except KeyboardInterrupt:
                    LOG.warning("KeyboardInterrupt in background loop ignored "
                                "- set obj.backrounded=False to end the loop.")
        except BaseException as exception:  # pylint: disable=broad-except
            LOG.error("Unhandled exception in background loop: %s", exception)
            handle_exception(exception)
        finally:
            LOG.info("Background loop ended (end=%s)",
                     self._background_end_event.is_set())
            self._background_end_event.set()
            self._foreground_loop_event.set()

    def trigger_background_action(self) -> None:
        """Trigger the execution of a background action (in the background
        process).  This method should be called in the main process and
        will then inform the background process to call the
        :py:meth:`_background_action` method.
        """
        self._background_loop_event.set()

    def _background_action(self) -> None:
        """This method is to be implemented in subclasses to perform
        background actions.  It is only invoked in the background process.
        """
        if self._background_queue_event.is_set():
            while not self._background_queue.empty():
                name = self._background_queue.get()
                value = self._background_queue.get()
                LOG.debug("background process: set attribute '%s' to %s",
                          name, type(value))
                setattr(self, name, value)
            self._background_queue_event.clear()

    def _stop_background_action(self) -> None:
        """This method is to be implemented in subclasses to stop
        actions currently runnding in the background process (started
        by :py:meth:`_background_action`). This method is only invoked
        in the background process.
        """
        self._background_end_event.set()
        self._background_loop_event.set()

    def signal_handler(self, signum: int, frame) -> None:
        """A signal handler to react to interrupts. It will end the
        background process, resetting the object to normal operation.
        """
        if self.backgrounded:
            LOG.warning("Received Ctrl+C (process=%s, signum=%s, frame=%s) - "
                        "Ending backgrounded", mp.current_process().name,
                        signum, type(frame))
            # self._end_background_loop()
            self.backgrounded = False

    def set_background_attribute(self, name, value) -> None:
        """Set an attribute for this object.  If backgrounded,
        the attribute will also be set in the background process.
        This method is intended to be called in the main process.
        """
        setattr(self, name, value)
        LOG.debug("main process: set attribute '%s' to %s [backgrounded=%s]",
                  name, type(value), self.backgrounded)
        if self.backgrounded:
            self._background_queue.put(name)
            self._background_queue_event.set()
            self.trigger_background_action()
            self._background_queue.put(value)
