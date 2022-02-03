"""Interface for a busy object.  Such an object can be in a busy state,
meaning it is currently performing some task and is not ready to start
another task (in parallel, invoked from another thread) befort the first
task is fininished.



.. moduleauthor:: Ulf Krumnack

.. module:: dltb.base.busy

This module provides an API for busy objects.

"""

# standard imports
import threading
from types import MethodType
from contextlib import contextmanager
import logging

# toolbox imports
from .fail import FailableObservable
from .run import run_synchronous, run_asynchronous, should_run_asynchronous
from ..util.error import handle_exception

# logging
LOG = logging.getLogger(__name__)


def busy(message):
    """A decorator that marks a methods affecting the business of an
    BusyObservable. A method decorated this way sets the busy flag of
    the object and may not be called when this flag is already set.
    The change of the busy flag will be reported to the observers.
    """
    def decorator(function):
        def wrapper(self, *args, busy_message: str = message,
                    run: bool = None, **kwargs):
            method = MethodType(function, self)
            # FIXME[was]: self._busy_run(function, self, ...)
            return self._busy_run(method, *args, busy_message=busy_message,
                                  run=run, **kwargs)
        return wrapper
    return decorator


class BusyObject:
    """A class for instances that can be busy.  An object being busy means
    that it is performing some task and is not willing to do another
    task before that task has finished.

    A typical example of a `BusyObject` is a class with properties
    that hold some internal state of an operation.  In such a case the
    operation should not be invoked multiple times in parallel.


    Attributes
    ----------
    busy:
        A flag indicating whether the object is busy.
    busy_message:
        A short text describing the business.  `None` if the
        object currently is not busy.

    Private Attributes
    ------------------
    _busy: str
        A short text describing the task currently executed or None if
        this :py:class:`BusyObservable` is not busy.
    _busy_lock: threading.RLock
        A reentrant lock to ensure exclusive access to this
        :py:class:`BusyObservable`.
    _busy_thread: threading.Thread
        The thread performing the business

    """

    def __init__(self, *args, **kwargs):
        """Initialization of this :py:class:`BusyObservable`.
        The default state is that the object is not busy.
        """
        super().__init__(*args, **kwargs)
        self._busy = None
        self._busy_lock = threading.RLock()
        self._busy_thread = None

    @property
    def busy(self) -> bool:
        """A flag indicating if this :py:class:`BusyObservable` is
        busy (True) or not (False).
        """
        return self._busy is not None

    @property
    def busy_message(self) -> str:
        """A short text describing the kind of task the object
        is occupied with.
        """
        return self._busy if self._busy is not None else ''

    def _busy_run(self, func, *args, busy_message: str = None,
                  run: bool = None, _busy_takeover: bool = False,
                  **kwargs):
        """Become busy with running a function (or method). This private
        function starts all business.  It should not be called directly,
        but only from the `@busy` decorator.

        Arguments
        ---------
        func:
            The function (or method) to run.
        busy_message:
            A message describing the task currently performed.
        _busy_takeover:
            An internal flag, signaling that the function should be
            run in the current thread.
        *args:
            Positional arguments to be passed to the function.
        **kwargs:
            Keyword arguments to be passed to the function.
        """
        LOG.info("busy_run: {%s}(%s, %s): message='%s', async=%s, takeover=%s",
                 func, args, kwargs, busy_message, run, _busy_takeover)

        #
        # Step 1: Determine if we want to run asynchronously
        #
        if _busy_takeover:
            # Take business over to the current thread.
            self._busy_thread = threading.current_thread()
            LOG.debug("busy_run: taking over the thread")

        # We need to decide if we should run synchronously or
        # asynchronously. The main idea is that a graphical
        # user interface will usualy profit from running
        # asynchronously, as this will prevent us from
        # blocking the main event loop, while in other
        # situations synchronous processing is more suitable.
        if run is not None:
            # We are explicitly instructed how to run the method.
            LOG.debug("busy_run: run is %s", run)
        elif self._busy_thread is threading.current_thread():
            # we are already running in our own thread and hence do
            # not want to start a new one.
            run = False
            LOG.debug("busy_run: already running in an own busy thread.")
        else:
            LOG.debug("busy_run: automatically determining if to run "
                      "asynchronously.")

        #
        # Step 2: start the business
        #
        if should_run_asynchronous(run=run):
            #
            # asynchronous execution
            #

            # We will start the business in this thread (to make sure that
            # we can acquire the lock) but it will then be taken over by the
            # background thread, where it will also be ended.
            self._busy_start(busy_message)

            # set optional arguments
            kwargs['busy_message'] = busy_message
            kwargs['_busy_takeover'] = True
            # FIXME[old]: runner
            kwargs['runner'] = getattr(self, '_runner', None)

            # call this function (_busy_run) again in new Thread,
            # setting the argument '_busy_takeover' to signal that
            # we want to run in that thread.
            return run_asynchronous(self._busy_run, func, *args, **kwargs)

        #
        # synchronous execution
        #
        try:
            with self._busy_manager(busy_message):
                return run_synchronous(func, *args, **kwargs)
        except Exception as exception:
            if _busy_takeover:
                # we are running in new Thread, so no chance
                # to inform caller that something went wrong.
                # Use fallback exception handling to report
                # the error.
                handle_exception(exception)
                return None

            # we are running synchronously, so we can
            # pass the exception to the call.
            raise exception
        finally:
            if _busy_takeover:
                # we have taken the object from another Thread
                # that started the business - so we have to end it
                # here a second time (since we started it twice)
                self._busy_stop()

    def _busy_start(self, message: str):
        """Start some business.

        Starting a new business internally consists of two steps:
        (1) set the _busy_thread to be the current_thread. This may
            fail if the _busy_thread thread is already set to some other
            thread.
        (2) set _busy to the message briefly describing the business.

        Arguments
        ---------
        message:
            A short message describing the business.
        """
        old_thread = self._busy_thread
        if old_thread is None:
            # The object is not busy yet
            # pylint: disable=consider-using-with
            # (the standard context manager only supports blocking lock;
            # we could define an extra non-blocking lock context manager ...)
            if not self._busy_lock.acquire(blocking=False):
                raise RuntimeError("Could not acquire busy lock.")

            try:
                if self._busy_thread is None:  # make sure object is still free
                    self._busy_thread = threading.current_thread()
                else:
                    raise RuntimeError("Race condition setting busy thread.")
            finally:
                self._busy_lock.release()
        elif old_thread is not threading.current_thread():
            # Another thread is using this object
            raise RuntimeError(f"Object {self} is currently busy: "
                               f"{self._busy}")

        LOG.info("busy_start: '%s' -> '%s'", self._busy, message)
        old_busy = self._busy
        self._set_busy(message)
        return old_busy, old_thread

    def _busy_stop(self, busy_handle=None) -> None:
        """Stop a business. This will restore the message and release
        the lock.
        """
        old_busy, old_thread = busy_handle or (None, None)
        LOG.info("busy_stop: '%s' -> '%s'", self._busy, old_busy)
        if self._busy_thread is not threading.current_thread():
            raise RuntimeError("Cannot stop business of object "
                               "owned by another Thread.")
        self._busy_thread = old_thread
        self._set_busy(old_busy)

    @contextmanager
    def _busy_manager(self, message: str):
        """A context manager for a block of code to be run in busy mode.
        """

        old_busy = self._busy_start(message)
        try:
            yield None
        finally:
            self._busy_stop(old_busy)

    def _set_busy(self, message: str):
        """Private method to change the busy state. This method should
        only be called if running as _busy_thread.

        """
        if message != self._busy:
            LOG.debug("busy: new message: '%s' -> '%s'", self._busy, message)
            self._busy = message


class BusyObservable(BusyObject, FailableObservable, method='busy_changed',
                     changes={'busy_changed'}):
    """A :py:class:`BusyObservable` can inform observers on changes of
    the busy state.

    Changes
    -------
    state_changed:
        This change will be reported when a lazy object gets busy or
        a busy object gets lazy.
    busy_changed:
        This change will be reported whenever the kind of business,
        that is the description reported by :py:meth:`busy_message`
        changes.

    """

    def _set_busy(self, message: str):
        """Private method to change the busy state. This method should only be
        called if running as _busy_thread.

        """
        old_message = self._busy
        super()._set_busy(message)
        if old_message != message:
            change = self.Change('busy_changed')
            if (old_message is None) != (message is None):
                change.add('state_changed')
            self.notify_observers(change)
