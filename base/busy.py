"""
.. moduleauthor:: Ulf Krumnack

.. module:: dltb.base.busy

This module provides an API for busy objects.

"""

# standard imports
import threading
from contextlib import contextmanager
import logging

# toolbox imports
from util.error import handle_exception
from .fail import FailableObservable

# logging
LOG = logging.getLogger(__name__)


def busy(message):
    """A decorator that marks a methods affecting the business of an
    BusyObservable. A method decorated this way sets the busy flag of
    the object and may not be called when this flag is already set.
    The change of the busy flag will be reported to the observers.
    """
    def decorator(function):
        def wrapper(self, *args, **kwargs):
            kwargs['busy_message'] = message
            return self._busy_run(function, self, *args, **kwargs)
        return wrapper
    return decorator


class BusyObservable(FailableObservable, changes={'busy_changed'}):
    """A :py:class:`BusyObservable` object provides configuration data.
    It is an :py:class:Observable, allowing :py:class:Engine and user
    interfaces to be notified on changes.

    Changes
    -------
    state_change:
        This change will be reported when a lazy object gets busy or
        a busy object gets lazy.
    busy_change:
        This change will be reported whenever the kind of business,
        that is the description reported by :py:meth:`busy_message`
        changes.

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

    def _set_busy(self, message: str):
        """Private method to change the busy state. This method should only be
        called if running as _busy_thread.

        """
        if message != self._busy:
            LOG.debug("busy: new message: '%s' -> '%s'", self._busy, message)
            change = self.Change('busy_changed')
            if (self._busy is None) != (message is None):
                change.add('state_changed')
            self._busy = message
            self.notify_observers(change)

    def busy_start(self, message: str):
        """Start some business.
        """
        old_thread = self._busy_thread
        if old_thread is None:
            # The object is not busy yet
            if not self._busy_lock.acquire(blocking=False):
                raise RuntimeError("Could not acquire busy lock.")
            if self._busy_thread is None:  # make sure object is still free
                self._busy_thread = threading.current_thread()
                self._busy_lock.release()
            else:
                self._busy_lock.release()
                raise RuntimeError("Race condition setting busy thread.")
        elif old_thread is not threading.current_thread():
            # Anogher thread is using this object
            raise RuntimeError(f"Object {self} is currently busy: "
                               f"{self._busy}")

        LOG.info("busy_start: '%s' -> '%s'", self._busy, message)
        old_busy = self._busy
        self._set_busy(message)
        return old_busy, old_thread

    def busy_change(self, message):
        """Change the busy message during a business.
        """
        if self._busy_thread is not threading.current_thread():
            raise RuntimeError("Cannot change business of object "
                               "owned by another Thread.")
        if not self._busy:
            raise RuntimeError("Cannot change business of lazy object.")
        self._set_busy(message)

    def busy_stop(self, busy_handle=None) -> None:
        """Stop a business. This will restore the message and realease
        the lock.
        """
        old_busy, old_thread = busy_handle or (None, None)
        LOG.info("busy_stop: '%s' -> '%s'", self._busy, old_busy)
        if self._busy_thread is not threading.current_thread():
            raise RuntimeError("Cannot stop business of object "
                               "owned by another Thread.")
        self._busy_thread = old_thread
        self._set_busy(old_busy)

    def _busy_run(self, func, *args, busy_message: str = None,
                  busy_async: bool = None, busy_takeover: bool = False,
                  **kwargs):
        LOG.info("busy_run: {%s}(%s, %s): message='%s', async=%s, takeover=%s",
                 func, args, kwargs, busy_message, busy_async, busy_takeover)

        #
        # Determine if we want to run asynchronously
        #
        if busy_takeover:
            # Take business over to the current thread.
            self._busy_thread = threading.current_thread()
            LOG.debug("busy_run: taking over the thread")

        # We need to decide if we should run synchronously or
        # asynchronously. The main idea is that a graphical
        # user interface will usualy profit from running
        # asynchronously, as this will prevent us from
        # blocking the main event loop, while in other
        # situations synchronous processing is more suitable.
        if busy_async is not None:
            # We are explicitly instructed how to run the method.
            LOG.debug("busy_run: busy_async is %s", busy_async)
        elif self._busy_thread is threading.current_thread():
            # we are already running in our own thread and hence do
            # not want to start a new one.
            busy_async = False
            LOG.debug("busy_run: running in own thread")
        elif getattr(threading.current_thread(), 'GUI_event_loop', False):
            # If we are not the GUI event loop Thread, then
            # there is probably no need to start a new Thread:
            busy_async = True
            LOG.debug("busy_run: running async in a GUI eventloop")

        if busy_async:
            #
            # asynchronous execution
            #

            # We will start the business in this thread (to make sure that
            # we can acquire the lock) but it will then be taken over by the
            # background thread, where it will also be ended.
            self.busy_start(busy_message)
            try:
                kwargs['busy_message'] = busy_message
                kwargs['busy_takeover'] = True

                runner = getattr(self, '_runner', None)
                if runner is None:
                    threading.Thread(target=self._busy_run,
                                     args=(func,) + args,
                                     kwargs=kwargs).start()
                else:
                    runner.run_task(self._busy_run, func, *args, **kwargs)
            except Exception as exception:
                # FIXME[todo]: exception handling concept - we need to
                # deal with asynchronous situations
                handle_exception(exception)

        else:
            #
            # synchronous execution
            #
            try:
                with self.busy_manager(busy_message):
                    return func(*args, **kwargs)
            except Exception as exception:
                if busy_takeover:
                    # we are running in new Thread, so no chance
                    # to inform caller that something went wrong.
                    # Use fallback exception handling to report
                    # the error.
                    handle_exception(exception)
                else:
                    # we are running synchronously, so we can
                    # pass the exception to the call.
                    raise exception
            finally:
                if busy_takeover:
                    # we have taken the object from another Thread
                    # that started the business - so we have to end it
                    # here
                    self.busy_stop()

    @contextmanager
    def busy_manager(self, message: str):
        """A context manager for a block of code to be run in busy mode.
        """

        old_busy = self.busy_start(message)
        try:
            yield None
        finally:
            self.busy_stop(old_busy)
