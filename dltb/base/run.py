"""Support for asynchronous (non-blocking, parallel, background
thread) function invocation.

.. moduleauthor:: Ulf Krumnack

.. module:: dltb.base.run

This module provides an API for busy objects.

"""

# standard imports
from typing import Callable, Any, Optional, Union
from threading import Thread, current_thread, main_thread
from queue import Queue
from contextlib import contextmanager
import logging

# toolbox imports
from ..util.error import handle_exception

# logging
LOG = logging.getLogger(__name__)


def run_synchronous(function: Callable, *args,
                    run_callback: Optional[Callable] = None, **kwargs) -> Any:
    """Run a function (synchronously).

    Arguments
    ---------
    function:
        the function to be called.
    run_callback:
        If not `None`, the function will be called with the results of
        the function call as argument(s).
    """
    result = function(*args, **kwargs)
    if run_callback is None:
        return result

    if result is None:
        run_callback()
    elif isinstance(result, tuple):
        run_callback(*result)
    else:
        run_callback(result)
    return None


def run_asynchronous(function: Callable, *args,
                     run_name: Optional[str] = None,
                     run_callback: Optional[Callable] = None,
                     runner: Optional = None,
                     **kwargs) -> Thread:
    """Invoke a function asynchronously.
    """

    try:
        if runner is not None:
            # FIXME[old]: we have a runner - use it to run the function
            return runner.run_task(function, *args, **kwargs)

        args = (function, ) + args
        kwargs['run_callback'] = run_callback
        thread = Thread(target=run_synchronous, name=run_name,
                        args=args, kwargs=kwargs)
        thread.start()
        return thread
    except Exception as exception:  # pylint: disable=broad-except
        handle_exception(exception)
        return None


def should_run_asynchronous(run: bool = None, run_name: Optional[str] = None,
                            run_callback: Optional[Callable] = None) -> bool:
    """Check if a function should be run asynchronously.
    Several heuristics are applied to answer this questions.
    """
    if run is not None:
        return run

    # automatically determine if we want to run asynchronously
    if run_callback is not None:
        return True

    if run_name is not None:
        return True

    if getattr(current_thread(), 'GUI_event_loop', False):
        # If we are in the GUI event loop Thread, then
        # it is probably better to start a new Thread:
        return True

    # it seems we are already running in a background thread
    return False


def runnable(function) -> Callable:
    """A decorator to mark a function as runnable.  A runnable function
    can be executed asynchronously.

    Arguments and results documented below refer to the function marked as
    `@runnable`.

    Arguments
    ---------
    run: bool
        A flag indicating whether the function should be run asynchronously
        (in some background thread). If `True`, the function will return
        the `Thread` object.  In `None`, the function automatically determines
        whether to run synchronously (blocking) or asynchronously.
    run_name: str
        A name for the background thread.  If provided, the function
        will be run asynchronously.
    run_callback:
        A callback to be invoked with the result of the main function
        invocation.

    Result
    ------
    result:
        If run synchronously (`run=False`, blocking, no background thread),
        the function returns the result from the function call.
    thread:
        If run asynchronously (`run=True`, non-blocking, background thread),
        the function returns the `Thread` object.
    """

    def wrapper(*args, run: bool = None, run_name: Optional[str] = None,
                run_callback: Optional[Callable] = None,
                **kwargs) -> Union[Any, Thread]:
        if should_run_asynchronous(run=run, run_name=run_name,
                                   run_callback=run_callback):
            return run_asynchronous(function, *args, run_name=run_name,
                                    run_callback=run_callback, **kwargs)

        return run_synchronous(function, *args, **kwargs)

    return wrapper


def main_thread_only(function) -> Callable:
    """A decorator to mark a function that should can only be called
    in the main thread.  Calling the function from a background `Thread`
    will result in a `RuntimeError`.
    """

    def wrapper(*args, **kwargs) -> Any:
        if current_thread() is not main_thread():
            raise RuntimeError(f"Function {function.__name__} can only "
                               "be called from the main thread.")
        return function(*args, **kwargs)

    return wrapper


class MainThreader:
    """A class that has functions that can only be executed in the main
    thread.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._main_thread_loop_running = False
        self._main_thread_activities = Queue()

    def add_main_thread_activity(self, function, args, kwargs) -> None:
        """Register an activity to be performed in the thread.
        """
        self._main_thread_activities.put((function, args, kwargs))
        if not self._main_thread_loop_running:
            LOG.warning("Function %s is called from a background thread, "
                        "but there is no main thread loop running to "
                        "execute the function.", function.__name__)

    def perform_main_thread_activities(self) -> None:
        """Perform the registered main thread activities.
        """
        while not self._main_thread_activities.empty():
            function, args, kwargs = self._main_thread_activities.get()
            function(*args, **kwargs)

    @contextmanager
    def main_thread_loop(self):
        """A context manager for running a main thread loop.
        The flag `_main_thread_loop_running` will be set while running
        the code inside the context manager and it will be unset
        when leaving the context manager.
        """
        if self._main_thread_loop_running:
            raise RuntimeError("Main thread loop is already running.")
        self._main_thread_loop_running = True
        try:
            yield self._main_thread_loop_running
        finally:
            self._main_thread_loop_running = False


def main_thread_guard(function) -> Callable:
    """A decorator to mark a function that should be run in the main thread.
    If such a function is called from a background thread, it will not be
    executed directly, but rather it will be registered to be executed
    from the main thread.
    """

    # FIXME[concept]: there is no consistent return type for this function.
    # When registered for running in the main thread, we can not return
    # any useful value.  Hence one should only decorate functions without
    # return value!
    def wrapper(self, *args, **kwargs) -> Any:
        if current_thread() is main_thread():
            # we are in the main thread: regularly run the the function and
            # return its result.
            return function(self, *args, **kwargs)

        # we are in a background thread: register the function to be
        # executed in the main thread
        return self.add_main_thread_activity(function, (self,) + args, kwargs)

    return wrapper
