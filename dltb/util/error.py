"""
.. moduleauthor:: Ulf Krumnack

.. module:: util

This module collects miscellaneous error handling functions.

The ultimate goal is that no :py:class:`Exception` goes unnoticed.
This is an ambitious plan, especially in a multithreaded or even
multiprocessing system with event driven GUI. Several mechanisms
are provided to support this goal:

* The @protect decorator: this is intended to protect top level
  function, like signal slots in a GUI that are directly invoked by
  the main event loop.  This decorator simply catches all exceptions
  and reports them via `handle_exception`. The exception is not
  raised again - meaning that the system can be in an inconsistent
  state.

* Methods that can raise :py:class:`Exception`s should document this
  in their doc string.

When catching an :py:class:`Exception`, it should be reported to
the user in a suitable fashion. For this purpose, exception handlers
can be employed: an exception handler is a function that takes an
exception as argument and reports it to the user. This module
provides a global exception handler `handle_exeception`,
to which other exception handlers can register.

"""

import sys
import traceback


def print_exception(exception: BaseException) -> None:
    """Print the given exception including traceback to stdout.
    """
    print("-" * 79)
    print(f"\nUnhandled exception ({type(exception).__name__}): {exception}")
    # FIXME[bug]: sys.stderr may by set to os.devnull ...
    # traceback.print_tb(exception.__traceback__, file=sys.stderr)
    traceback.print_tb(exception.__traceback__, file=sys.stdout)
    print("-" * 79)


_exception_handler = print_exception


def handle_exception(exception: BaseException) -> None:
    """Handle the given exception.  This will delegate the exception
    to all registered exception handlers.  The default exception handler
    will simply print the exception to the standard error stream.
    """
    if _exception_handler is not None:
        _exception_handler(exception)


def set_exception_handler(handler) -> None:
    """Set an alternative exception handler.

    Arguments
    ---------
    handler:
        An exception handler. This can be any function that takes one
        :py:class:`BaseException` as argument.
    """
    global _exception_handler
    _exception_handler = handler


def protect(function):
    """A decorator for top-level functions to protect the program from
    crashing due to unhandled exceptions.

    Notes:
    (1) Only use @protect to protect for event handlers, signal slots,
        and similar functions in user interfaces!
    (2) Never call a `@protect`ed function.

    The @protect decorator will handle the exception by calling the
    generic exception handler (`util.error.handle_exception()`) and
    not re-raise it again. This may be fine for functions called
    directly by the main event loop of some user interface (like event
    handlers) or that are invoked by some unidirectional signaling
    mechanism (like signal slots), but it may be inadequate in most
    other situations, where the caller should be notified that something
    went wrong so she can react to the exception.

    If you call a `@protect`ed function, something is wrong with your
    design. You have no chance to detect any Exception raised within
    this function. If you have a function that can be called top-level
    (e.g. from the event loop of the user interface) and from your
    code, split it into two: a slim `@protect`ed wrapper and the
    actual function.

    """
    def closure(self, *args, **kwargs):
        # pylint: disable=broad-except
        try:
            return function(self, *args, **kwargs)
        except KeyboardInterrupt:
            # FIXME[hack]: in some caes KeyboardInterrupt may be
            # processed by some outer loop.
            # It would be good to have a way to indicate that some
            # exception should not be protected.
            # FIXME[bug]: when used with Qt event-handlers, this
            # may cause core dumps
            raise
        except BaseException as exception:
            handle_exception(exception)
            return None
    return closure
