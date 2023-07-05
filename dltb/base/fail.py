"""
.. moduleauthor:: Ulf Krumnack

.. module:: dltb.base.fail

This module contains a base definition for :py:class:`Failable` class.
"""

# standard imports
from typing import Callable
from contextlib import contextmanager
import logging

# toolbox imports
from .observer import Observable
from ..util.error import handle_exception
# from ..util.debug import debug_object as object


class Failable:
    """A base class for objects that are :py:class:`Failable`. Those are
    classes which can invoke methods that may fail.  The
    :py:class:`Failable` provides a mechanism to set instances of such
    classes into a failed state.  It is designed in a way to integrate
    with the overal framework of the Deep Learning Toolbox, allowing
    for asynchronous execution, etc.

    Classes implementing methods that are prone to fail should
    subclass this class and then enclose critical code pieces into an
    `with self.falure_manager()` block.
    """
    def __init__(self, *args, **kwargs):
        """Create a new :py:class:`Failable` object.  Initially, the object
        will be in a sane state (no failure occured).
        """
        super().__init__(*args, **kwargs)
        self._failure = None

    @property
    def failed(self) -> bool:
        """A boolean property indicating if the object is in a failed state.
        """
        return self._failure is not None

    @property
    def failure_description(self) -> str:
        """A textual description of the failure.
        """
        return str(self._failure) if self._failure else ''

    @property
    def failure_exception(self) -> BaseException:
        """The actual exception that caused the failure.
        """
        return self._failure

    def failure(self, exception: BaseException) -> None:
        """Set the object into a failed state.

        Arguments
        ---------
        exception: BaseException
            The exception responsible for the failed state.
        """
        self._failure = exception

    def clean_failure(self) -> None:
        """Reset the object into a fine (non-failed) state.
        """
        self._failure = None

    @contextmanager
    def failure_manager(self, cleanup: Callable[[], None] = None,
                        logger: logging.Logger = None,
                        message: str = 'failure',
                        catch: bool = False) -> None:
        """Provide a manager for contexts in which failures may arise.
        If any failure is detected, that is, if an exception is raised,
        the object will be set in a failed state.

        Arguments
        ---------
        cleanup: Callable
            A callback to be invoked in case of a failure that can
            reset the object into a well defined state (the object
            will nevertheless stay in a failed state, but inconsistent
            configurations caused by the failure may be fixed).
        catch: bool
            A flag indicating if a failure should be catched
            (and reported via the error handling mechanisms provided
            by util.error) or if it should be reraised allowing the
            caller to react.
        """
        try:
            yield None
        except Exception as exception:  # pylint: disable=broad-except
            self.failure(exception)
            if logger:
                logger.error((message or 'failure') + ': ' + str(exception))
            if cleanup is not None:
                cleanup()
            if catch:
                # FIXME[concept]: this will output the exception
                # (including stack trace) on the terminal. Maybe useful
                # for debugging, but not necessary in general (as we store
                # the exception in the object and also provide logging)
                handle_exception(exception)
            else:
                raise exception


class FailableObservable(Observable, Failable, changes={'failure'}):
    """The :py:class:`FailableObservable` class is an extension of
    :py:class:`Failable`, that can notify observers in case of a
    failure.
    """

    def failure(self, exception: BaseException) -> None:
        """Set the object into a failed state.

        Arguments
        ---------
        exception: BaseException
            The exception responsible for the failed state.
        """
        super().failure(exception)
        self.change('failure', 'state_changed')

    def clean_failure(self) -> None:
        """Reset the object into a fine (non-failed) state.
        """
        super().clean_failure()
        self.change('failure', 'state_changed')
