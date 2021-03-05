"""
.. moduleauthor:: Ulf Krumnack

.. module:: util.logging


This module provides classes that can be plugged into the standard python
logging mechanism provided by the logging module.  Logging is realized
by the interplay of Loggers, Filters, and Handlers:

* Loggers are assigned to modules or otherwise connected parts of the
  program, identified by a common name. These parts put all messages
  to their associated Logger.

* Handlers are contacted by loggers in order to handle messages.
  One Logger can be connected to multiple Handlers simultanously and
  one Handler can also receive messages from multiple Loggers at
  the time.

* Filters can be put on Handlers to restrict what messages should
  be used by the Handler and which can be dropped.


This module only require the standard python logging module.
"""

# standard imports
import logging

# toolbox imports
from .terminal import Terminal


class RecorderHandler(logging.Handler):
    """A :py:class:`RecorderHandler` stores
    :py:class:`logging.LogRecord`s, allowing to replay the log
    history. Similar to a :py:class:`logging.BufferingHandler`, but
    allowing multiple replays.

    A :py:class:`RecorderHandler` can be, like any other
    :py:class:`logging.Handler`, added to a
    :py:class:`logging.Logger`.  It will then receive all new
    :py:class:`logging.LogRecord`s from that
    :py:class:`logging.Logger` via the :py:meth:`emit()` method.

    To record all :py:class:`logging.LogRecord`s produced by a
    program, the :py:class:`RecorderHandler` has to be attached to the
    :py:class:`logging.RootLogger`.  This can be achieved as follows:

        logRecorder = util.RecorderHandler()
        root_logger = logging.getLogger()
        root_logger.addHandler(logRecorder)

    However, the :py:class:`RecorderHandler` can also be used to only
    collect :py:class:`logging.LogRecord`s from one or multiple specific
    :py:class:`logging.Logger`s:

        logRecorder = util.RecorderHandler()
        logging.getLogger('numpy').addHandler(logRecorder)
        logging.getLogger('matplotlib').addHandler(logRecorder)

    This will record all :py:class:`logging.LogRecord`s emitted
    by numpy and matplotlib.

    Attributes
    ----------

    _records: list
        The list of records recorded by the :py:class:`RecorderHandler`.

    """

    # Note: originally we had derived this class from list instead of
    # having the list attribute `_records`. However, starting with
    # Python 3.7 this stopped working, as the `logging` module now
    # collects all `logging.Handler`s in a set.  Sets only allow
    # hashable elements, but Python lists are not hashable (as their
    # content can change), resulting in a "TypeError: unhashable type:
    # 'RecorderHandler'" upon initialization.

    def __init__(self):
        super().__init__()
        self._records = []

    def emit(self, record: logging.LogRecord) -> None:
        """The emit() method is invoked as a reaction to the invocation of
        the handle() by a Logger. It will only be invoked, if no Filter
        blocks the :py:class:`logging.LogRecord`. This method simply
        store the :py:class:`logging.LogRecord` in this
        :py:class:`RecorderHandler`.
        """
        self._records.append(record)

    def replay(self, handler: logging.Handler) -> None:
        """Replay all :py:class:`logging.LogRecord`s stored in this
        :py:class:`RecorderHandler` for the given
        :py:class:`logging.Handler`.

        Arguments
        ---------
        handler: logging.Handler
            The log handler to which the messages should be replayed.
        """
        for record in self._records:
            handler.handle(record)

    def __len__(self) -> int:
        """The number of :py:class:`logging.LogRecord`s stored in this
        :py:class:`RecorderHandler`.

        """
        return len(self._records)


class TerminalFormatter(logging.Formatter):
    """
    """

    def __init__(self, *args, terminal: Terminal = None, **kwargs) -> None:
        """
        """
        super().__init__(*args, **kwargs)
        self._terminal = terminal or Terminal()

    def format(self, record: logging.LogRecord) -> None:
        """
        """
        message = super().format(record)
        if record.levelno <= logging.DEBUG:
            color = Terminal.Bformat.BLUE
        elif record.levelno <= logging.INFO:
            color = Terminal.Bformat.GREEN
        elif record.levelno <= logging.WARNING:
            color = Terminal.Bformat.YELLOW
        else:
            color = Terminal.Bformat.RED
        return self._terminal.form(message, color)
