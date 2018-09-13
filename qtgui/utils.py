from PyQt5.QtCore import QObject, pyqtSignal

from controller import AsyncRunner
from observer import Observable, BaseChange

class QtAsyncRunner(AsyncRunner, QObject):
    """:py:class:`AsyncRunner` subclass which knows how to update Qt widgets.

    Attributes
    ----------
    _completion_signal: pyqtSignal
        Signal emitted once the computation is done.
        The signal is connected to :py:meth:`onCompletion` which will be run
        on the main thread by the Qt magic.
    """

    # signals must be declared outside the constructor, for some weird reason
    _completion_signal = pyqtSignal(object)

    def __init__(self, observable: Observable):
        """Connect a signal to :py:meth:`Observable.notifyObservers`."""
        super().__init__(observable)
        self._completion_signal.connect(observable.notifyObservers)

    def onCompletion(self, future):
        """Emit the completion signal to have
        :py:meth:`Observable.notifyObservers` called.
        """
        result = future.result()
        if isinstance(result, BaseChange):
            self._completion_signal.emit(result)
