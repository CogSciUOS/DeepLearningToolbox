
from base.observer import BusyObservable

def run(function):

    def wrapper(self, *args, **kwargs):
        return self.run(function, *args, **kwargs)

    return wrapper

class Controller(BusyObservable):
    """Base class for all kinds of controllers.

    Attributes
    ----------
    _datasource : DataSource
        The current :py:class:`datasources.DataSource`
    """
    def __init__(self, runner: 'AsyncRunner'=None) -> None:
        super().__init__()
        self._runner = runner

    def set_runner(self, runner: 'AsyncRunner') -> None:
        """Set that the :py:class:`AsyncRunner` for this Controller.

        Parameters
        ----------
        runner: 'AsyncRunner'
            The runner to be used for asynchronous execution. If `None`,
            no asynchronous operations will be performed by this
            BaseController.
        """
        self._runner = runner

    def run(self, function, *args, **kwargs):
        self.busy = True
        if self._runner is None:
            return self._run(function, *args, **kwargs)
        else:
            self._runner.runTask(self._run, function, *args, **kwargs)

    def _run(self, function, *args, **kwargs):
        callback = kwargs.pop('async_callback', None)
        result = function(self, *args, **kwargs)
        self.busy = False
        if callback is None:
            return result
        elif isinstance(result, tuple):
            callback(*result)
        else:
            callback(result)
