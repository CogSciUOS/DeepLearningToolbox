from observer import Observable
from concurrent.futures import ThreadPoolExecutor, Future

class AsyncRunner(object):
    """Base class for runner objects which must be provided for each
    controller/user interface.

    .. note:: *Why is this necessary?*

    When working with Qt for instance, it is not possible to update
    the gui from a non-main thread.  So we cannot simply run a thread
    to do expensive computations and have it call
    :py:meth:`model.Model.notifyObservers` since that call would
    invoke methods on :py:class:`PyQt5.QWidget`s. We would need some
    way to call back into the main thread, but that one is busy running
    Qt's main loop so the app can respond to further user action. Qt's
    way of resolving this is with signals. But we do not want to make
    this a fixed solution since the app should be as independent of Qt
    as possible. I have thus decided that a dedicated runner must be
    provided for each kind of user interface which knows how to handle
    asynchronous updates.

    Attributes
    ----------
    _observable: Observable
        An observable whose updates should be broadcast asynchronously
    _executor: ThreadPoolExecutor
        Singular reusable thread to run computations in.

    """

    def __init__(self, observable: Observable) -> None:
        super().__init__()
        self._observable = observable
        self._executor = ThreadPoolExecutor(max_workers=1)

    def runTask(self, fn, *args, **kwargs) -> None:
        """Schedule the execution of a function.
        This is equivalent to running::

            fn(*args, **kwargs)

        asynchronously.

        Parameters
        ----------
        fn  :   function
                Function to run asynchronously
        args    :   list
                    Non-keyword args to ``fn``
        kwargs  :   dict
                    keyword args to ``fn``
        """
        future = self._executor.submit(fn, *args, **kwargs)
        future.add_done_callback(self.onCompletion)

    def onCompletion(self, result):
        """Callback exectuted on completion of a running task. This
        method must be implemented by subclasses and - together with
        any necessary initialisations in :py:meth:`__init__` - should
        lead to calling :py:meth:`model.Model.notifyObservers` on the
        main thread.

        Parameters
        ----------
        result: object or model.ModelChange
            Object returned by the asynchronously run method.
        """
        raise NotImplementedError('This abstract base class '
                                  'should not be used directly.')
