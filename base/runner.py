class Runner:
    """Base class for runner objects which must be provided for each
    controller/user interface.

    """

    def __init__(self) -> None:
        pass

    def runTask(self, function, *args, **kwargs) -> None:
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
        function(*args, **kwargs)


from concurrent.futures import ThreadPoolExecutor, Future

class AsyncRunner(Runner):
    """Base class for asynchronous runner objects.

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
    _executor: ThreadPoolExecutor
        Singular reusable thread to run computations in.

    """
    _executor: ThreadPoolExecutor = None

    def __init__(self) -> None:
        super().__init__()
        self._submitted = 0
        self._completed = 0
        self._executor = ThreadPoolExecutor(max_workers=4,
                                            thread_name_prefix='runner')

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
        # FIXME[hack]: shutdown does not work smoothly yet -
        # it seems the executor is deleted too early ..
        if self._executor is not None:
            future = self._executor.submit(fn, *args, **kwargs)
            future.add_done_callback(self.onCompletion)
            self._submitted += 1
        else:
            fn(*args, **kwargs)

    def onCompletion(self, result):
        # FIXME[old]: explain what is done now!
        """Callback exectuted on completion of a running task. This
        method must be implemented by subclasses and - together with
        any necessary initialisations in :py:meth:`__init__` - should
        lead to calling :py:meth:`model.Model.notifyObservers` on the
        main thread.

        Parameters
        ----------
        result: object or Observable.Change
            Object returned by the asynchronously run method.
        """
        raise NotImplementedError('This abstract base class '
                                  'should not be used directly.')

    @property
    def active_workers(self):
        """Determine how many threads are currently running.
        """
        # FIXME[bug]: this does not work!
        # self._executor._threads is a set of Threads that have been
        # created. This number will grow up to max_workers.
        # Threads will not be terminated on completion. They rather
        # stay alive and wait for new functions to be executed.
        #return len(self._executor._threads)
        return self._submitted - self._completed

    @property
    def max_workers(self):
        """Determine the maximal number of threads that can run.
        """
        return self._executor._max_workers


    def quit(self):
        """Quit this runner.
        """
        if self._executor is not None:
            print("AsyncRunner: Shutting down the executor ...")
            self._executor.shutdown(wait=True)
            print("AsyncRunner: ... executor finished.")
            self._executor = None


#
# FIXME[experimental]
#

# - probably a better way to go is to implement an additional
#   class Parallelizable
#   - allow additional parameter 'parallel: bool' to the prepare
#     method, which then prepares the object in another process
#   - automatically defer all (or some) methods to that process
#   - unprepare will stop the other process

# - probably even better would be to have a ProcessRunner

import multiprocessing

class ProcessRunner(Runner):
    """

    The main difference between as ProcessRunner and a ThreadRunner
    is that in the ProcessRunner all computed data live in another
    process and can not be directly accessed from the main process,
    but only via some inter process communication mechanism.
    """

    # FIXME[todo]: implementation
    pass

class ProcessObservable: # (Observable):

    def __init__(name: str, prepare: bool=True):
        self._name = name
        self._prepare = prepare

    def prepare(self):
        self._queue = multiprocessing.Queue()
        self._task = multiprocessing.Event()
        self._ready = multiprocessing.Event()
        self._process = multiprocessing.Process(target=self._process_loop)
        self._process.start()

    def unprepare(self):
        self._process.terminate()


    def _process_loop(self, detector=None):
        if detector is None:
            self._detector = Detector.create(self._name, self._prepare)
        while True:
            self._task.wait()
            method, args, kwargs = self._queue.get()
            self._task.clear()
            print(method, args, kwargs)
            # func = getattr(self, method)
            # result = func(*args, **kwargs)
            result = "the result"
            self._ready.set()
            self._queue.put(result)
            

    @property
    def busy(self):
        return not self._ready.is_set()
            
    def detect(self, data, **kwargs):
        if self.busy:
            raise RuntimeException("Object is not ready.")
        ## we are beginning a new task - hence we are not ready for other tasks
        self._ready.clear()

        ## signal the other process that a new task will start
        self._task.set()

        ## queue the task information
        self._queue.put(['detect', (data, ), kwargs])

        ## make this an synchronous call - wait for the process to finish
        self._ready.wait()

        detections = self._queue.get()
        # FIXME[todo]: receive the result from the queue
        return detections
