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


# concurrent.futures is new in Python 3.2.
from concurrent.futures import ThreadPoolExecutor, Future

class AsyncRunner(Runner):
    """Base class for asynchronous runner objects.

    .. note:: *Why is this necessary?*

    When working with Qt for instance, it is not possible to update
    the gui from a non-main thread.  So we cannot simply run a thread
    to do expensive computations and have it call
    :py:meth:`model.Model.notify_observers` since that call would
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
        lead to calling :py:meth:`model.Model.notify_observers` on the
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

            # shutdown: Signal the executor that it should free any
            # resources that it is using when the currently pending
            # futures are done executing.
            #
            # If wait is True then this method will not return until
            # all the pending futures are done executing and the
            # resources associated with the executor have been freed.
            #
            # If wait is False then this method will return
            # immediately and the resources associated with the
            # executor will be freed when all pending futures are done
            # executing.
            #
            # But: Regardless of the value of wait, the entire Python
            # program will not exit until all pending futures are done
            # executing.
            #
            # FIXME[problem]: this can be a problem, if some runner is
            # waiting for another Thread to set some Event, but that
            # other Thread will never set this Event (e.g., because it
            # has crashed). We need some strategy to deal with such
            # cases (otherwise the progam may block and not exit
            # gracefully).
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
    # FIXME[question]: what is this class supposed to do. Some
    # documentation would be good
    """

    Arguments
    ---------
    name:
        The name of the detector class to use.
    """

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
            self._detector = Detector(self._name)
            if self._prepare:
                self._detector.prepare()
        while True:
            self._task.wait()
            method, args, kwargs = self._queue.get()
            self._task.clear()
            print("ProcessObservable:", method, args, kwargs)
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


    
# FIXME[new]: the following code is not used really used now - but
# may be interesting in the context of multiprocessing: Start a
# loop in some thread/process. This loop waits for Events, which
# will trigger the execution of certain operations in the
# thread/process. The code as it is worked as part of the
# DetectorMTCNN class, but needs some adaptation to be usable in a
# more general context.
#
# FIXME[old]: originally designed to run the MTCNN detector in its
# own Thread. This was motivated to some strange
# Keras/Tensorflow/MTCNN behaviour, which was actually due to
# neglecting TensorFlow's Graph and Sessions, which has now been
# repaired.
class NEWBackgroundRunner:

    def _threadLoop(self):
        """Run a processing loop.
        
        This method is intended to be run in a separate thread. It
        initializes the MTCNN detector and then waits for images that
        it can process with that detector.  Availability of an image
        is signaled by the `_thread_new_task` :py:class:`Event` and
        completion will be signaled by setting the `_thread_finished`
        :py:class:`Event`.

        """
        print(f"NEWBackgroundRunner[{threading.currentThread().getName()}]: _threadLoop: "
              "preparing the MTCNN detector.")
        try:
            self._prepare2()
        except BaseException as error:
            print(f"NEWBackgroundRunner[{threading.currentThread().getName()}]: _threadLoop: "
                  f"preparation failed ({error}).")
            handle_exception(error)
                
        print(f"NEWBackgroundRunner[{threading.currentThread().getName()}]: _threadLoop: "
              f" ... preparation finished ({self.prepared}).")

        self._thread_finished.set()

        print(f"NEWBackgroundRunner[{threading.currentThread().getName()}]: _threadLoop: "
              "Starting the processing loop ...")
        while True:
            self._thread_new_task.wait()
            self._thread_new_task.clear()
            if self._thread_stop:
                print(f"NEWBackgroundRunner[{threading.currentThread().getName()}]: "
                      "_threadLoop: received a stop signal.")
                break
            print(f"NEWBackgroundRunner[{threading.currentThread().getName()}]: _threadLoop: "
                  "detecting faces with the MTCNN detector "
                  f"{self._thread_image.shape}")
            self._thread_metadata = self._detect2(self._thread_image)
            self._thread_finished.set()
        print(f"NEWBackgroundRunner[{threading.currentThread().getName()}]: _threadLoop: "
              " ... processing loop finished.")

    def _prepare2(self) -> None:
        """Create a Thread and start a processing loop
        (:py:meth:`_threadLoop`). Initialization and prediction will be
        done by that loop.
        """
        print(f"NEWBackgroundRunner[{threading.currentThread().getName()}]: _prepare: "
              "preparing mtcnn ...")
        self._thread_new_task = threading.Event()
        self._thread_finished = threading.Event()
        self._thread = threading.Thread(target=self._threadLoop,
                                        name="MTCNN-Thread")
        self._thread_stop = False
        self._thread.start()
        print(f"NEWBackgroundRunner[{threading.currentThread().getName()}]: _prepare: "
              "... waiting for background task to finish ....")
        self._thread_finished.wait()
        self._thread_finished.clear()
        print(f"NEWBackgroundRunner[{threading.currentThread().getName()}]: _prepare: "
              "... prepared [{self.prepared}].")

    def _unprepare2(self) -> bool:
        """The DetectorMTCNN is prepared, once the model data
        have been loaded.
        """
        self._thread_stop = True
        self._thread_new_task.set()
        self._thread.join()
        self._thread_new_task = None
        self._thread_finished = None
        self._thread = None

    def _detect2(self, image): #  -> Metadata
        """Running detect in a separate Thread.
        """
        self._thread_image = image
        self._thread_new_task.set()
        self._thread_finished.wait()
        self._thread_finished.clear()
        return self._thread_metadata

#
# FIXME[experiment]: a context manager that starts a background thread
#

# The following is not exactly what I am looking for: it starts a
# background thread and stops that Thread once the context is exited
#
# with Sleeper(sleep=2.0) as sleeper:
#     time.sleep(5)
#
# BTW - this will crash ipython when sleep=8.0 (i.e. the background thread
# runs longer than the context)

import threading

class ThreadContext(threading.Thread):
    def __init__(self, sleep=5.0):
        super().__init__(name='Background')
        self.stop_event = threading.Event()
        self.sleep = sleep

    def run(self):
        print(f"ThreadContext: ThreadContext {threading.current_thread()} started")
        while self.sleep > 0 and not self.stop_event.is_set():
            time.sleep(1.0)
            self.sleep -= 1
        print(f"ThreadContext: Thread {threading.current_thread()} ended")

    def stop(self):
        self.stop_event.set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        self.stop()
        print('ThreadContext: Force set Thread Sleeper stop_event')

