"""The `base` module of the Deep Learning Toolbox provides several
base classes for use in other parts of the toolbox.

Code in the `base` module should be as self-contained as possible.  It
should not rely on code from other parts of the Deep Learning Toolbox.
and it should not require any thirdparty modules (except `numpy`).

                      pylint  mypy  tests  imports   fixmes
  busy.py             9.39    5     -      1-2----   1
  data.py             10.0    4     1      1---3--   0
  fail.py             9.23    3     -      1-2----   1
  hardware.py         5.57    4     -      ------4   0
  image.py            9.66    5     -      1-2----   12
  implementation.py   9.93    0     10     -------   1
  __init__.py         10.0    0     -      1------   0
  install.py          8.15    1     -      1------   5
  meta.py             10.0    2     -      1---3--   0
  observer.py         9.43    19    -      --2----   6
  prepare.py          8.86    3     7      1-2----   5
  register.py         10.0    42    2      1-2----   0
  resource.py         5.58    26    -      1------   5
  sound.py            9.93    33    -      1-2-3--   2
  state.py            10.0    2     -      1------   0
  store.py            10.0    15    -      1------   0
  types.py            9.77    0     -      -------   1
  video.py            6.56    27    -      1-2-3--   12


Toolbox `base` imports (0=none, 1=base, 2=toolbox, 3=numpy, 4=other thirdparty)
  busy.py: 1: fail; 2: util.error
  data.py: 1: observer;  3: numpy
  fail.py: 1: observer;  2: util.error, util.debug
  hardware.py: 4: py3nvml
  image.py: 1: observer, data, implementation;
            2: util.error, util.image;
            3: numpy
  implementation.py: 0
  __init__.py: 1: types, observer, prepare
  install.py: 1: busy, fail
  meta.py: 1: image; 3: numpy
  observer.py: 3: util.error, util.debug
  prepare.py: 1: busy; 2: config
  register.py: 1: observer, busy, prepare, fail; 2: util.debug, thirdparty
  resource.py: 1: busy
  sound.py: 1: ., observer, data; 2: thirdparty; 3: numpy
  state.py: 1: observer
  store.py: 1: prepare
  types.py: 0
  video.py: 1: image, prepare; 2: util.time, thirdparty; 3: numpy
"""

# standard imports
# concurrent.futures is new in Python 3.2.
from concurrent.futures import ThreadPoolExecutor
import threading

# Toolbox imports
from .types import Extendable
from .observer import Observable
from .prepare import Preparable


_executor: ThreadPoolExecutor = \
    ThreadPoolExecutor(max_workers=4, thread_name_prefix='runner')


# pylint: disable=redefined-outer-name
def get_default_run(run: bool = None) -> bool:
    """Determine if a new thread should be spawned for executing some
    task.  The current implementation checks whether the we are
    already running a custom thread (in which case no new thread will
    be spawned) or if we are running in the main event loop of a
    graphical user interface (in which case a new thread should be
    spawend).

    Arguments
    ---------
    run:

    """
    # If a run parameter is specified, we will use that value
    if run is not None:
        return run

    # We usually do not want to spawn a new thread when already
    # executing some thread. Our threads' names start with 'runner'.
    current_thread = threading.currentThread()
    if current_thread.getName().startswith('runner'):
        return False

    # Finally, if the current thread is the event loop of some
    # graphical user interface, we choose to run in the background.
    # We assume that GUI threads (and only GUI threads) have an
    # attribute 'GUI_event_loop' that is set to True.
    return getattr(current_thread, 'GUI_event_loop', False)


def run(function):
    """A decorator for functions which may be run in a separate
    `Thread`. The decorator will consult the function
    `get_default_run` to determine if a new `Thread` should
    be started.

    The decorator also adds an additional keyword argument `run` to
    the function, allowing to specify if the function should be run in
    a new thread.  That argument is passed to the function
    :py:func:`get_default_run` for the final decision.  The default
    value is `False`, meaning that no new Thread will be spawned.
    """

    # pylint: disable=redefined-outer-name
    def wrapper(self, *args, run: bool = False, **kwargs):
        if get_default_run(run):
            _executor.submit(function, self, *args, **kwargs)
        else:
            function(self, *args, **kwargs)

    return wrapper
