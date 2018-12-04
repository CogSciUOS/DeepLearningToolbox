"""
.. moduleauthor:: Rasmus Diederichsen

.. module:: util

This module collects miscellaneous utilities.
"""


class ArgumentError(ValueError):
    """Invalid argument exception"""
    pass


def grayscaleNormalized(array):
    """Convert a float array to 8bit grayscale

    Parameters
    ----------
    array: np.ndarray
        Array of 2/3 dimensions and numeric dtype.
        In case of 3 dimensions, the image set is normalized globally.

    Returns
    -------
    np.ndarray
        Array mapped to [0,255]

    """
    import numpy as np

    # normalization (values should be between 0 and 1)
    min_value = array.min()
    max_value = array.max()
    div = max(max_value - min_value, 1)
    return (((array - min_value) / div) * 255).astype(np.uint8)


class Identifiable:
    _id: str = None
    _counter: int = 0

    def __init__(self, id=None):
        if id is None:
            self._ensure_id()
        else:
            self._id = id

    def _ensure_id(self):
        if self._id is None:
            Identifiable._counter += 1
            self._id = self.__class__.__name__ + str(Identifiable._counter)
        return self._id

    def get_id(self):
        return self._ensure_id()

    def __hash__(self):
        return hash(self._ensure_id())

    def __eq__(self, other):
        if isinstance(other, Identifiable):
            return self._ensure_id() == other._ensure_id()
        return False


from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=4,
                               thread_name_prefix='async')


runner = None

def run_async(function, *args, **kwargs):
    _executor.submit(function, *args, **kwargs)


def async(function):
    """A decorator to enforce asyncronous execution. 
    """
    def wrapper(*args, **kwargs):
        runner.runTask(function, *args, **kwargs)
        #run_async(function, *args, **kwargs)
    return wrapper


import resource
        
           
timer = True
import time, threading
def foo():

    # The Python docs aren't clear on what the units are exactly,
    # but the Mac OS X man page for getrusage(2) describes the
    # units as bytes. The Linux man page isn't clear, but it seems
    # to be equivalent to the information from /proc/self/status,
    # which is in kilobytes.
    rusage = resource.getrusage(resource.RUSAGE_SELF)

    nvidia_smi_q = subprocess.check_output(["nvidia-smi", "-q"],
                                           universal_newlines=True)
    match = re.search('GPU Current Temp *: ([^ ]*) C', nvidia_smi_q)
    temperature = match.group(1) if match else '?'
    
    print(f"{time.ctime()}, Memory: " +
          "Shared={:,} kiB, ".format(rusage.ru_ixrss) +
          "Unshared={:,} kiB, ".format(rusage.ru_idrss) +
          "Peak={:,} kiB, ".format(rusage.ru_maxrss) +
          f"GPU temperature={temperature}C")
    if timer:
        threading.Timer(1, foo).start()

def start_timer():
    foo()

def stop_timer():
    global timer
    timer = False
