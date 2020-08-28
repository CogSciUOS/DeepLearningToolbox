# concurrent.futures is new in Python 3.2.
from concurrent.futures import ThreadPoolExecutor, Future
import threading

_executor: ThreadPoolExecutor = \
    ThreadPoolExecutor(max_workers=4, thread_name_prefix='runner')


def get_default_run(run: bool = None) -> bool:
    # If a run paramter is specified, we will use that value
    if run is not None:
        return run

    # We usually do not want to spawn a new thread when
    # already executing some thread.
    current_thread = threading.currentThread()
    if current_thread.getName().startswith('runner'):
        return False

    # Finally, if the current thread is the event loop of some
    # graphical user interface, we choose to run in the background
    return getattr(current_thread, 'GUI_event_loop', False)


def run(function):

    def wrapper(self, *args, run: bool = False, **kwargs):
        if get_default_run(run):
            _executor.submit(function, *args, **kwargs)
        else:
            function(*args, **kwargs)

    return wrapper
