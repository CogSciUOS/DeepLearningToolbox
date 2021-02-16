from typing import Iterable, Iterator, TypeVar
import datetime
import time

def time_str(seconds: float) -> str:
    """A string representing the given times.

    Arguments
    ---------
    seconds: float
        The time in seconds.
    """
    the_time = datetime.timedelta(seconds=seconds)
    secs = int(the_time.seconds)
    hours, minutes, seconds = (secs // 3600), (secs // 60) % 60, secs % 60
    deciseconds = the_time.microseconds // 100000
    return f"{hours:02}:{minutes:02}:{seconds:02}.{deciseconds}"


T = TypeVar('T')


def pacemaker(iterable: Iterable[T], seconds: float,
              absolute: bool = True) -> Iterator[T]:
    """Provide values from an iterable in regular time steps.

    Arguments
    ---------
    iterable:
        The :py:class:`Iterable` that should be stepped through.
    seconds:
        Time in seconds between consecutive steps.
    absolute:
        A flag indicating how to react to delays in processing.
        If `True`, the pacemaker compensate for delays by
        shortening subsequent steps. If `False`, the pacemake will
        always use the last step as reference point.

    Note that delays may be caused at two places: by the
    original iterable (the producer of the sequence) and by the
    code processing the sequence (the consumer of the sequence).

    Examples
    --------

    >>> from dltb.util.time import pacemaker
    >>> from time import time, sleep

    >>> start = time()
    >>> for i in pacemaker(range(5), 2.0):
    >>>     print(f"({i}) {time()-start:.3f}s")

    >>> start = time()
    >>> for i in pacemaker(range(5), 2.0, absolute=True):
    >>>     print(f"({i}) {time()-start:.3f}s")
    >>>     sleep(1.0)

    >>> start = time()
    >>> for i in pacemaker(range(5), 2.0, absolute=False):
    >>>     print(f"({i}) {time()-start:.3f}s")
    >>>     sleep(1.0)
    """
    last_time = time.time() - seconds
    for element in iterable:  # first source of delay: the producer
        current_time = time.time()
        if last_time + seconds > current_time:
            time.sleep(last_time + seconds - current_time)
        yield element  # second source of delay: the producer
        last_time = (last_time + seconds) if absolute else time.time()
