"""Collection of time related functions, types, and classes.

"""

from typing import Iterable, Iterator, TypeVar, Union, Optional
from time import time as now, sleep
import datetime

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


def parse_time(time_string: str) -> float:
    """Parse a timestring. The string is expected to have the format
    `'[[hh:]mm:]ss[.d+]'` or `s+[.d+]`.

    Arguments
    ---------
    time_string:
        The time string.

    Result
    ------
    seconds:
        The time in seconds.
    """
    parts = time_string.split(':')
    if len(parts) > 3:
        raise ValueError(f"Invalid time string: '{time_string}'")

    seconds = float(parts[-1])
    if len(parts) > 1:
        seconds *= int(parts[-2])
        if len(parts) > 2:
            seconds *= int(parts[-3])
    return seconds


ElementType = TypeVar('ElementType')


def pacemaker(iterable: Iterable[ElementType], seconds: float,
              absolute: bool = True) -> Iterator[ElementType]:
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
    last_time = now() - seconds
    for element in iterable:  # first source of delay: the producer
        current_time = now()
        if last_time + seconds > current_time:
            sleep(last_time + seconds - current_time)
        yield element  # second source of delay: the producer
        last_time = (last_time + seconds) if absolute else now()


Time = float
Timelike = Union[Time, str]


class IndexTiming:
    """Translating between index values (integer) and time in seconds
    (float).

    Just a simple version - will not work for variable samplerates ...
    """

    def __init__(self, samplerate: float) -> None:
        self._samplerate = samplerate

    def index_to_time(self, index: int) -> Time:
        """Translate the given index into a point in time.
        """
        return index / self._samplerate

    def time_to_index(self, time: Timelike) -> int:
        """Translate a given point in time into an index.
        """
        if not isinstance(time, Time):
            time = parse_time(time)
        return round(time * self._samplerate)

    @property
    def samplerate(self) -> float:
        """The samplerate applied by this `IndexTiming`.
        """
        return self._samplerate
