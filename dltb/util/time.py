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
