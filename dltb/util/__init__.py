"""Several utility functions.
"""

# standard imports
from typing import Any
import os
import pickle
import logging

# third party imports
try:
    from appdirs import AppDirs
    APPNAME = "deepvis"  # FIXME[hack]: not the right place to define here!
    APPAUTHOR = "krumnack"
    _appdirs = AppDirs(APPNAME, APPAUTHOR)
except ImportError:
    _appdirs = None

# logging
LOG = logging.getLogger(__name__)


def cache_path(cache: str = None) -> str:
    """Get th path the cache directory or a cache file.
    """
    cache_dir = 'cache' if _appdirs is None else _appdirs.user_cache_dir
    return cache_dir if cache is None else os.path.join(cache_dir, cache)


def read_cache(cache: str) -> Any:
    """Load data from a cache file.

    Arguments
    ---------
    cache: str
        The chache filename to be used. If not an absolute path,
        the method place it relative to the default cache directory
        of this application.

    Result
    ------
    data: Any
        The data read from the file.
    """
    if cache is None:
        return None

    cache_filename = cache_path(cache)
    LOG.info("Trying to load cache file '%s'", cache_filename)
    if not os.path.isfile(cache_filename):
        return None
    return pickle.load(open(cache_filename, 'rb'))

def write_cache(cache: str, data: Any) -> None:
    """Write data into a cache file.

    Arguments
    ---------
    cache: str
        The chache filename to be used. If not an absolute path,
        the method place it relative to the default cache directory
        of this application.
    data: Any
        The data to be written to the cache file.
    """
    if cache is None:
        return  # we can not determine a cache file

    cache_directory = cache_path()
    cache_filename = cache_path(cache)
    LOG.info("Writing filenames to %s", cache_filename)
    os.makedirs(cache_directory, exist_ok=True)
    pickle.dump(data, open(cache_filename, 'wb'))
