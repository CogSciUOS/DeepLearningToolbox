"""General types an utility functions used in the deep learning toolbox.
"""

from typing import Union
from pathlib import Path

Pathlike = Union[str, Path]

def as_path(pathlike: Pathlike) -> Path:
    """Create a `Path` from a `Pathlike` object.  If `path` is an
    object itself, it will be returned, otherwise a new `Path` is
    constructed.

    Arguments
    ---------
    pathlike:
        The `Pathlike` object from which the path is to be created.

    Result
    ------
    path
        The resulting `Path`.
    """
    return pathlike if isinstance(pathlike, Path) else Path(pathlike)
