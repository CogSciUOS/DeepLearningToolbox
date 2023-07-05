"""Basic definitions for the image module.
"""

# standard imports
from typing import Union, TypeVar, Callable, Optional
from pathlib import Path
import logging

# thirdparty imports
import numpy as np

# logging
LOG = logging.getLogger(__name__)


# Imagelike is intended to be everything that can be used as
# an image.
#
# np.ndarray:
#    The raw image data
# str:
#    A URL.
Imagelike = Union[np.ndarray, str, Path]

ImagelikeT = TypeVar('ImagelikeT', bound=Imagelike)

# A function mappling an Imagelike to another Imagelike
ImageConverter = Callable[[ImagelikeT, Optional[bool]], ImagelikeT]
