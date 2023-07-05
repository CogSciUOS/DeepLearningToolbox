"""Types relevant in the context of images.
"""

# standard imports
from typing import Union, List, Tuple, Optional
from enum import Enum
from collections import namedtuple

# third-party imports
import numpy as np


Sizelike = Union[Tuple[int, int], List[int], str]


class Size(namedtuple('Size', ['width', 'height'])):
    """Representation of an image size in pixel.
    Attributs `width` and `height` describe the image size.

    Size can be instantiated from strings, or integers or pairs of
    integers.  In case a pair is given, the order is `(width,
    height)`, which is the standad mathematical way.  Note that this
    is the oposite of the array shapes used in numpy, which is (rows,
    columns).
    """

    def __new__(cls, size, *args):
        """Allow to instantiate size from any `Sizeable` objects and
        also from a pair of arguments.
        """
        if isinstance(size, Size):
            return size

        if args:
            return super().__new__(cls, size, *args)

        if isinstance(size, str):
            separator = next((sep for sep in size if sep in ",x"), None)
            size = ((int(number) for number in size.split(separator))
                    if separator else int(size))
        elif isinstance(size, float):
            size = int(size)

        if isinstance(size, int):
            return super().__new__(cls, size, size)

        return super().__new__(cls, *size)

    def __eq__(self, size: Sizelike) -> bool:
        """Allow to compare `Size` to any `Sizeable` objects.
        """
        return super().__eq__(Size(size))


Sizelike = Union[Sizelike, Size]


Color = Tuple[int, int, int]

class Colorspace(Enum):
    """Enumeration of potential colorspace for representing images.
    """
    RGB = 1
    BGR = 2
    HSV = 3


class Format:
    # pylint: disable=too-few-public-methods
    """Data structure for representing image format. This includes
    the datatype of the image, colorspace, and min and max values.
    It may also include an image size.
    """
    dtype = np.uint8
    colorspace = Colorspace.RGB
    _min_value = None
    _max_value = None

    size: Optional[Size] = None

    @property
    def min_value(self) -> Union[int, float]:
        """The minimal possible pixel value in an image.
        """
        if self._min_value is not None:
            return self._min_value
        if issubclass(self.dtype, (int, np.integer)):
            return 0
        return 0.0

    @property
    def max_value(self) -> Union[int, float]:
        """The minimal possible pixel value in an image.
        """
        if self._max_value is not None:
            return self._max_value
        if issubclass(self.dtype, (int, np.integer)):
            return 255
        return 1.0
