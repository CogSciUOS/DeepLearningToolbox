"""Image utility functions.
"""

# third party imports
from typing import Union, Tuple
import numpy as np

# toolbox imports
from ..base.image import ImageReader, ImageWriter, ImageDisplay, ImageResizer

_reader: ImageReader = None
_writer: ImageWriter = None
_display: ImageDisplay = None
_resizer: ImageResizer = None


def imread(filename: str, **kwargs) -> np.ndarray:
    """Read an image from a given file. FIXME[todo]: or URL
    """
    global _reader
    if _reader is None:
        _reader = ImageReader()
    return _reader.read(filename, **kwargs)


def imwrite(filename: str, image: np.ndarray, **kwargs) -> None:
    """
    """
    global _writer
    if _writer is None:
        _writer = (_reader if isinstance(_reader, ImageWriter) else
                   ImageWriter())
    _writer.write(filename, image, **kwargs)


def imshow(image: np.ndarray, **kwargs) -> None:
    """
    """
    global _display
    if _display is None:
        _display = ImageDisplay(module='qt')  # FIXME[hack]
    _display.show(image, **kwargs)


def imresize(image: np.ndarray, size: Tuple[int, int], **kwargs) -> np.ndarray:
    """Resize the given image. The result will be a new image
    of the specified size.

    """
    global _resizer
    if _resizer is None:
        _resizer = ImageResizer()
    return _resizer.resize(image, size, **kwargs)


def imimport(img: Union[np.ndarray, str], dtype=np.uint8,
             size=None, none_size=None, **kwargs) -> np.ndarray:
    """Convert a given argument into an image, that is a numpy array.

    The import may be adapted by providing additional arguments. The
    default behavior is to return an image as numpy array of type
    uint8 (values 0-255) with RGB channels.

    Arguments
    ---------
    img:
        This function checks the type of the argument and adapts it
        according to the following rules:
        * numpy.ndarray: the argument is supposed to be an image and
          is not modified.
        * string (str): the argument is supposed to be filename or URL.
          The image is read in via imread.

    dtype: np.uint8 or np.float32 (FIXME[todo])
        Return image of the given type. A uint8 image will have values
        in the range 0 to 255, while a float image will be scaled to
        values between 0 and 1.

    size: Tuple[int, int] (FIXME[todo])
        Return an image of the given size. If necessary, the image will
        be resized.

    maxsize: Union[int, Tuple[int, int]] (FIXME[todo])
        The maximal size of the image. The image will be resized if the
        image extends this size in any of the two dimensions
        (width and height). If only one value is given, this is
        considered to maximal value for both, width and height.

    crop: 'central' or 'random'

    aspect: 'keep'

    read:
        The
    resize:
        The resize function to use. This defaults to 'imresize'.
    none: 'random' or 'zeros'
        Action to perform if the image argument is None.
    """
    if isinstance(img, np.ndarray):  # already image (no conversion required)
        return img
    if isinstance(img, str):  # filename/URL
        return imread(img)
    if img is None:
        if size is None:
            size = none_size
        if size is None:
            raise ValueError("No size was specified when importing None image")
        if none == 'random':
            img = np.random.randint(0, 256, size, dtype=np.uint8)
    raise TypeError(f"Cannot import image of type {type(img)}")
