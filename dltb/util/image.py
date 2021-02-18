"""Image utility functions.
"""

# standard imports
from typing import Union, Tuple

# third party imports
import numpy as np

# toolbox imports
from ..base.image import Imagelike
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


def imwrite(filename: str, image: Imagelike, **kwargs) -> None:
    """Write an image to given file.
    """
    global _writer
    if _writer is None:
        _writer = (_reader if isinstance(_reader, ImageWriter) else
                   ImageWriter())
    _writer.write(image, filename, **kwargs)


def imshow(image: Imagelike, module: str = None, **kwargs) -> None:
    """Show the given image.

    FIXME[todo]:
    Showing an image can be done in different ways:
    - blocking=True: the execution of the main program is blocked.
      The display will run an event loop to guarantee a responsive
      GUI behaviour. Blocking may stop on different occassions
        - when the display window is closed
          (either by GUI or programmatically)
        - after some timeout (the display window may then either close
          or switch into non-blocking mode, or stay open and unresponsive.
          the last should only be used, if a new image will be shown by
          the caller immediately after she regained control)
    - blocking=False: the execution of the main program is continued.
      The image display may start some background thread to ensure
      a responsive GUI behaviour

    - unblock: the unblock option specifies what should happen, when
      a blocking display ends its block:
      'close': close the display window
      'show': continue showing the image in non-blocking mode.
      'freeze': continue showing the image but without event loop,
          leaving a frozen (unresponsive) image display window.
          The caller is responsible for dealing with this window
          (either closing it or showing some new image).
    """
    display = get_display(module=module)

    if image is not None:
        display.show(image, **kwargs)
    else:
        display.close()
    return display


def imresize(image: np.ndarray, size: Tuple[int, int],
             **kwargs) -> np.ndarray:
    """Resize the given image. The result will be a new image
    of the specified size.

    """
    global _resizer
    if _resizer is None:
        _resizer = ImageResizer()
    return _resizer.resize(image, size, **kwargs)


def imscale(image: np.ndarray, scale: Union[float, Tuple[float, float]],
            **kwargs) -> np.ndarray:
    """Scale the given image. The result will be a new image
    scaled by the specified scale.

    """
    global _resizer
    if _resizer is None:
        _resizer = ImageResizer()
    return _resizer.scale(image, scale, **kwargs)


def imimport(img: Union[np.ndarray, str], dtype=np.uint8,
             size=None, none=None, **kwargs) -> np.ndarray:
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
            size = none
        if size is None:
            raise ValueError("No size was specified when importing None image")
        if none == 'random':
            img = np.random.randint(0, 256, size, dtype=np.uint8)
    raise TypeError(f"Cannot import image of type {type(img)}")


def get_display(module: str = None) -> ImageDisplay:
    # obtain a display:
    # - if a module is given, use that module
    # - otherwise use the default display
    # FIXME[hack]
    global _display

    if module is not None:
        display = ImageDisplay(module=module)
        if _display is None:
            _display = display
    elif _display is not None:
        display = _display
    else:
        module = 'qt'  # FIXME[hack]:
        display = ImageDisplay(module=module)
    if _display is None:
        _display = display
    return display


def grayscaleNormalized(array: np.ndarray) -> np.ndarray:
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

    # normalization (values should be between 0 and 1)
    min_value = array.min()
    max_value = array.max()
    div = max(max_value - min_value, 1)
    return (((array - min_value) / div) * 255).astype(np.uint8)
