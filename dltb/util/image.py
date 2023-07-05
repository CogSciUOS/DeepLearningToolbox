"""Image utility functions.
"""

# standard imports
from typing import Union, Tuple, List, Optional
from pathlib import Path
import logging

# third party imports
import numpy as np

# toolbox imports
from ..base.image import Imagelike
from ..base.image import ImageReader, ImageWriter, ImageDisplay, ImageResizer
from ..network import Network

# logging
LOG = logging.getLogger(__name__)

_reader: ImageReader = None
_writer: ImageWriter = None
_display: ImageDisplay = None
_resizer: ImageResizer = None


def imread(filename: Union[str, Path], module: Union[str, List[str]] = None,
           **kwargs) -> np.ndarray:
    """Read an image from a given file. FIXME[todo]: or URL
    """
    global _reader
    if module is not None:
        # FIXME[todo]: and _reader not from that module
        # (to avoid unneccessary reinstantiations ...)
        _reader = None
    if _reader is None:
        _reader = ImageReader(module=module)
    image = _reader.read(str(filename), **kwargs)
    LOG.debug("imread: read image of shape %s, dtype=%s from '%s' with %s.",
              image.shape, image.dtype, filename, _reader)
    return image


def imwrite(filename: Union[str, Path], image: Imagelike,
            **kwargs) -> None:
    """Write an image to a file.

    Arguments
    ---------
    filename:
        The (fully qualified) filename (including file extension)
        to which the file should be stored.
    image:
        The image to be stored.
    """
    global _writer
    if _writer is None:
        _writer = (_reader if isinstance(_reader, ImageWriter) else
                   ImageWriter())
    _writer.write(image, str(filename), **kwargs)


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

    if image is None:
        display.close()
    else:
        display.show(image, **kwargs)
    return display


def imresize(image: Imagelike, size: Tuple[int, int],
             **kwargs) -> np.ndarray:
    """Resize the given image. The result will be a new image
    of the specified size.

    """
    global _resizer
    if _resizer is None:
        _resizer = ImageResizer()
    return _resizer.resize(image, size, **kwargs)


def imscale(image: Imagelike,
            scale: Union[float, Tuple[float, float]],
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
    """Get a :py:class:`ImageDisplay` object.

    If no global display is set yet, the new display will also become
    the global display (calls to :py:func:`imshow` will use that
    display).

    Arguments
    ---------
    module:
        The module that provides the :py:class:`ImageDisplay` class.
        If this argument is given, a new :py:class:`ImageDisplay`
        will be created.

    Result
    ------
    display:
        An :py:class:`ImageDisplay`.

    """
    global _display

    if module is None:
        if _display is not None:
            display = _display
        else:
            module = 'qt'  # FIXME[hack]:
    if module is not None:
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


"""
.. module:: resize

This module defines the :py:class:`ShapeAdaptor` and
:py:class:`ResizePolicy` classes which can be used to wrap a
:py:class:`datasource.Datasource` object so that the items it yields
contain a resized version of the original image.

.. moduleauthor:: Rasmus Diederichsen
"""


class _ResizePolicyBase:
    """Base class defining common properties of all resizing policies.

    Attributes
    ----------
    _new_shape: tuple
        The shape to convert images to. Will be stripped of singular
        dimensions.
    """
    _new_shape: tuple = None

    def setShape(self, new_shape):
        """Set the shape to match. If the last dimension is 1, it is removed.

        Parameters
        ----------
        new_shape   :   tuple or list
                        Shape to match.

        Raises
        ------
        ValueError
            If leading dimension is None (aka batch)
        """
        if new_shape[0] is None:
            raise ValueError('Cannot work with None dimensions')
        if new_shape[-1] == 1:
            # remove channel dim
            new_shape = new_shape[:-1]
        self._new_shape = new_shape

    def resize(self, image):
        """Resize an image according to this policy.

        Parameters
        ----------
        image   :   np.ndarray
                    Image to resize
        """
        raise NotImplementedError("Abstract base class ResizePolicy "
                                  "cannot be used directly.")


class ResizePolicyBilinear(_ResizePolicyBase):
    """Resize policy which bilinearly interpolates images to the target
    size.

    """

    def resize(self, img):
        from dltb.util.image import imresize

        if self._new_shape is None:
            return img

        if self._new_shape[0:2] == img.shape[0:2]:
            return img

        # return resize(img, self._new_shape, preserve_range=True)
        return imresize(img, self._new_shape[0:2])


class ResizePolicyPad(_ResizePolicyBase):
    """Resize policy which will pad the input image to the target
    size. This will not work if the target size is smaller than the
    source size in any dimension.

    """

    def __init__(self, mode: str, **kwargs):
        """
        Parameters
        ----------
        mode:
            Any of the values excepted by :py:func:`np.pad`
        kwargs:
            Additional arguments to pass to :py:func:`np.pad`,
            such as the value to pad with
        """
        self._pad_mode = mode
        self._pad_kwargs = kwargs

    def resize(self, img):
        if self._new_shape is None or self._new_shape == img.shape:
            return img
        h, w = img.shape[:2]
        new_h, new_w = self._new_shape[:2]

        if new_h < h or new_w < w:
            raise ValueError("Cannot pad to a smaller size. "
                             "Use `ResizePolicy.crop` instead.")

        # necessary padding to reach desired size
        pad_h = new_h - h
        pad_w = new_w - w

        # If padding is not even, we put one more padding pixel to the
        # bottom/right
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        return np.pad(img, (top, bottom, left, right), self._pad_mode,
                      **self._pad_kwargs)


class ResizePolicyChannels(_ResizePolicyBase):
    """Resize policy which adapts the number of channels.

    Attributes
    ----------
    _channels: int
        The number of (color) cannels.
    """

    _channels = None

    def setShape(self, new_shape):
        if len(new_shape) == 3:
            self._channels = new_shape[2]
        else:
            self._channels = None

    def resize(self, img):
        if self._channels is None:
            if img.ndim == 3 and img.shape[2] == 1:
                img = np.squeeze(img, axis=2)
            elif img.ndim == 3:
                # FIXME[hack]: find better way to do RGB <-> grayscale
                # conversion
                # FIXME[question]: what are we trying to achieve here?
                # We have self._channels is None, so why do we want
                # to reduce the number of channels?
                img = np.mean(img, axis=2).astype(np.uint8)
            elif img.ndim != 2:
                raise ValueError('Incompatible shape.')
        elif img.ndim == 2:
            if self._channels == 1:
                img = img[..., np.newaxis]
            else:
                # Blow up to three dimensions by repeating the channel
                img = img[..., np.newaxis].repeat(self._channels, axis=2)
        elif self._channels > 1 and img.shape[2] == 1:
            img = img.repeat(3, axis=2)
        elif self._channels == 1 and img.shape[2] > 1:
            # FIXME[hack]: find better way to do RGB <-> grayscale
            # conversion
            img = np.mean(img, axis=2, keepdims=True).astype(np.uint8)
        elif self._channels != img.shape[2]:
            raise ValueError("Incompatible network input shape: "
                             f"network expects {self._channels} channels, "
                             f"but image has shape {img.shape}.")
        return img


class ResizePolicy:
    """
    """

    @staticmethod
    def Bilinear():
        """Create a bilinear interpolation policy."""
        return ResizePolicyBilinear()

    @staticmethod
    def Pad():
        """Create a pad-with-zero policy."""
        return ResizePolicyPad('constant', constant_values=0)

    @staticmethod
    def crop():
        raise NotImplementedError

    @staticmethod
    def Channels():
        return ResizePolicyChannels()


class ShapeAdaptor:
    """Adaptive wrapper around a :py:class:`Datasource`
    """

    def __init__(self, resize_policy: ResizePolicy,
                 network: Optional[Network] = None):
        """
        Parameters
        ----------
        resize:
            Policy to use for resizing images
        network:
            Network to adapt to
        """
        self._resize = resize_policy
        self.setNetwork(network)

    def setNetwork(self, network: Network):
        """Change the network to adapt to.

        Parameters
        ----------
        network
        """
        if network is not None:
            self._resize.setShape(network.get_input_shape(include_batch=False))

    def __call__(self, data: np.ndarray):
        return self._resize.resize(data)
