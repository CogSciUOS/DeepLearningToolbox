"""Defintion of abstract classes for image handling.

The central data structure is :py:class:`Image`, a subclass of
:py:class:`Data`, specialized to work with images.  It provides,
for example, properties like size and channels.

Relation to other `image` modules in the Deep Learning ToolBox:

* :py:mod:`dltb.util.image`: This defines general functions for image I/O and
  basic image operations. That module should be standalone, not
  (directly) requiring other parts of the toolbox (besides util) or
  third party modules (besides numpy). However, implementation for
  the interfaces defined there are provided by third party modules,
  which are automagically loaded if needed.

* :py:mod:`dltb.tool.image`: Extension of the :py:class:`Tool` API to provide
  a class :py:class:`ImageTool` which can work on `Image` data objects.
  So that module obviously depends on :py:mod:``dltb.base.image` and
  it may make use of functionality provided by :py:mod:`dltb.util.image`.
"""

# standard imports
from typing import Union, List, Tuple, Dict, Any
from abc import abstractmethod, ABC
from collections import namedtuple
from enum import Enum
import threading
import logging
import time

# third-party imports
import numpy as np

# toolbox imports
from .observer import Observable
from .data import Data, DataDict, BatchDataItem
from ..util.error import handle_exception
from .. import thirdparty

# logging
LOG = logging.getLogger(__name__)


# FIXME[todo]: create an interface to work with different image/data formats
# (as started in dltb.thirdparty.pil)
# * add a way to specify the default format for reading images
#   - in dltb.util.image.imread(format='pil')
#   - for Imagesources
# * add on the fly conversion for Data objects, e.g.
#   data.pil should
#   - check if property pil already exists
#   - if not: invoke Image.as_pil(data)
#   - store the result as property data.pil
#   - return it
# * this method could be extended:
#   - just store filename and load on demand
#   - compute size on demand
#


# Imagelike is intended to be everything that can be used as
# an image.
#
# np.ndarray:
#    The raw image data
# str:
#    A URL.
Imagelike = Union[np.ndarray, str]

Size = namedtuple('Size', ['width', 'height'])


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

    size: Size = None

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


class Image(DataDict):
    """A collection of image related functions.
    """

    converters = {
        'array': [
            (np.ndarray, lambda array, copy: (array, copy)),
            (Data, lambda data, copy: (data.array, copy)),
            (BatchDataItem, lambda data, copy: (data.array, copy))
        ],
        'image': [
            (np.ndarray, Data)
        ]
    }

    @classmethod
    def add_converter(cls, source: type, converter,
                      target: str = 'image') -> None:
        """Register a new image converter. An image converter is
        a function, that can convert an given image into another
        format.

        Arguments
        ---------
        source:
            The input type of the converter, that is the type of
            its first argument.
        convert:
            The actual converter function.
        target:
            The output format. This can be `image` (the converter
            produces an instance of `Image`) or `array` (a numpy array).
        """
        # FIXME[todo]: make this more flexible, use introspection,
        # get rid off the copy parameter, deal with other arguments
        cls.converters[target].append((source, converter))

    @classmethod
    def as_array(cls, image: Imagelike, copy: bool = False,
                 dtype=None,  # FIXME[todo]: not implemented yet
                 colorspace: Colorspace = None) -> np.ndarray:
        """Get image-like object as numpy array. This may
        act as the identity function in case `image` is already
        an array, or it may extract the relevant property, or
        it may even load an image from a filename.

        Arguments
        ---------
        image: Imagelike
            An image like object to turn into an array.
        copy: bool
            A flag indicating if the data should be copied or
            if the original data is to be returned (if possible).
        dtype:
            Numpy datatype, e.g., numpy.float32.
        colorspace: Colorspace
            The colorspace in which the pixels in the resulting
            array are encoded.  If no colorspace is given, or
            if the colorspace of the input image Image is unknown,
            no color conversion is performed.
        """
        for source_class, converter in cls.converters['array']:
            if isinstance(image, source_class):
                LOG.debug("Using image converter for type %s (copy=%s)",
                          type(image), copy)
                image, copy = converter(image, copy)
                break
        else:
            if isinstance(image, str):
                # FIXME[hack]: local imports to avoid circular module
                # dependencies ...
                # pylint: disable=import-outside-toplevel
                from dltb.util.image import imread
                LOG.debug("Loading image '%s' using imread.", image)
                image, copy = imread(image), False
            else:
                raise NotImplementedError(f"Conversion of "
                                          f"{type(image).__module__}"
                                          f".{type(image).__name__} to "
                                          "numpy.ndarray is not implemented")
        LOG.debug("Obtained image of shape %s, dtype=%s.",
                  image.shape, image.dtype)

        if colorspace == Colorspace.RGB:
            if len(image.shape) == 2:  # grayscale image
                rgb = np.empty(image.shape + (3,), dtype=image.dtype)
                rgb[:, :, :] = image[:, :, np.newaxis]
                image = rgb
                copy = False
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBD
                image = image[:, :, :3]

        if dtype is not None and dtype != image.dtype:
            image = image.astype(dtype)  # /256.
            copy = False

        if copy:
            image = image.copy()

        LOG.debug("Returning image of shape %s, dtype=%s.",
                  image.shape, image.dtype)
        return image

    @staticmethod
    def as_data(image: Imagelike, copy: bool = False) -> 'Data':
        """Get image-like objec as :py:class:`Data` object.
        """
        if isinstance(image, Data) and not copy:
            return image

        data = Image(image, copy=copy)
        if isinstance(image, str):
            data.add_attribute('url', image)
        return data

    def __init__(self, image: Imagelike = None, array: np.ndarray = None,
                 copy: bool = False, **kwargs) -> None:
        if image is not None:
            array = self.as_array(image, copy=copy)
        super().__init__(array=array, **kwargs)


class ImageAdapter(ABC):
    """If an object is an ImageAdapter, it can adapt images to
    some internal representation. It has to implement the
    :py:class:`image_to_internal` and :py:class:`internal_to_image`
    methods. Such an object can then be extended to do specific
    image processing.

    The :py:class:`ImageAdapter` keeps a map of known
    :py:class:`ImageExtension`. If a subclass of
    :py:class:`ImageAdapter` also subclasses a base class of these
    extensions it will be adapted to also subclass the corresponding
    extension, e.g., a :py:class:`ImageAdapter` that is a `Tool` will
    become an `ImageTool`, provided the mapping of `Tool` to
    `ImageTool` has been registered with the `ImageAdapter` class.
    Creating `ImageTool` as an :py:class:`ImageExtension` of
    `base=Tool` will automatically do the registration.
    """

    _image_extensions: Dict[type, type] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        for base, replacement in ImageAdapter._image_extensions.items():
            if base in cls.__mro__ and replacement not in cls.__mro__:
                new_bases = []
                found = False
                for base_class in cls.__bases__:
                    if base_class is base:
                        found = True
                        new_bases.append(replacement)
                        continue
                    if not found and issubclass(base_class, base):
                        new_bases.append(replacement)
                        found = True
                    new_bases.append(base_class)
                LOG.debug("ImageAdapter.__init_subclass__(%s): %s -> %s",
                          cls, cls.__bases__, new_bases)
                cls.__bases__ = tuple(new_bases)

    @abstractmethod
    def image_to_internal(self, image: Imagelike) -> Any:
        "to be implemented by subclasses"

    @abstractmethod
    def internal_to_image(self, data: Any) -> Imagelike:
        "to be implemented by subclasses"


class ImageExtension(ImageAdapter):
    """An :py:class:`ImageExtension` extends some base class to be able to
    process images. In that it makes use of the :py:class:`ImageAdapter`
    interface.

    In addition to deriving from :py:class:`ImageAdapter`, the
    :py:class:`ImageExtension` introduces some "behind the scene
    magic": a class `ImageTool` that is declared as an `ImageExtension`
    with base `Tool` is registered with the :py:class:`ImageAdapter`
    class, so that any common subclass of :py:class:`ImageAdapter`
    and `Tool` will automagically become an `ImageTool`.
    """

    def __init_subclass__(cls, base: type = None, **kwargs) -> None:
        # pylint: disable=arguments-differ
        super().__init_subclass__(**kwargs)
        if base is not None:
            new_bases = [ImageAdapter, base]
            for base_class in cls.__bases__:
                if base_class is not ImageExtension:
                    new_bases.append(base_class)
            cls.__bases__ = tuple(new_bases)
            ImageAdapter._image_extensions[base] = cls


class ImageObservable(Observable, method='image_changed',
                      changes={'image_changed'}):
    """A base for classes that can create and change images.
    """

    @property
    def image(self) -> Imagelike:
        """Provide the current image.
        """


class ImageGenerator(ImageObservable):
    # pylint: disable=too-few-public-methods
    """An image :py:class:`Generator` can generate images.
    """
    # FIXME[todo]: spell this out


class ImageIO:
    # pylint: disable=too-few-public-methods
    """An abstract interface to read, write and display images.
    """


class ImageReader(ImageIO):
    """An :py:class:`ImageReader` can read images from file or URL.
    The :py:meth:`read` method is the central method of this class.
    """

    def __new__(cls, module: Union[str, List[str]] = None) -> 'ImageReader':
        if cls is ImageReader:
            new_cls = thirdparty.import_class('ImageReader', module=module)
        else:
            new_cls = cls
        return super(ImageReader, new_cls).__new__(new_cls)

    def __str__(self) -> str:
        return type(self).__module__ + '.' + type(self).__name__

    def read(self, filename: str, **kwargs) -> np.ndarray:
        """Read an image from a file or URL.
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to "
                                  "be an ImageReader, but does not implement "
                                  "the read method.")


class ImageWriter(ImageIO):
    """An :py:class:`ImageWriter` can write iamges to files or upload them
    to a given URL.  The :py:meth:`write` method is the central method
    of this class.

    """

    def __new__(cls, module: Union[str, List[str]] = None) -> 'ImageWriter':
        if cls is ImageWriter:
            new_cls = thirdparty.import_class('ImageWriter', module=module)
        else:
            new_cls = cls
        return super(ImageWriter, new_cls).__new__(new_cls)

    def write(self, filename: str, image: Imagelike, **kwargs) -> None:
        """Write an `image` to a file with the given `filename`.
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to "
                                  "be an ImageWriter, but does not implement "
                                  "the write method.")


class ImageResizer:
    """FIXME[todo]: there is also the network.resize module, which may be
    incorporated!

    Image resizing is implemented by various libraries, using slightly
    incompatible interfaces.  The idea of this class is to provide a
    well defined resizing behaviour, that offers most of the functionality
    found in the different libraries.  Subclasses can be used to map
    this interface to specific libraries.

    Enlarging vs. Shrinking
    -----------------------

    Interpolation:
    * Linear, cubic, ...
    * Mean value:

    Cropping
    --------
    * location: center, random, or fixed
    * boundaries: if the crop size is larger than the image: either
      fill boundaries with some value or return smaller image



    Parameters
    ----------

    * size:
      scipy.misc.imresize:
          size : int, float or tuple
          - int - Percentage of current size.
          - float - Fraction of current size.
          - tuple - Size of the output image.

    * zoom : float or sequence, optional
      in scipy.ndimage.zoom:
         "The zoom factor along the axes. If a float, zoom is the same
         for each axis. If a sequence, zoom should contain one value
         for each axis."

    * downscale=2, float, optional
      in skimage.transform.pyramid_reduce
         "Downscale factor.

    * preserve_range:
      skimage.transform.pyramid_reduce:
          "Whether to keep the original range of values. Otherwise, the
          input image is converted according to the conventions of
          img_as_float."

    * interp='nearest'
      in scipy.misc.imresize:
          "Interpolation to use for re-sizing
          ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic')."

    * order: int, optional
      in scipy.ndimage.zoom, skimage.transform.pyramid_reduce:
          "The order of the spline interpolation, default is 3. The
          order has to be in the range 0-5."
          0: Nearest-neighbor
          1: Bi-linear (default)
          2: Bi-quadratic
          3: Bi-cubic
          4: Bi-quartic
          5: Bi-quintic

    * mode: str, optional
      in scipy.misc.imresize:
          "The PIL image mode ('P', 'L', etc.) to convert arr
          before resizing."

    * mode: str, optional
      in scipy.ndimage.zoom, skimage.transform.pyramid_reduce:
          "Points outside the boundaries of the input are filled
          according to the given mode ('constant', 'nearest',
          'reflect' or 'wrap'). Default is 'constant'"
          - 'constant' (default): Pads with a constant value.
          - 'reflect': Pads with the reflection of the vector mirrored
            on the first and last values of the vector along each axis.
          - 'nearest':
          - 'wrap': Pads with the wrap of the vector along the axis.
             The first values are used to pad the end and the end
             values are used to pad the beginning.

    * cval: scalar, optional
      in scipy.ndimage.zoom, skimage.transform.pyramid_reduce:
          "Value used for points outside the boundaries of the input
          if mode='constant'. Default is 0.0"

    * prefilter: bool, optional
      in scipy.ndimage.zoom:
          "The parameter prefilter determines if the input is
          pre-filtered with spline_filter before interpolation
          (necessary for spline interpolation of order > 1). If False,
          it is assumed that the input is already filtered. Default is
          True."

    * sigma: float, optional
      in skimage.transform.pyramid_reduce:
          "Sigma for Gaussian filter. Default is 2 * downscale / 6.0
          which corresponds to a filter mask twice the size of the
          scale factor that covers more than 99% of the Gaussian
          distribution."


    Libraries providing resizing functionality
    ------------------------------------------

    Scikit-Image:
    * skimage.transform.resize:
        image_resized = resize(image, (image.shape[0]//4, image.shape[1]//4),
                               anti_aliasing=True)
      Documentation:
      https://scikit-image.org/docs/dev/api/skimage.transform.html
          #skimage.transform.resize

    * skimage.transform.rescale:
      image_rescaled = rescale(image, 0.25, anti_aliasing=False)

    * skimage.transform.downscale_local_mean:
       image_downscaled = downscale_local_mean(image, (4, 3))
       https://scikit-image.org/docs/dev/api/skimage.transform.html
           #skimage.transform.downscale_local_mean

    Pillow:
    * PIL.Image.resize:

    OpenCV:
    * cv2.resize:
      cv2.resize(image,(width,height))

    Mahotas:
    * mahotas.imresize:

      mahotas.imresize(img, nsize, order=3)
      This function works in two ways: if nsize is a tuple or list of
      integers, then the result will be of this size; otherwise, this
      function behaves the same as mh.interpolate.zoom

    * mahotas.interpolate.zoom

    imutils:
    * imutils.resize

    Scipy (deprecated):
    * scipy.misc.imresize:
      The documentation of scipy.misc.imresize says that imresize is
      deprecated! Use skimage.transform.resize instead. But it seems
      skimage.transform.resize gives different results from
      scipy.misc.imresize.
      https://stackoverflow.com/questions/49374829/scipy-misc-imresize-deprecated-but-skimage-transform-resize-gives-different-resu

      SciPy: scipy.misc.imresize is deprecated in SciPy 1.0.0,
      and will be removed in 1.3.0. Use Pillow instead:
      numpy.array(Image.fromarray(arr).resize())

    * scipy.ndimage.interpolation.zoom:
    * scipy.ndimage.zoom:
    * skimage.transform.pyramid_reduce: Smooth and then downsample image.

    """

    def __new__(cls, module: Union[str, List[str]] = None) -> 'ImageWriter':
        if cls is ImageResizer:
            new_cls = thirdparty.import_class('ImageResizer', module=module)
        else:
            new_cls = cls
        return super(ImageResizer, new_cls).__new__(new_cls)

    def resize(self, image: np.ndarray,
               size: Size, **_kwargs) -> np.ndarray:
        """Resize an image to the given size.

        Arguments
        ---------
        image:
            The image to be scaled.
        size:
            The target size.
        """
        if type(self).scale is ImageResizer.scale:
            raise NotImplementedError(f"{type(self)} claims to be an "
                                      "ImageResizer, but does not implement "
                                      "the resize method.")
        image_size = image.shape[:2]
        scale = (size[0]/image_size[0], size[1]/image_size[1])
        return self.scale(image, scale=scale)

    def scale(self, image: np.ndarray,
              scale: Union[float, Tuple[float, float]],
              **kwargs) -> np.ndarray:
        """Scale an image image by a given factor.

        Arguments
        ---------
        image:
            The image to be scaled.
        scale:
            Either a single float value being the common
            scale factor for horizontal and vertical direction, or
            a pair of scale factors for these two axes.
        """
        if type(self).resize is ImageResizer.resize:
            raise NotImplementedError(f"{type(self)} claims to be an "
                                      "ImageResizer, but does not implement "
                                      "the scale method.")

        if isinstance(scale, float):
            scale = (scale, scale)

        image_size = image.shape[:2]
        size = (int(image_size[0] * scale[0]), int(image_size[1] * scale[1]))
        return self.resize(image, size=size, **kwargs)

    @staticmethod
    def crop(image: Imagelike, size: Size, **_kwargs) -> np.ndarray:
        """Crop an :py:class:`Image` to a given size.

        If now position is provided, a center crop will be performed.
        """
        # FIXME[todo]: deal with sizes extending the original size
        # FIXME[todo]: allow center/random/position crop
        image = Image.as_array(image)
        old_size = image.shape[:2]
        center = old_size[0]//2, old_size[1]//2
        point1 = center[0] - size[0]//2, center[1] - size[1]//2
        point2 = point1[0] + size[0], point1[1] + size[1]
        return image[point1[0]:point2[0], point1[1]:point2[1]]


class ImageOperator:
    """An :py:class:`ImageOperator` can be applied to an image to
    obtain some transformation of that image.
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Perform the actual operation.
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to "
                                  "be an ImageOperator, but does not "
                                  "implement the `__call__` method.")

    def transform(self, source: str, target: str) -> None:
        """Transform a source file into a target file.
        """
        # FIXME[concept]: this requires the util.image module!
        # pylint: disable=import-outside-toplevel
        from ..util.image import imread, imwrite
        imwrite(target, self(imread(source)))

    def transform_data(self, image: Image,
                       target: str, source: str = None) -> None:
        """Apply image operator to an :py:class:`Image` data object.
        """
        image.add_attribute(target, value=self(image.get_attribute(source)))


class ImageDisplay(ImageIO, ImageGenerator.Observer):
    """An :py:class:`ImageDisplay` can display images.  Typically, it will
    use some graphical user interface to open a window in which the
    image is displayed. It may also provide some additional controls
    to addapt display properties.

    Blocking and non-blocking display
    ---------------------------------

    There are two ways how an image can be displayed.  In blocking
    mode the execution of the main program is paused while the image
    is displayed and is only continued when the display is closed.  In
    non-blocking mode, the the execution of the main program is
    continued while the image is displayed.

    The blocking behaviour can be controlled by the `blocking`
    argument. It can be set to `True` (running the GUI event loop in
    the calling thread and thereby blocking it) or `False` (running
    the GUI event loop in some other thread).  It can also be set to
    `None` (meaning that no GUI event loop is started, which is
    similar to the non-blocking mode, however it will usually result
    in an inresponsive display window if no additional actions are
    undertaken; see the section on "GUI Event loop" below for more
    information).

    Ending the display
    ------------------

    Different conditions can be set up to determine when the display
    should end.  The most natural one is to wait until the display
    window is closed (using the standard controls of the window
    system). Additionally, the display can be terminated when a key is
    pressed on the keyboard or after a given amount of time.
    If run in a multi-threaded setting, it is also possible to end
    the display programatically, calling :py:meth:`close`.

    The next question is: what should happen once the display ended?
    Again the most natural way is to close the window.  However,
    if more images are going to be displayed it may be more suitable
    to leave the window on screen an just remove the image, until the
    next image is available.

    GUI Event loop
    --------------

    An :py:class:`ImageDisplay` displays the image using some
    graphical user interface (GUI).  Such a GUI usually requires to
    run an event loop to stay responsive, that is to react to mouse
    and other actions, like resizing, closing and even repainting the
    window. The event loop regularly checks if such events have
    occured and processes them. Running a display without an event
    loop usually results in unpleasant behaviour and hence should be
    avoided.

    Nevertheless, running an event loop is not always straight forward.
    Different GUI libraries use different concepts. For example, some
    libraries require that event loops are run in the main thread of
    the application, which can not always be realized (for example, it
    would not be possible to realize a non-blocking display in the
    main thread).  The :py:class:`ImageDisplay` provides different
    means to deal with such problems.

    Usage scenarios
    ---------------

    Example 1: show an image in a window and block until the window is
    closed:

    >>> display = Display()  # blocking=True (default)
    >>> display.show(imagelike)

    Example 2: show an image in a window without blocking (the event loop
    for the window will be run in a separate thread):

    >>> display = Display(blocking=False)
    >>> display.show(imagelike)

    Example 3: show an image in a window without blocking. No event loop
    is started for the window and it is the caller's responsibility to
    regularly call display.process_events() to keep the interface
    responsive.

    >>> display = Display(blocking=None)
    >>> display.show(imagelike)

    Example 4: show an image for five seconds duration.
    After 5 seconds the display is closed.

    >>> display = Display()
    >>> display.show(imagelike, timeout=5.0)

    Example 5: show multiple images, each for five seconds, but don't close
    the window in between:

    >>> with Display() as display:
    >>>     for image in images:
    >>>         display.show(image, timeout=5.0)

    Example 6: presenter:

    >>> def presenter(display, video):
    >>>     while frame in video:
    >>>         if display.closed:
    >>>             break
    >>>         display.show(frame)
    >>>
    >>> display = Display()
    >>> display.present(presenter, (video,))

    """

    def __new__(cls, module: Union[str, List[str]] = None,
                **_kwargs) -> 'ImageDisplay':
        if cls is ImageDisplay:
            new_cls = thirdparty.import_class('ImageDisplay', module=module)
        else:
            new_cls = cls
        return super(ImageDisplay, new_cls).__new__(new_cls)

    def __init__(self, module: Union[str, List[str]] = None,
                 blocking: bool = True, **kwargs) -> None:
        # pylint: disable=unused-argument
        super().__init__(**kwargs)

        # _opened: a flag indicating the current state of the display
        # window: True = window is open (visible), False = window is closed
        self._opened: bool = False

        # _blocking: a flag indicating if the display should operate
        # in blocking mode (True) or non-blocking mode (False).
        self._blocking: bool = blocking

        # _entered: a counter to for tracing how often the context manager
        # is used (usually it should only be used once!)
        self._entered: int = 0

        # _event_loop: some Thread object, referring to the thread running the
        # event loop.  If None, then currently no event loop is running.
        self._event_loop = None

        # _presentation: a Thread object running a presentation, initiated
        # by the method `present`
        self._presentation: threading.Thread = None

    @property
    def blocking(self) -> bool:
        """Blocking behaviour of this image :py:class:`Display`.  `True` means
        that an event loop is run in the calling thread and execution
        of the program is blocked while showing an image, `False`
        means that the event loop is executed in a background thread
        while the calling thread immediately returns. `None` means
        that no event loop is started. The caller is responsible for
        processing events, by regurlarly calling either
        :py:meth:`process_events` or :py:meth:`show` (which internally
        calls :py:meth:`process_events`).

        """
        return self._blocking

    @blocking.setter
    def blocking(self, blocking: bool) -> None:
        if blocking is self._blocking:
            return  # nothing to do
        if not self.closed:
            raise RuntimeError("Cannot change blocking state of open Display.")
        self._blocking = blocking

    #
    # context manager
    #

    def __enter__(self) -> 'ImageDisplay':
        self._entered += 1
        if self._entered > 1:
            LOG.warning("Entering Display multiple times: %d", self._entered)
        else:
            LOG.debug("Entering Display")
        self.open()
        return self

    def __exit__(self, _exception_type, _exception_value, _traceback) -> None:
        LOG.debug("Exiting Display (%d)", self._entered)
        self._entered -= 1
        if self._entered == 0:
            self.close()

    #
    # public interface
    #

    def show(self, image: Imagelike, blocking: bool = None, close: bool = None,
             timeout: float = None, **kwargs) -> None:
        """Display the given image.

        This method may optionally pause execution of the main program
        to display the image, if the wait_for_key or timeout arguments
        are given.  If both are given, the first event that occurs
        will stop pausing.

        Arguments
        ---------
        image: Imagelike
            The image to display. This may be a single image or a
            batch of images.
        blocking: bool
            A flag indicating if the image should be shown in blocking
            mode (`True`) or non-blocking mode (`False`).  If no value
            is specified, the value of the property :py:prop:`blocking`
            is used.
        close: bool
            A flag indicating if the display should be closed after
            showing. Closing the display will also end all event
            loops that are running. If no value is provided, the
            display will be kept open, if it was already open when
            this method is called, and it will be closed in case it
            was closed before.
        wait_for_key: bool
            A flag indicating if the display should pause execution
            and wait or a key press.
        timeout: float
            Time in seconds to pause execution.

        """
        if self._presentation is not None:
            blocking = None
        else:
            blocking = self._blocking if blocking is None else blocking

        if close is None:
            close = self.closed and (blocking is True)

        # make sure the window is open
        if self.closed:
            if self._presentation is threading.current_thread():
                raise RuntimeError("Presentation is trying to use closed "
                                   "ImageDisplay.")
            self.open()

        # show the image
        array = Image.as_array(image, dtype=np.uint8)
        LOG.debug("Showing image of shape %s, close=%s, timout=%s, "
                  "event loop=%s, presentation=%s",
                  array.shape, close, timeout, self.event_loop_is_running(),
                  self._presentation is not None)
        self._show(array, **kwargs)

        # run the event loop
        if blocking is True:
            if not self.event_loop_is_running() is None:
                self._run_blocking_event_loop(timeout=timeout)
        elif blocking is False:
            if timeout is not None:
                LOG.warning("Setting timeout (%f) has no effect "
                            " for non-blocking image Display", timeout)
            if not self.event_loop_is_running():
                self._run_nonblocking_event_loop()
        elif blocking is None:
            self._process_events()

        # close the window if desired
        if close:
            if self._entered > 0:
                LOG.warning("Closing image Display inside a context manager.")
            self.close()

    def present(self, presenter, args=(), kwargs={}) -> None:
        # pylint: disable=dangerous-default-value
        """Run the given presenter in a background thread while
        executing the GUI event loop in the calling thread (which
        by some GUI library is supposed to be the main thread).

        The presenter will get the display as its first argument,
        and `args`, `kwargs` as additional arguments. The presenter
        may update the display by calling the :py:meth:`show` method.
        The presenter should observe the display's `closed` property
        and finish presentation once it is set to `True`.

        Arguments
        ---------
        presenter:
            A function expecting a display object as first argument
            and `args`, and `kwargs` as additional arguments.
        """
        def target() -> None:
            # pylint: disable=broad-except
            LOG.info("ImageDisplay[background]: calling presenter")
            try:
                presenter(self, *args, **kwargs)
            except BaseException as exception:
                LOG.error("Unhandled exception in presentation.")
                handle_exception(exception)
            finally:
                self.close()

        with self:
            LOG.info("ImageDisplay[main]: Starting presentation")
            self._presentation = threading.Thread(target=target)
            self._presentation.start()
            self._run_blocking_event_loop()

    def open(self) -> None:
        if not self._opened and self._presentation is None:
            self._open()
            self._opened = True

    def close(self) -> None:
        """Close this :py:class:`ImageDisplay`. This should also stop
        all background threads, like event loops or ongoing presentatons
        """
        LOG.info("Closing ImageDisplay "
                 "(opened=%s, presentation=%s, event loop=%s)",
                 self._opened, self._presentation is not None,
                 self.event_loop_is_running())
        if self._opened:
            self._opened = False
            self._close()

        presentation = self._presentation
        if presentation is not None:
            # we have started a presentation in a background Thread and
            # hence we will wait that this presentation finishes. In
            # order for this to work smoothly, the presentation should
            # regularly check the display.closed property and exit
            # (before calling display.show) if that flag is True.
            if presentation is not threading.current_thread():
                presentation.join()
                self._presentation = None

        event_loop = self._event_loop
        if isinstance(event_loop, threading.Thread):
            if event_loop is not threading.current_thread():
                event_loop.join()
            self._event_loop = None

    @property
    def opened(self) -> bool:
        """Check if this image :py:class:`Display` is opened, meaning
        the display window is shown and an event loop is running.
        """
        return self._opened

    @property
    def closed(self) -> bool:
        """Check if this image :py:class:`Display` is closed, meaning
        that no window is shown (and no event loop is running).
        """
        return not self._opened

    #
    # ImageObserver
    #

    def image_changed(self, tool, change) -> None:
        """Implementation of the :py:class:`ImageObserver` interface.
        The display will be updated if the image has changed.
        """
        if change.image_changed:
            self.show(tool.image)

    #
    # methods to be implemented by subclasses
    #

    def _open(self) -> None:
        """Open the display window. The function is only called if
        no window is open yet.
        """
        raise NotImplementedError(f"{type(self)} claims to be a ImageDisplay, "
                                  "but does not implement an _open() method.")

    def _show(self, image: np.ndarray, wait_for_key: bool = False,
              timeout: float = None, **kwargs) -> None:
        raise NotImplementedError(f"{type(self).__name__} claims to "
                                  "be an ImageDisplay, but does not implement "
                                  "the _show method.")

    def _close(self) -> None:
        raise NotImplementedError(f"{type(self)} claims to be a ImageDisplay, "
                                  "but does not implement an _close() method.")

    def _process_events(self) -> None:
        raise NotImplementedError(f"{type(self)} claims to be a ImageDisplay, "
                                  "but does not implement "
                                  "_process_events().")

    def _run_event_loop(self) -> None:
        if self.blocking is True:
            self._run_blocking_event_loop()
        elif self.blocking is False:
            self._run_nonblocking_event_loop()

    def _dummy_event_loop(self, timeout: float = None) -> None:
        # pylint: disable=broad-except
        interval = 0.1

        start = time.time()
        try:
            print("ImageDisplay: start dummy event loop. "
                  f"closed={self.closed}")
            while (not self.closed and
                   (timeout is None or time.time() < start + timeout)):
                self._process_events()
                time.sleep(interval)
        except BaseException as exception:
            LOG.error("Unhandled exception in event loop")
            handle_exception(exception)
        finally:
            LOG.info("ImageDisplay: ended dummy event loop (closed=%s).",
                     self.closed)
            self._event_loop = None
            self.close()

    def _run_blocking_event_loop(self, timeout: float = None) -> None:
        self._event_loop = threading.current_thread
        self._dummy_event_loop(timeout)

    def _run_nonblocking_event_loop(self) -> None:
        """Start a dummy event loop. This event loop will run in the
        background and regularly trigger event processing. This may be
        slightly less responsive than running the official event loop,
        but it has the advantage that this can be done from a background
        Thread, allowing to return the main thread to the caller.
        In other words: this function is intended to realize a non-blocking
        image display with responsive image window.

        FIXME[todo]: check how this behaves under heavy load (GPU computation)
        and if in case of problems, resorting to a QThread would improve
        the situation.
        """
        if self.event_loop_is_running():
            raise RuntimeError("Only one event loop is allowed.")
        self._event_loop = \
            threading.Thread(target=self._nonblocking_event_loop)
        self._event_loop.start()

    def _nonblocking_event_loop(self) -> None:
        self._dummy_event_loop()

    def event_loop_is_running(self) -> bool:
        """Check if an event loop is currently running.
        """
        return self._event_loop is not None

    # ------------------------------------------------------------------------
    # FIXME[old/todo]: currently used by ./contrib/styletransfer.py ...
    def run(self, tool):
        """Monitor the operation of a Processor. This will observe
        the processor and update the display whenever new data
        are available.
        """
        self.observe(tool, interests=ImageGenerator.Change('image_changed'))
        try:
            print("Starting thread")
            thread = threading.Thread(target=tool.loop)
            thread.start()
            # FIXME[old/todo]: run the main event loop of the GUI to get
            # a responsive interface - this is probably framework
            # dependent and should be realized in different subclasses
            # before we can design a general API.
            # Also we would need some stopping mechanism to end the
            # display (by key press or buttons, but also programmatically)
            # self._application.exec_()
            print("Application main event loop finished")
        except KeyboardInterrupt:
            print("Keyboard interrupt.")
        tool.stop()
        thread.join()
        print("Thread joined")

    # FIXME[old/todo]: currently used by ./dltb/thirdparty/qt.py/dltb/thirdparty/qt.py ...
    @property
    def active(self) -> bool:
        """Check if this :py:class:`ImageDisplay` is active.
        """
        return True  # FIXME[hack]



class Location:
    """A :py:class:`Location` identifies an area in a two-dimensional
    space.  A typical location is a bounding box (realized by the
    subclass :py:class:`BoundingBox`), but this abstract definition
    also allows for alternative ways to describe a location.

    """

    def __init__(self, points) -> None:
        pass

    def mark_image(self, image: Imagelike, color=(1, 0, 0)):
        """Mark this :py:class:`Location` in some given image.

        Arguments
        ---------
        image:
        """
        raise NotImplementedError(f"Location {self.__class__.__name__} "
                                  f"does not provide a method for marking "
                                  f"an image.")

    def extract_from_image(self, image: Imagelike) -> np.ndarray:
        """Extract this :py:class:`Location` from a given image.

        Arguments
        ---------
        image:
        """
        raise NotImplementedError(f"Location {self.__class__.__name__} "
                                  f"does not provide a method for extraction "
                                  f"from an image.")

    def scale(self, factor):
        raise NotImplementedError(f"Location {self.__class__.__name__} "
                                  f"does not provide a method for scaling.")


class PointsBasedLocation:
    """A :py:class:`PointsBasedLocation` is a :py:class:`Location`
    that can be described by points, like a polygon area, or more
    simple: a bounding box.

    Attributes
    ----------
    _points: np.ndarray
        An array of shape (n, 2), providing n points in form of (x, y)
        coordinates.
    """

    def __init__(self, points: np.ndarray) -> None:
        super().__init__()
        self._points = points

    def __contains__(self, point) -> bool:
        return ((self._points[:, 0].min() <= point[0] <=
                 self._points[:, 0].max()) and
                (self._points[:, 1].min() <= point[1] <=
                 self._points[:, 1].max()))

    def mark_image(self, image: np.ndarray, color=(1, 0, 0)):
        """Mark this :py:class:`PointsBasedLocation` in an image.
        """
        for point in self._points:
            image[max(point[1]-1, 0):min(point[1]+1, image.shape[0]),
                  max(point[0]-1, 0):min(point[0]+1, image.shape[1])] = color

    def extract_from_image(self, image: Imagelike) -> np.ndarray:
        """Extract this :py:class:`Location` from a given image.

        Arguments
        ---------
        image:
            The image from which this :py:class:`PointsBasedLocation`
            is to be extracted.
        """
        image = Image.as_array(image)
        height, width = image.shape[:2]
        point1_x, point1_y = self._points.min(axis=0)
        point2_x, point2_y = self._points.max(axis=0)
        point1_x, point1_y = max(0, int(point1_x)), max(0, int(point1_y))
        point2_x, point2_y = \
            min(width, int(point2_x)), min(height, int(point2_y))
        return image[point1_y:point2_y, point1_x:point2_x]

    def scale(self, factor) -> None:
        """Scale the :py:class:`Location`.

        Arguments
        ---------
        factor:
            The scaling factor. This can either be a float, or a pair
            of floats in which case the first number is the horizontal (x)
            scaling factor and the second numger is the vertical (y)
            scaling factor.
        """
        self._points *= factor

    @property
    def points(self):
        return self._points

    def __len__(self):
        return len(self._points)


class Landmarks(PointsBasedLocation):
    """Landmarks are an ordered list of points.
    """

    def __len__(self) -> int:
        return 0 if self._points is None else len(self._points)


class BoundingBox(PointsBasedLocation):
    # pylint: disable=invalid-name
    """A bounding box describes a rectangular arae in an image.
    """

    def __init__(self, x1=None, y1=None, x2=None, y2=None,
                 x=None, y=None, width=None, height=None) -> None:
        super().__init__(np.ndarray((2, 2)))
        if x1 is not None:
            self.x1 = x1
        elif x is not None:
            self.x1 = x

        if y1 is not None:
            self.y1 = y1
        elif y is not None:
            self.y1 = y

        if x2 is not None:
            self.x2 = x2
        elif width is not None:
            self.width = width

        if y2 is not None:
            self.y2 = y2
        elif height is not None:
            self.height = height

    @property
    def x1(self):
        return self._points[0, 0]

    @x1.setter
    def x1(self, x1):
        self._points[0, 0] = x1

    @property
    def y1(self):
        return self._points[0, 1]

    @y1.setter
    def y1(self, y1):
        self._points[0, 1] = y1

    @property
    def x2(self):
        return self._points[1, 0]

    @x2.setter
    def x2(self, x2):
        self._points[1, 0] = x2

    @property
    def y2(self):
        return self._points[1, 1]

    @y2.setter
    def y2(self, y2):
        self._points[1, 1] = y2

    @property
    def x(self):
        return self.x1

    @x.setter
    def x(self, x):
        self.x1 = x

    @property
    def y(self):
        return self.y1

    @y.setter
    def y(self, y):
        self.y1 = y

    @property
    def width(self):
        return self.x2 - self.x1

    @width.setter
    def width(self, width):
        self.x2 = self.x1 + width

    @property
    def height(self):
        return self.y2 - self.y1

    @height.setter
    def height(self, height):
        self.y2 = self.y1 + height

    def mark_image(self, image: np.ndarray, color=None) -> None:
        color = color or (0, 255, 0)
        size = image.shape[1::-1]
        thickness = max(1, max(size)//300)
        t1 = thickness//2
        t2 = (thickness+1)//2
        x1 = max(int(self.x1), t2)
        y1 = max(int(self.y1), t2)
        x2 = min(int(self.x2), size[0]-t1)
        y2 = min(int(self.y2), size[1]-t1)
        # print(f"mark_image[{self}]: image size={size}"
        #       f"shape={image.shape}, {image.dtype}:"
        #       f"{image.min()}-{image.max()}, box:({x1}, {y1}) - ({x2}, {y2})")
        for offset in range(-t2, t1):
            image[(y1+offset, y2+offset), x1:x2] = color
            image[y1:y2, (x1+offset, x2+offset)] = color

    def extract(self, image: Imagelike, padding: bool = True,
                copy: bool = None) -> np.ndarray:
        """Extract the region described by the bounding box from an image.
        """
        image = Image.as_array(image)
        image_size = image.shape[1::-1]
        channels = 1 if image.ndim < 3 else image.shape[2]

        x1, x2 = int(self.x1), int(self.x2)
        y1, y2 = int(self.y1), int(self.y2)
        invalid = (x1 < 0 or x2 > image_size[0] or
                   y1 < 0 or y2 > image_size[1])

        if invalid and padding:
            copy = True
        else:
            # no padding: resize bounding box to become valid
            x1, x2 = max(x1, 0), min(x2, image_size[0])
            y1, y2 = max(y1, 0), min(y2, image_size[1])
            invalid = False
        width, height = x2 - x1, y2 - y1

        if copy:
            shape = (height, width) + ((channels, ) if channels > 1 else ())
            box = np.zeros(shape, dtype=image.dtype)
            slice_box0 = slice(max(-y1, 0), height-max(y2-image_size[1], 0))
            slice_box1 = slice(max(-x1, 0), width-max(x2-image_size[0], 0))
            slice_image0 = slice(max(y1, 0), min(y2, image_size[1]))
            slice_image1 = slice(max(x1, 0), min(x2, image_size[0]))
            LOG.debug("Extracting[%s]: image[%s, %s] -> box[%s, %s]", self,
                      slice_image0, slice_image1, slice_box0, slice_box1)
            box[slice_box0, slice_box1] = image[slice_image0, slice_image1]
        else:
            box = image[y1:y2, x1:x2]

        return box

    def __str__(self) -> str:
        # return (f"BoundingBox at ({self.x}, {self.y})"
        #         f" of size {self.width} x {self.height}")
        return (f"BoundingBox from ({self.x1}, {self.y1})"
                f" to ({self.x2}, {self.y2})")


class Region:
    """A region in an image, optionally annotated with attributes.

    Attributes
    ----------
    _location:
        The location of the region. This can be a :py:class:`BoundingBox`
        or any other description of a location (a contour, etc.).

    _attributes: dict
        A dictionary with further attributes describing the region,
        e.g., a label.
    """

    _location = None
    _atributes = None

    color_min_confidence = np.asarray((255., 0., 0.))  # red
    color_max_confidence = np.asarray((0., 255., 0.))  # green

    def __init__(self, location, **attributes):
        self._location = location
        self._attributes = attributes

    def __str__(self) -> str:
        return f"{self._location} with {len(self._attributes)} attributes"

    def __contains__(self, point) -> bool:
        return point in self._location

    def __getattr__(self, name: str):
        if name in self._attributes:
            return self._attributes[name]
        raise AttributeError(f"Region has no attribute '{name}'. Valid "
                             f"attributes are: {self._attributes.keys()}")

    def __len__(self) -> str:
        return len(self._attributes)

    @property
    def location(self):
        """The :py:class:`Location` describing this :py:class:`Region`.
        """
        return self._location

    def mark_image(self, image: Imagelike, color: Tuple = None):
        """Mark this :py:class:`region` in a given image.

        Arguments
        ---------
        image:
            The image into which the region is to be marked.
        color:
            The color to be used for marking.
        """
        # FIXME[concept]: how to proceed for images that can not (easily)
        # be modified in place (e.g. filename/URL) -> should we rather
        # return the marked image?
        if color is None and 'confidence' in self._attributes:
            confidence = max(0, min(1.0, self._attributes['confidence']))
            color = ((1-confidence) * self.color_min_confidence +
                     confidence * self.color_max_confidence)
            color = tuple(color.astype(np.uint8))
        self._location.mark_image(image, color=color)

    def extract_from_image(self, image: Imagelike) -> np.ndarray:
        """Extract this :py:class:`Region` from a given image.

        Arguments
        ---------
        image:
            The image from the the region is to be extracted.

        Result
        ------
        patch:
            A numpy array (`dtype=np.uint8`) containing the extracted
            region.
        """
        return self._location.extract_from_image(image)

    def scale(self, factor) -> None:
        """Scale this region by a given factor.

        Arguments
        ---------
        factor:
            The scaling factor. This can either be a float, or a pair
            of floats in which case the first number is the horizontal (x)
            scaling factor and the second numger is the vertical (y)
            scaling factor.
        """
        if self._location is not None:
            self._location.scale(factor)
