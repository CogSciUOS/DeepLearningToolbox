"""The `Image` class as a specific type of `Data` class.
"""

# standard imports
from typing import Union, Tuple, Dict, Any, Optional, Iterable
from typing import Literal, overload
from abc import abstractmethod, ABC
from pathlib import Path
from threading import Thread
import inspect
import math

# third-party imports
import numpy as np

# toolbox imports
from ..observer import Observable
from ..implementation import Implementable
from ..reuse import Reusable, reusable
from ..data import Data, DataDict, Datalike, BatchDataItem
from ._base import LOG, Imagelike, ImageConverter
from .types import Size, Sizelike, Colorspace


class ImageProperties:
    """A collection of properties that am image may possess.

    Arguments
    ---------
    size:
        The size of the image.
    channels:
        The number of (color) channels. Default is ``3`` for RGB images.
    """

    def __init__(self, size: Optional[Sizelike] = None,
                 channels: int = 3) -> None:
        self._size = size
        self._channels = channels

    @property
    def size(self) -> Optional[Size]:
        """Size of the image.
        """
        return self._size

    @property
    def channels(self) -> int:
        """Number of (color) channels.
        """
        return self._channels


class Image(DataDict):
    """A collection of image related functions.


    Arguments
    ---------
    image/array:
        An `Imagelike` object from which the `Image` is to be initialized.
    init:
        A list of attributes that should automatically be filled during
        initialization.
    """

    # _types: a dictionary mapping type names to known image types.
    # Images can be represented in any of these types. The
    # representation for type 'img_type' is accessible as
    # `image.img_type`.
    # New image types should be added by calling Image.add_type()
    _image_types = {
        'array': np.ndarray,
    }

    # dictionary of registered image converters.
    # the dictionary maps the target type to a list of possible
    # converters that yield that type.
    _converters = {
        'array': [
            (np.ndarray, lambda array, _copy: array),
            (Data, lambda data, _copy: data.array),
            (BatchDataItem, lambda data, _copy: data.array)
        ],
        'image': [
            (np.ndarray, Data)
        ]
    }

    @classmethod
    def _type_name(cls, image_type: type) -> str:
        for name_type in cls._image_types.items():
            if image_type is name_type[1]:
                return name_type[0]
        raise TypeError(f"Not a valid image type: {image_type}")

    @classmethod
    def add_type(cls, image_type: type, name: Optional[str] = None) -> None:
        """Add a new image type.  Image types can be used wherever
        an `Imagelike` argument is expected.

        Arguments
        ---------
        image_type:
            The new image type.
        name:
            A name for that type.
        """
        if name is None:
            name = image_type.__name__
        if cls._image_types.get(name, image_type) is not image_type:
            raise TypeError(f"Inconsistent image types for '{name}':"
                            f"new type {image_type} is inconsistent with "
                            f"registered type {cls._image_types[name]}")
        cls._image_types[name] = image_type

        # pylint: disable=global-statement
        global Imagelike
        Imagelike = Union[tuple(cls._image_types.values())]

    @classmethod
    def add_converter(cls, converter: ImageConverter,
                      source: Union[str, type, type(None)] = None,
                      target: Union[str, type, type(None)] = None) -> None:
        """Register a new image converter. An image converter is
        a function, that can convert a given image into another
        format.

        Arguments
        ---------
        convert:
            The actual converter function. This function takes two
            arguments: `image` is the image to convert and `bool` is
            a flag indicating if the image data should be copied.
        source:
            The input type of the converter, that is the type of
            its first argument of the `convert` function.
        target:
            The output format. This can be `Image` (the converter
            produces an instance of `Image`) or `array` (a numpy array),
            or another string identifying a third party format, if
            available.
        """
        argspec = inspect.getfullargspec(converter)
        # signature = inspect.signature(converter)
        # hints = typing.get_type_hints(converter)
        if source is None:
            first_arg_name = argspec.args[0]
            source = argspec.annotations.get(first_arg_name, 'image')
        if target is None:
            target = argspec.annotations.get('return', 'image')

        if not isinstance(source, str):
            source = cls._type_name(source)
        if not isinstance(target, str):
            target = cls._type_name(target)

        if source == target:
            raise TypeError(f"Converter source type ({source}) and"
                            f"target type ({target}) should be different.")

        source = cls._image_types[source]
        if target not in cls._converters:
            cls._converters[target] = [(source, converter)]
        else:
            cls._converters[target].append((source, converter))

    @classmethod
    def supported_formats(cls) -> Iterable[str]:
        """The names of supported image formats.
        """
        return cls._converters.keys()

    @classmethod
    def supported_types(cls) -> Iterable[type]:
        """The supoorted image types.
        """
        return cls._converters.values()

    @classmethod
    def as_type(cls, image: Imagelike, target: Union[str, type],
                copy: Optional[bool] = None) -> Imagelike:
        """Convert a given image.
        """
        target_name = \
            target if isinstance(target, str) else cls._type_name(target)

        for source_class, converter in cls._converters[target_name]:
            if isinstance(image, source_class):
                LOG.debug("Using image converter for type %s (copy=%s)",
                          type(image), copy)
                image = converter(image, copy)
                copy = copy and None  # do not copy again
                break
        else:
            raise ValueError("No converter available to convert "
                             f"{type(image)} to {target}")
        return image

    @classmethod
    def as_array(cls, data: Optional[Datalike] = None,
                 copy: Optional[bool] = None,
                 dtype: Optional[type] = None,
                 image: Optional[Imagelike] = None,
                 colorspace: Colorspace = None) -> np.ndarray:
        # pylint: disable=too-many-arguments
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
            if the original data is to be returned.
            The default value `None` indicates that copying
            should only happen if unavoidable.
        dtype:
            Numpy datatype, e.g., numpy.float32.
        colorspace: Colorspace
            The colorspace in which the pixels in the resulting
            array are encoded.  If no colorspace is given, or
            if the colorspace of the input image Image is unknown,
            no color conversion is performed.
        """
        if image is None:
            image = data
        for source_class, converter in cls._converters['array']:
            if isinstance(image, source_class):
                LOG.debug("Using image converter for type %s (copy=%s)",
                          type(image), copy)
                image = converter(image, copy)
                copy = copy and None  # do not copy again
                break
        else:
            if isinstance(image, Path):
                image = str(image)
            if isinstance(image, str):
                LOG.debug("Loading image '%s' using imread.", image)
                image = ImageReader.read(image)
                copy = copy and None  # do not copy again
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

    @staticmethod
    def as_shape(image: Imagelike) -> Tuple[int]:
        """Get the shape of an image-like object.
        """
        if isinstance(image, np.ndarray):
            return image.shape
        if isinstance(image, Image):
            return image.array.shape
        raise TypeError(f"Cannot determine shape of {type(image)}")

    # Image(array)  - inherited from Data
    # Image(imagelike) - nice to have, most natural
    # Image(image=imagelike)  - should also work
    def __new__(cls, image: Imagelike = None, array: np.ndarray = None,
                copy: bool = False,
                init: Optional[Any] = None, **kwargs) -> None:
        del init  # init not used here
        if isinstance(image, Image) and not copy:
            return image  # just reuse the given Image instance
        return super().__new__(cls, array=array, copy=copy, **kwargs)

    def __init__(self, image: Imagelike = None, array: np.ndarray = None,
                 copy: Optional[bool] = None,
                 init: Optional[Tuple[str,...]] = None,
                 **kwargs) -> None:
        if isinstance(image, Image) and not copy:
            return  # just reuse the given Image instance
        try:
            if image is not None:
                array = self.as_array(image, copy=copy)
        finally:
            # make sure super().__init__() is called even if
            # preparing the array fails. If ommitted, the object may
            # be in an incomplete state, causing problems at destruction.
            super().__init__(array=array, **kwargs)

        if isinstance(image, str):
            self.add_attribute('filename', image)

        # auto initialize attributes
        if init is not None:
            for attr in init:
                try:
                    getattr(self, attr)
                except AttributeError:
                    raise ValueError("Can not automatically initialize "
                                     f"attribute {attr}") from None

    def __getattr__(self, attr: str) -> Any:
        try:
            return super().__getattr__(attr)
        except AttributeError:
            pass

        if attr in self._converters:
            value = self.convert(attr)
            self.add_attribute(attr, value)
            return value

        raise AttributeError(f"{type(self)} object has no attribute '{attr}'")

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr == 'array' and value is not None:
            self.add_attribute('size',
                               Size(value.shape[1], value.shape[0]))
            if value.ndim == 3:
                self.add_attribute('channels', value.shape[-1])
        super().__setattr__(attr, value)

    def visualize(self, size=None) -> np.ndarray:
        """Provide a visualization of this image. The may be simply
        the image (in case of a single image)
        In case of a batch, it can be an image galery.
        """
        if not self.is_batch:
            return self.array

        # self is a batch of images: create a matrix showing all images.
        rows = int(math.sqrt(len(self)))
        columns = math.ceil(len(self) / rows)
        if size is None:
            size = (self[0].shape[1], self[0].shape[0])
        matrix = np.zeros((size[1]*rows, size[0]*columns, 3),
                          dtype=self[0].array.dtype)
        for idx, image in enumerate(self):
            column = idx % columns
            row = idx // columns
            image = ImageResizer.resize(image.array, size)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=2).repeat(3, axis=2)
            matrix[row*size[1]:(row+1)*size[1],
                   column*size[0]:(column+1)*size[0]] = image
        return matrix

    def size(self) -> Size:
        """The size of this image.
        """
        if self.has_attribute('array'):
            return Size(*self.shape[1::-1])

        raise TypeError("Unable to determine the image size.")


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

    def image_to_internal(self, image: Imagelike) -> np.ndarray:
        """Convert a given image into the internal imaga representation.

        Arguments
        ---------
        image:
            The imagelike object to be converted

        Result
        ------
        internal:
            The image in its internal representation.
        """
        # batch handling
        if isinstance(image, Data) and image.is_batch:
            result = np.ndarray((len(image), 227, 227, 3))
            for index, img in enumerate(image.array):
                result[index] = self._image_to_internal(img)
            return result

        if isinstance(image, list):
            result = np.ndarray((len(image), 227, 227, 3))
            for index, img in enumerate(image):
                result[index] = self._image_to_internal(img)
            return result

        # convert single image to batch of length 1
        image = self._image_to_internal(image)
        return image[np.newaxis]

    @abstractmethod
    def _image_to_internal(self, image: Imagelike) -> Any:
        "to be implemented by subclasses"

    @abstractmethod
    def internal_to_image(self, data: Any) -> Imagelike:
        "to be implemented by subclasses"


class ImageExtension(ImageAdapter, ABC):
    # pylint: disable=abstract-method
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
        ImageWriter.write(target, self(ImageReader.read(source)))

    def transform_data(self, image: Image,
                       target: str, source: str = None) -> None:
        """Apply image operator to an :py:class:`Image` data object.
        """
        image.add_attribute(target, value=self(image.get_attribute(source)))


class ImageGenerator(ImageObservable):
    # pylint: disable=too-few-public-methods
    """An image :py:class:`Generator` can generate images.

    There can be different modes of generation:
     * just generate
     * generate from some latent vector
     * generate based on some condition (e.g. class label, text, ...)

    Generation can occur in different modes
     * instantanous (call method and return image)
     * asynchronous/once (initiate generation process and get informed
       once the image generation process is completed)
     * asynchronous/continuos
    """

    # the following variables are only used in asynchronous mode
    _image: Image = None
    _thread: Thread = None

    @overload
    def generate_image(self, run: Literal[True], **kwargs) -> None: ...

    @overload
    def generate_image(self, run: Literal[False], **kwargs) -> Image: ...

    def generate_image(self, run: bool = False,
                       timing: Optional[float] = None,
                       **kwargs) -> Union[Image, type(None)]:
        """Generate an image and return it.
        """
        if not run:
            return self._generate_image(**kwargs)

        if self._thread is not None:
            raise RuntimeError("ImageGenerator is busy.")
        self._image = None
        if timing is None:
            self._thread = Thread(target=self._run_generate_image,
                                  kwargs=kwargs)
        else:
            self._thread = Thread(target=self._run_generate_image_sequence,
                                  args=(timing,), kwargs=kwargs)
        return None  # avoid inconsistent-return-statements

    def _run_generate_image(self, **kwargs) -> None:
        self._image = self._generate_image(**kwargs)
        self.notify_observers('image_changed')
        self._thread = None

    def _run_generate_image_sequence(self, **kwargs) -> None:
        """
        """
        for image in self._generate_image_sequence(**kwargs):
            self._image = image
            self.notify_observers('image_changed')
        self._thread = None

    def _generate_image_sequence(self, steps: Optional[int] = None,
                                 **kwargs) -> Iterable[Image]:
        if steps is not None:
            for _ in range(steps):
                yield self._generate_image(**kwargs)

    @abstractmethod
    def _generate_image(self, **kwargs) -> Image:
        """To be implemented by subclasses
        """


class ImageIO:
    # pylint: disable=too-few-public-methods
    """An abstract interface to read, write and display images.
    """


class ImageReader(ImageIO, Implementable, Reusable):
    """An :py:class:`ImageReader` can read images from file or URL.
    The :py:meth:`read` method is the central method of this class.
    """

    def __str__(self) -> str:
        return type(self).__module__ + '.' + type(self).__name__

    @reusable
    def read(self, filename: str, **kwargs) -> np.ndarray:
        # pylint: disable=bad-classmethod-argument
        """Read an image from a file or URL.
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to "
                                  "be an ImageReader, but does not implement "
                                  "the read method.")


class ImageWriter(ImageIO, Implementable, Reusable):
    """An :py:class:`ImageWriter` can write iamges to files or upload them
    to a given URL.  The :py:meth:`write` method is the central method
    of this class.

    """

    @reusable
    def write(self, filename: str, image: Imagelike, **kwargs) -> None:
        # pylint: disable=bad-classmethod-argument
        """Write an `image` to a file with the given `filename`.
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to "
                                  "be an ImageWriter, but does not implement "
                                  "the write method.")


class ImageResizer(Implementable, Reusable):
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

    @classmethod
    def _check_implementation(cls, name: str) -> bool:
        """Check if the given method was implmented by a subclass.
        """
        return getattr(cls, name) is getattr(ImageResizer, name)

    @reusable
    def resize(self, image: Imagelike,
               size: Size, **kwargs) -> Imagelike:
        """Resize an image to the given size.

        Arguments
        ---------
        image:
            The image to be scaled.
        size:
            The target size.
        """
        img = self._internalize(image)
        size = Size(size)
        if self._check_implementation('_resize'):
            result = self._resize(img, size, **kwargs)
        elif self._check_implementation('_scale'):
            image_size = image.shape[:2]
            scale = (size[0]/image_size[0], size[1]/image_size[1])
            result = self._scale(image, scale=scale, **kwargs)
        else:
            raise NotImplementedError(f"{type(self)} claims to be an "
                                      "ImageResizer, but does neither "
                                      "implement resize nor scale.")
        return self._externalize(result)

    @reusable
    def scale(self, image: np.ndarray,
              scale: Union[float, Tuple[float, float]],
              **kwargs) -> Imagelike:
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
        img = self._internalize(image)
        if isinstance(scale, float):
            scale = (scale, scale)
        if self._check_implementation('_scale'):
            result = self._resize(img, scale, **kwargs)
        elif self._check_implementation('_resize'):
            image_size = image.shape[:2]
            size = Size(int(image_size[0] * scale[0]),
                        int(image_size[1] * scale[1]))
            result = self._resize(image, size=size, **kwargs)
        else:
            raise NotImplementedError(f"{type(self)} claims to be an "
                                      "ImageResizer, but does neither "
                                      "implement resize nor scale.")
        return self._externalize(result)

    def _internalize(self, image: Imagelike) -> Any:
        """Convert the given image into the internal format.
        """
        return image

    def _externalize(self, image: Any) -> Imagelike:
        """Convert the internal format into the desired external format.
        """
        return image

    def _resize(self, image: Imagelike, size: Size, **_kwargs) -> Imagelike:
        """To be implemented by subclasses. It is sufficient to implement
        either _resize or _scale.
        """
        del size
        return image

    def _scale(self, image: Imagelike, scale: Tuple[float, float],
               **_kwargs) -> Imagelike:
        """To be implemented by subclasses. It is sufficient to implement
        either _resize or _scale.
        """
        del scale
        return image
