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
from typing import Union, List, Tuple
from threading import Thread

# third party imports
import numpy as np

# toolbox imports
from base.observer import Observable
from .data import Data
from .. import thirdparty


# Imagelike is intended to be everything that can be used as
# an image.
#
# np.ndarray:
#    The raw image data
# str:
#    A URL.
Imagelike = Union[np.ndarray, str]


class Image:
    """A collection of image related functions.
    """

    @staticmethod
    def as_array(image: Imagelike, copy: bool = False) -> np.ndarray:
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
        """
        # FIXME[hack]: local imports to avoid circular module dependencies ...
        from dltb.util.image import imread
        if isinstance(image, Data):
            image = image.array
        if isinstance(image, np.ndarray):
            return image.copy()
        if isinstance(image, str):
            return imread(image)
        raise NotImplementedError(f"Conversion of {type(image).__module__}."
                                  f"{type(image).__name__} to numpy.ndarray "
                                  "is not implemented")

    @staticmethod
    def as_data(image: Imagelike, copy: bool = False) -> 'Data':
        """Get image-like objec as :py:class:`Data` object.
        """
        if isinstance(image, Data) and not copy:
            return image

        array = Image.as_array(image, copy)
        data = Data(array)
        data.type = Data.TYPE_IMAGE
        if isinstance(image, str):
            data.add_attribute('url', image)
        return data


class ImageIO:
    """An abstract interface to read, write and display images.
    """

    def __init__(self, **kwargs):
        pass

    def __del__(self):
        pass


class ImageReader(ImageIO):
    """An :py:class:`ImageReader` can read iamges from file or URL.
    The :py:meth:`read` method is the central method of this class.
    """

    def __new__(cls, module: Union[str, List[str]] = None) -> 'ImageReader':
        if cls is ImageReader:
            cls = thirdparty.import_class('ImageReader', module=module)
        return super(ImageReader, cls).__new__(cls)

    def read(self, filename: str, **kwargs) -> np.ndarray:
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
            cls = thirdparty.import_class('ImageWriter', module=module)
        return super(ImageWriter, cls).__new__(cls)

    def write(self, filename: str, image: np.ndarray, **kwargs) -> None:
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
            cls = thirdparty.import_class('ImageResizer', module=module)
        return super(ImageResizer, cls).__new__(cls)

    def resize(self, image: np.ndarray, size, **kwargs) -> np.ndarray:
        raise NotImplementedError(f"{type(self)} claims to "
                                  "be an ImageResizer, but does not implement "
                                  "the resize method.")

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
        raise NotImplementedError(f"{type(self)} claims to "
                                  "be an ImageResizer, but does not implement "
                                  "the scale method.")

    def crop(self, image: np.ndarray, size, **_kwargs) -> np.ndarray:
        # FIXME[todo]: deal with sizes extending the original size
        # FIXME[todo]: allow center/random/position crop
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
        imwrite(target, self(imread(source)))

    def transform_data(self, image: Image,
                       target: str, source: str = None) -> None:
        """Apply image operator to an :py:class:`Image` data object.
        """
        image.add_attribute(target, value=self(image.get_attribute(source)))


class ImageTool(Observable, method='image_changed', changes={'image_changed'}):
    """Base class for tools that can iteratively create images:

    Design pattern 1 (observer)
    ---------------------------

    * loop
      - hold the current version of the image as a object state
      - loop over the following steps:
        1, perform one step of image creation
        2. notify observers

    Design pattern 2 (functional):
    ------------------------------

    * iterator for generating N (or unlimited) images

    for image in tool:
        do something with image

    for image in tool(30):


    Stateless inremental image create API:

    * next_image = tool.next(image)

    * bool tool.finished(image)

    """

    def __init__(self) -> None:
        super().__init__()
        self._step = 0
        self._image = None
        self._thread = None
        self._stop = True

    def __call__(self) -> np.ndarray:
        """The actual image generation. To be impolemented by subclasses.

        Result
        ------
        image: np.ndarray
            The next image generated in standard representation
            (RGB, uint8, 0-255).
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to "
                                  "be an ImageTool, but does not implement "
                                  "the __next__ method.")

    #
    # loop API (stateful)
    #

    @property
    def image(self) -> np.ndarray:
        """The current image provided by this :py:class:`ImageTool`
        in standard format (RGB, uint8, 0-255).
        """
        return self._image

    @property
    def step(self) -> int:
        """The current step performed by this :py:class:`ImageTool`
        """
        return self._step

    def next(self) -> None:
        """Do one step adapting the current image. This changes the
        property `image` and notifies observers that a new image is
        available.
        """
        self._image = self()
        self._step += 1
        self.change('image_changed')

    def __next__(self) -> np.ndarray:
        """Create a new image by doing the next step.

        Result
        ------
        image: np.ndarray
            The image in standard format (RGB, uint8, 0-255).
        """
        self.next()
        return self._image

    def loop(self, threaded: bool = False, **kwargs) -> None:
        if self.looping:
            raise RuntimeError("Object is already looping.")
        if threaded:
            self._thread = Thread(target=self._loop, kwargs=kwargs)
            self._thread.start()
        else:
            self._loop(**kwargs)

    def _loop(self, stop: int = None, steps: int = None) -> None:
        """Run an  loop

        Arguments
        ---------
        steps: int
            The number of steps to perform.

        stop: int
            The loop will stop once the internal step counter
            reaches this number. If no stop value is given
        """
        self._stop = False
        while not self._stop:
            try:
                self.next()
                if steps is not None:
                    steps -= 1
                    if steps <= 0:
                        self.stop()
                if stop is not None and self._step >= stop:
                    self.stop()

            except KeyboardInterrupt:
                self.stop()

    @property
    def looping(self) -> bool:
        return not self._stop

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def pause(self) -> None:
        self._stop = True


class ImageOptimizer(ImageTool):
    """An image optimizer can incrementally optimize an image, e.g., using
    some gradient-based optimization strategy.  The
    :py:class:`ImageOptimizer` provides an API for accessing image
    optimizers.

    Stateless vs. stateful optimizer
    --------------------------------
    An :py:class:`ImageOptimizer` has an internal state, the current
    version of the image.

    An :py:class:`ImageOptimizer` may also provide different loss
    values and evaluation metrics that may be .

    Internal optimizer
    ------------------

    An Image optimizer may employ some external engine to do the
    optimization.  In such a situation, the image may need to be
    converted into an internal representation prior to optimization,
    and the result has to be converted back into a standard image.

    """

    # FIXME[todo]: optimization values
    def __call__(self, image: np.ndarray) -> np.ndarray:
        internal_image = self._internalize(image)
        internal_result = self._internal_optimizer(internal_image)
        return self._externalize(internal_result)

    def __next__(self) -> np.ndarray:
        self._internal_image = self._internal_optimizer(self._internal_image)
        # FIXME[concept/design]: should we store and return the image?
        # - There may be observers!
        self._image = self._externalize(self._internal_image)
        return self._image


class ImageDisplay(ImageIO, ImageTool.Observer):
    """An `ImageDisplay` can display images.
    """

    def __new__(cls, module: Union[str, List[str]] = None,
                **kwargs) -> 'ImageDisplay':
        if cls is ImageDisplay:
            cls = thirdparty.import_class('ImageDisplay', module=module)
        return super(ImageDisplay, cls).__new__(cls)

    #
    # context manager
    #

    def __enter__(self) -> 'ImageDisplay':
        return self

    def __exit__(self, _exception_type, _exception_value, _traceback) -> None:
        pass # FIXME[todo]

    #
    # public interface
    #

    def show(self, image: Imagelike, wait_for_key: bool = False,
             timeout: float = None, **kwargs) -> None:
        """Display the given image.

        This method may optionally pause execution until to display
        the image, if the wait_for_key or timeout arguments are given.
        If both are given, the first one will stop pausing.

        Arguments
        ---------
        image: Imagelike
            The image to display. This may be a single image or a
            batch of images.
        wait_for_key: bool
            A flag indicating if the display should pause execution
            and wait or a key press.
        timeout: float
            Time in seconds to pause execution.
        """
        raise NotImplementedError(f"{type(self).__name__} claims to "
                                  "be an ImageDisplay, but does not implement "
                                  "the show method.")

    def image_changed(self, tool, change) -> None:
        self.show(tool.image)

    # FIXME[old/todo]:
    def run(self, tool):
        """Monitor the operation of a Processor. This will observe
        the processor and update the display whenever new data
        are available.
        """
        self.observe(tool, interests=ImageTool.Change('image_changed'))
        try:
            print("Starting thread")
            thread = Thread(target=tool.loop)
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

    @property
    def closed(self) -> bool:
        return False  # FIXME[hack]

    @property
    def active(self) -> bool:
        return True  # FIXME[hack]
