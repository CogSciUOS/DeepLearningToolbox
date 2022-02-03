"""Abstract base class for tools working on images.
"""


# standard imports
from typing import Tuple
from threading import Thread
import logging

# third party imports
import numpy as np

# toolbox imports
from .tool import Tool
from ..base.data import Data
from ..base.image import Image, Imagelike, ImageGenerator
from ..util.image import imscale

# logging
LOG = logging.getLogger(__name__)


# FIXME[design]: there is also a class dltb.base.image.ImageTool
# -> it would make sense to combine these classes
class ImageTool(Tool):
    # pylint: disable=abstract-method
    """Abstract base class for tools that operate on images.
    Several convenience methods are provided by this class.
    """
    internal_arguments: Tuple[str] = ('image', )

    def __init__(self, size: Tuple[int, int] = None,
                 max_size: Tuple[int, int] = None,
                 min_size: Tuple[int, int] = None,
                 resize_policy: str = 'error', **kwargs) -> None:
        """
        Arguments
        ---------
        size:
            The exact size an image is expected to have for processing
            with this :py:class:`ImageTool`.
        min_size:
            The minimal image size processable by this :py:class:`ImageTool`.
        min_size:
            The maximal image size processable by this :py:class:`ImageTool`.
        resize_policy:
            The resize policy to adapt, if the aspect ratio of the image
            does not fit to this tool. Options are: 'pad' = pad the
            borders on the shorter side (with 0s), 'crop' = cut away
            some parts from the longer side, 'distort' = use different
            scaling along horizontal and vertical axis, resulting
            in a change of aspect ratio, 'error' = raise a ValueError.
        """
        super().__init__(**kwargs)
        if size is not None:
            if max_size is not None:
                LOG.warning("Argument 'max_size' (=%s) is ignored "
                            "when 'size' (=%s) is given.", max_size, size)
            if min_size is not None:
                LOG.warning("Argument 'min_size' (=%s) is ignored "
                            "when 'size' (=%s) is given.", min_size, size)
            self._max_size = size
            self._min_size = size
        else:
            if (min_size is not None and max_size is not None and
                    (min_size[0] > max_size[0] or min_size[1] > max_size[1])):
                raise ValueError(f"Minimal size (={min_size}) is larger than"
                                 f"maximal size (={max_size}).")
            self._max_size = max_size
            self._min_size = min_size

        self._resize_policy = resize_policy

    @property
    def input_size(self) -> Tuple[int, int]:
        """Get the input size on which this ImageTool operates.
        """
        return self._min_size

    def fit_image(self, image: Imagelike, policy: str = None) -> np.ndarray:
        """Resize the image to be suitable for processing with
        this :py:class:`ImageTool`.
        """
        size = image.shape[:1:-1]

        if self._min_size is not None:
            min_scale_x = max(self._min_size[0] / size[0], 1.)
            min_scale_y = max(self._min_size[1] / size[1], 1.)
            min_scale = max(min_scale_x, min_scale_y)
        else:
            min_scale = 0.0

        if self._max_size is not None:
            max_scale_x = min(self._max_size[0] / size[0], 1.)
            max_scale_y = min(self._max_size[1] / size[1], 1.)
            max_scale = min(max_scale_x, max_scale_y)
        else:
            max_scale = float('inf')

        if min_scale <= 1. <= max_scale:
            # The image fits into the desired size
            return image

        if min_scale <= max_scale:
            # Choose the scale closest to 1.
            scale = max_scale if max_scale < 1. else min_scale
            return imscale(image, scale=scale)

        # Here max_scale < min_scale, that is there is no way to fit
        # the image to a valid size and maintain the aspect ratio.
        # Different strategies can be applied here.
        if self._resize_policy == 'error':
            ValueError(f"Cannot fit image of size {size} into the "
                       f"acceptable size for procssing with {self} "
                       f"(min_size={self._min_size}, "
                       f"max_size={self._max_size}).")

        if self._resize_policy == 'pad':
            scale_x = min_scale
            scale_y = min_scale
        elif self._resize_policy == 'crop':
            scale_x = max_scale
            scale_y = max_scale
        else:  # if self._resize_policy == 'distort':
            # FIXME[todo]: find optimal scale (minimal distortion)
            # (this should be a compromise between minimal and
            # maximal scaling)
            # FIXME[hack]
            scale_x = min_scale_x if min_scale_x < 1. else max_scale_x
            scale_y = min_scale_y if min_scale_y < 1. else max_scale_y

        scaled_image = imscale(image, scale=(scale_x, scale_y))

        if self._resize_policy == 'pad':
            # FIXME[todo]: do padding
            padded_image = scaled_image
            return padded_image

        if self._resize_policy == 'pad':
            # FIXME[todo]: do cropping
            cropped_image = scaled_image
            return cropped_image

        return scaled_image

    def _preprocess(self, image: Imagelike, *args, **kwargs) -> Data:
        """Preprocessing of the image argument.  This will
        add some image specific properties to the context:

        input_image:
            the original image as numpy array
        input_size:
            the size of the original input image.

        image:
            the preprocessed image as numpy array
        size:
            the size of the preprocessed array

        Arguments
        ---------
        image:

        *args, **kwargs:
            Further arguments to be preprocessed by super classes.

        Results
        -------
        context:
            The context in which the input image and the preprocessed
            image are stored.
        """
        context = super()._preprocess(*args, **kwargs)

        # get the input image as numpy array
        array = Image.as_array(image, dtype=np.uint8)

        # we store the original image, as some tools may use it
        # for producing some output, e.g., painting bounding boxes
        # or heat maps.
        context.add_attribute('input_image', array)
        context.add_attribute('input_size', array.shape[:-1:-1])

        # now do the actual image preprocessing and stro results
        # as 'image' and 'size'
        array = self.image_to_internal(array)
        if hasattr(array, 'shape'):
            context.add_attribute('size', array.shape[:-1:-1])
        context.add_attribute('image', array)

        return context

    def _image_to_internal(self, image: Imagelike) -> np.ndarray:
        """Preprocess a single image provided as a numpy array.
        """
        # obtain array representation of the image
        image = Image.as_array(image)

        # resizing the image
        if self._min_size is not None or self._max_size is not None:
            image = self.fit_image(image)

        return image

    # FIXME[todo] names:
    #   image_region  <- region_of_image  [BoundingBox]
    #   image_size <- size_of_image       [Size]
    #   image_patch <-                    [np.ndarray]

    # FIXME[todo] -> BoundingBox
    def region_of_image(self, image: Imagelike) -> Tuple[int, int, int, int]:
        """The region of the given image that would be processed
        by the tool.

        Result
        ------
        left (x1): int
        top (y1): int
        right (x2): int
        bottom (y2): int
        """
        image = Image.as_array(image)
        return (40, 40, image.shape[1]-40, image.shape[0]-40)  # FIXME[hack]

    def size_of_image(self, image: Imagelike) -> Tuple[int, int]:
        """The size to which the region of interest of the image
        will be transformed when fed to this tool

        Result
        ------
        width: int
        height: int
        """
        return (100, 100)  # FIXME[hack]


# FIXME[todo]: integrate this in the framework - combine with IterativeTool
class IterativeImageTool(ImageGenerator):
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
