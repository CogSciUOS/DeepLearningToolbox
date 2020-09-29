"""Abstract base class for tools working on images.
"""


# standard imports
from typing import Tuple
import logging

# third party imports
import numpy as np

# toolbox imports
from .tool import Tool
from ..base.data import Data
from ..base.image import Image, Imagelike
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
        array = Image.as_array(image)
        data = super()._preprocess(self, array, *args, **kwargs)
        data.add_attribute('image', array)
        if self._min_size is not None or self._max_size is not None:
            data.add_attribute('scaled', self.fit_image(array))
        return data
