"""Interface to the Scikit-Image (skimage) library.
"""
# standard imports
import logging

# third party imports
import numpy as np
from skimage.transform import resize

# toolbox imports
from ..base.image import Image, Imagelike, ImageResizer

# logging
LOG = logging.getLogger(__name__)


class ImageUtil(ImageResizer):

    def resize(self, image: Imagelike, size=(640, 360)) -> np.ndarray:
        """Resize the frame to a smaller resolution to save computation cost.
        """
        # note: skimage.transform.resize takes on output_shape, not a size!
        # in the output_shape the number of channels is optional.
        output_shape = size[::-1]
        image = Image.as_array(image)
        result = resize(image, output_shape, preserve_range=True)
        result = result.astype(image.dtype)
        return result
