"""Interface to the Scikit-Image (skimage) library.
"""
# standard imports
import logging

# third party imports
import numpy as np
from skimage.transform import resize

# toolbox imports
from ..base import image

# logging
LOG = logging.getLogger(__name__)


class ImageUtil(image.ImageResizer):

    def resize(self, image: np.ndarray, size=(640, 360)) -> np.ndarray:
        """Resize the frame to a smaller resolution to save computation cost.
        """
        # note: skimage.transform.resize takes on output_shape, not a size!
        # in the output_shape the number of channels is optional.
        output_shape = size[::-1]
        result = resize(image, output_shape, preserve_range=True)
        result = result.astype(image.dtype)
        return result
