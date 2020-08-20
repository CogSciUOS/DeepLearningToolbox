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


class ImageUtils(image.ImageResizer):

    def resize(self, image: np.ndarray, size=(640, 360)) -> np.ndarray:
        """Resize the frame to a smaller resolution to save computation cost.
        """
        return resize(image, size)
