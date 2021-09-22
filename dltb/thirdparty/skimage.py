"""Interface to the Scikit-Image (skimage) library.
"""
# standard imports
import logging

# third party imports
import numpy as np
from skimage.transform import resize, warp, SimilarityTransform

# toolbox imports
from ..base.image import Image, Imagelike, ImageResizer
from ..tool.align import PointAligner

# logging
LOG = logging.getLogger(__name__)


class ImageUtil(ImageResizer, PointAligner):

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

    def align_points(self, image: Imagelike, points, size) -> np.ndarray:
        """Align an image by applying an (affine) transformation that maps
        source points to target points.

        Arguments
        ---------
        image:
            The image to align.
        points:
            A list of points to be mapped onto the reference points,
            given as (x,y) coordinates
        size:
            The size of the resulting image.

        Result
        ------
        aligned:
            The aligned image.
        """
        image = Image.as_array(image)

        transform = SimilarityTransform()
        transform.estimate(points, self._reference_points)
        # transform.params: 3x3 matrix, projective coordinates,
        # last row [0,0,1]
        aligned = warp(image, transform, output_shape=(size[1], size[0]))

        return aligned
