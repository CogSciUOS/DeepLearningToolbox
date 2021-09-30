"""Interface to the Scikit-Image (skimage) library.
"""
# standard imports
import logging

# third party imports
import numpy as np
from skimage.transform import resize, warp, SimilarityTransform

# toolbox imports
from dltb.base.image import Image, Imagelike, ImageResizer, ImageWarper
from dltb.base.image import Size, Sizelike

# logging
LOG = logging.getLogger(__name__)


class ImageUtils(ImageResizer, ImageWarper):
    """Implementation of several image utilities, including resizing and
    warping.

    """

    def resize(self, image: Imagelike, size=(640, 360)) -> np.ndarray:
        """Resize the frame to a smaller resolution to save computation cost.
        """
        # note: skimage.transform.resize takes on output_shape, not a size!
        # in the output_shape the number of channels is optional.
        output_shape = size[::-1]
        image = Image.as_array(image)
        resized = resize(image, output_shape, preserve_range=True)
        resized = resized.astype(image.dtype)
        return resized

    @staticmethod
    def warp(image: Imagelike, transformation: np.ndarray,
             size: Sizelike) -> np.ndarray:
        """Warp an image by applying a transformation.
        """
        image = Image.as_array(image)
        size = Size(size)
        output_shape = (size[1], size[0])
        # further argument: order (int, optional):
        #    The order of interpolation. The order has to be in the range 0-5:
        #        0: Nearest-neighbor
        #        1: Bi-linear (default)
        #        2: Bi-quadratic
        #        3: Bi-cubic
        #        4: Bi-quartic
        #        5: Bi-quintic
        warped = warp(image, transformation, output_shape=output_shape,
                      preserve_range=True)
        warped = warped.astype(image.dtype)
        return warped

    @staticmethod
    def compute_transformation(points: np.ndarray,
                               reference: np.ndarray) -> np.ndarray:
        """Obtain a tranformation for aligning key points to
        reference positions

        Arguments
        ---------
        points:
            A sequence of points to be mapped onto the reference points,
            given as (x,y) coordinates
        reference:
            A sequence with the same number of points serving as reference
            points to which `points` should be moved.

        """
        transformation = SimilarityTransform()
        transformation.estimate(reference, points)
        # transform.params: 3x3 matrix, projective coordinates,
        # last row [0,0,1]
        return transformation
