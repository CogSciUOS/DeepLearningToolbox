"""Transformations to be applied to an image.
"""
# standard imports
from typing import Tuple
from random import randint

# third-party imports
import numpy as np

# toolbox imports
from ..implementation import Implementable
from .types import Size, Sizelike
from .types2 import BoundingBox
from .image import Image, Imagelike


class ImageResizer(Implementable):
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

    @staticmethod
    def crop(image: Imagelike, size: Sizelike,
             mode: str = 'center',  # or 'random'
             position: Tuple[int, int] = None,
             copy: bool = True, **_kwargs) -> np.ndarray:
        """Crop an :py:class:`Image` to a given size.

        If no position is provided, a center crop will be performed.
        """
        image = Image.as_array(image)
        new_size = Size(size)
        old_size = image.shape[2::-1]

        if position is None:
            if mode == 'random':
                x_range = old_size[0] - new_size[0]
                y_range = old_size[1] - new_size[1]
                position = (randint(min(x_range, 0), max(x_range, 0)),
                            randint(min(y_range, 0), max(y_range, 0)))
            else:  # mode == 'center':
                center = old_size[0]//2, old_size[1]//2
                position = (center[0] - new_size.width//2,
                            center[1] - new_size.height//2)

        bounding_box = BoundingBox(x1=position[0], y1=position[1],
                                   width=new_size[0], height=new_size[1])
        return bounding_box.extract_from_image(image, copy=copy)


class ImageWarper(Implementable):
    """An :py:class:`ImageWarper` can warp an image (apply some affine
    transformation).
    """

    @staticmethod
    def warp(image: Imagelike, transformation: np.ndarray,
             size: Size) -> np.ndarray:
        """Warp an image by applying a transformation.

        To be implemented by subclasses.
        """

    @staticmethod
    def compute_transformation(points: np.ndarray,
                               reference: np.ndarray) -> np.ndarray:
        """Obtain a tranformation for aligning key points to
        reference positions

        To be implemented by subclasses.

        Arguments
        ---------
        points:
            A sequence of points to be mapped onto the reference points,
            given as (x,y) coordinates
        reference:
            A sequence with the same number of points serving as reference
            points to which `points` should be moved.

        Result
        ------
        transformation:
            A affine transformation matrix.  This is a 2x3 matrix,
            allowing to compute [x',y'] = matrix * [x,y,1].

        Note
        ----
        Affine transformations are more general than similarity
        transformations, which can always be decomposed into a
        combination of scaling, rotating, and translating.  General
        affine tansformations can not be decomposed in this way.
        The affine transformation matrix contains the following entries:
        ```
        cos(theta) * s   -sin(theta) * s    t_x
        sin(theta) * s    cos(theta) * s    t_y
        ```
        with theta being the rotation angle, s the scaling factor and
        t the translation.
        """

    @classmethod
    def align(cls, image: Imagelike, points, reference,
              size: Sizelike) -> np.ndarray:
        """Align an image by applying an (affine) transformation that maps
        key points to reference positions.

        Arguments
        ---------
        image:
            The image to align.
        points:
            A sequence of points to be mapped onto the reference points,
            given as (x,y) coordinates
        reference:
            A sequence with the same number of points serving as reference
            points to which `points` should be moved.
        size:
            The size of the resulting image.

        Result
        ------
        aligned:
            The aligned image.
        """
        transformation = cls.compute_transformation(points, reference)
        return cls.warp(image, transformation, size)
