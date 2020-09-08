"""Utitlity image functions to be used by different parts of the Toolbox.
This module should only depend on standard imports and third party packages
but should not use any toolbox functionality.
"""

# standard imports
import sys
import logging

# third party imports
import numpy as np

# logging
LOG = logging.getLogger(__name__)


# There are several Python packages around that provide function for
# image I/O and image manipulation:
#
# * imageio:
#   several image I/O functions, including video and webcam access
#   Python 3.5+
#   https://imageio.github.io/
#   https://github.com/imageio/imageio
#   https://imageio.readthedocs.io/en/stable/
#   Version 2.8.0 - February 2020
#
# * cv2 (OpenCV):
#   functionality for image I/O, including video and webcam access (via ffmpeg)
#   resizing, drawing, ...
#   https://pypi.org/project/opencv-python
#   Version 4.2.0.32 - Februar 2020
#
# * scikit-image:
#   Image processing algorithms for SciPy, including IO, morphology, filtering, warping, color manipulation, object detection, etc.
#   https://scikit-image.org/
#   https://pypi.org/project/scikit-image/
#   Version 0.16.2 - October 2019
#
# * mahotas
#   Computer Vision in Python. It includes many algorithms implemented
#   in C++ for speed while operating in numpy arrays and with a very
#   clean Python interface.
#   https://github.com/luispedro/mahotas
#   https://mahotas.readthedocs.io/en/latest/
#   Version 1.4.8 - October 2019
#
# * imutils: (by Adrian Rosebrock from PyImageSearch)
#   A series of convenience functions to make basic image processing
#   functions such as translation, rotation, resizing,
#   skeletonization, displaying Matplotlib images, sorting contours,
#   detecting edges, and much more easier with OpenCV
#   https://github.com/jrosebr1/imutils
#   Version 0.5.3 - August 2019
#
# * Pillow/PIL
#   Pillow is the friendly PIL fork by Alex Clark and Contributors.
#   PIL is the Python Imaging Library by Fredrik Lundh and Contributors.
#   https://pypi.org/project/Pillow/
#   Version 7.1.0 - April 2020



try:
    from imageio import imread, imwrite
    LOG.info("using imread, imwrite from imageio")
except ImportError:
    try:
        from scipy.misc import imread, imsave as imwrite
        LOG.info("using imread, imsave as imwrite from scipy.misc")
    except ImportError:
        try:
            from matplotlib.pyplot import imread, imsave as imwrite
            LOG.info("using imread, imsave as imwrite from matplotlib.pyplot")
        except ImportError:
            # FIXME[hack]: better strategy to inform on missing modules
            explanation = ("Could not find any module providing 'imread'. "
                           "At least one such module is required "
                           "(e.g. imageio, scipy, matplotlib).")
            LOG.fatal(explanation)
            sys.exit(1)
        # maybe also cv2, but convert the colorchannels


class Location:

    def __init__(self, points) -> None:
        pass

    def mark_image(self, image, color=(1,0,0)):
        raise NotImplementedError(f"Location {self.__class__.__name__} "
                                  f"does not provide a method for marking "
                                  f"an image.")

    def scale(self, factor):
        raise NotImplementedError(f"Location {self.__class__.__name__} "
                                  f"does not provide a method for scaling.")


def grayscaleNormalized(array):
    """Convert a float array to 8bit grayscale

    Parameters
    ----------
    array: np.ndarray
        Array of 2/3 dimensions and numeric dtype.
        In case of 3 dimensions, the image set is normalized globally.

    Returns
    -------
    np.ndarray
        Array mapped to [0,255]

    """
    import numpy as np

    # normalization (values should be between 0 and 1)
    min_value = array.min()
    max_value = array.max()
    div = max(max_value - min_value, 1)
    return (((array - min_value) / div) * 255).astype(np.uint8)

class PointsBasedLocation:

    _points: np.ndarray

    def __init__(self, points: np.ndarray) -> None:
        super().__init__()
        self._points = points

    def mark_image(self, image, color=(1,0,0)):
        for p in self._points:
            image[max(p[1]-1,0):min(p[1]+1,image.shape[0]),
                  max(p[0]-1,0):min(p[0]+1,image.shape[1])] = color

    def scale(self, factor) -> None:
        """Scale the :py:class:`Location`.

        Arguments
        ---------
        factor:
            The scaling factor. This can either be a float, or a pair
            of floats in which case the first number is the horizontal (x)
            scaling factor and the second numger is the vertical (y)
            scaling factor.
        """
        self._points *= factor

    @property
    def points(self):
        return self._points

    def __len__(self):
        return len(self._points)

    
class Landmarks(PointsBasedLocation):

    def __len__(self) -> int:
        return 0 if self._points is None else len(self._points) 


class BoundingBox(PointsBasedLocation):

    def __init__(self, x1=None, y1=None, x2=None, y2=None,
                 x=None, y=None, width=None, height=None) -> None:
        super().__init__(np.ndarray((2,2)))
        if x1 is not None:
            self.x1 = x1
        elif x is not None:
            self.x1 = x

        if y1 is not None:
            self.y1 = y1
        elif y is not None:
            self.y1 = y

        if x2 is not None:
            self.x2 = x2
        elif width is not None:
            self.width = width

        if y2 is not None:
            self.y2 = y2
        elif height is not None:
            self.height = height

    @property
    def x1(self):
        return self._points[0,0]
    
    @x1.setter
    def x1(self, x1):
        self._points[0,0] = x1
        
    @property
    def y1(self):
        return self._points[0,1]
    
    @y1.setter
    def y1(self, y1):
        self._points[0,1] = y1
        
    @property
    def x2(self):
        return self._points[1,0]
    
    @x2.setter
    def x2(self, x2):
        self._points[1,0] = x2

    @property
    def y2(self):
        return self._points[1,1]
    
    @y2.setter
    def y2(self, y2):
        self._points[1,1] = y2
        
    @property
    def x(self):
        return self.x1

    @x.setter
    def x(self, x):
        self.x1 = x

    @property
    def y(self):
        return self.y1

    @y.setter
    def y(self, y):
        self.y1 = y

    @property
    def width(self):
        return self.x2 - self.x1

    @width.setter
    def width(self, width):
        self.x2 = self.x1 + width

    @property
    def height(self):
        return self.y2 - self.y1

    @height.setter
    def height(self, height):
        self.y2 = self.y1 + height

    def mark_image(self, image: np.ndarray, color=None) -> None:
        color = color or (0, 255, 0)
        size = image.shape[1::-1]
        thickness = max(1, max(size)//300)
        t1 = thickness//2
        t2 = (thickness+1)//2
        x1 = max(int(self.x1), t2)
        y1 = max(int(self.y1), t2)
        x2 = min(int(self.x2), size[0]-t1)
        y2 = min(int(self.y2), size[1]-t1)
        print(f"mark_image[{self}]: image size={size}"
              f"shape={image.shape}, {image.dtype}:"
              f"{image.min()}-{image.max()}, box:({x1}, {y1}) - ({x2}, {y2})")
        for offset in range(-t2, t1):
            image[(y1+offset, y2+offset), x1:x2] = color
            image[y1:y2, (x1+offset, x2+offset)] = color

    def extract(self, image, padding: bool=True,
                copy: bool=None) -> np.ndarray:
        """Extract the region described by the bounding box from an image.
        """
        image_size = image.shape[1::-1]
        channels = 1 if image.ndim < 3 else image[2]

        x1, x2 = int(self.x1), int(self.x2)
        y1, y2 = int(self.y1), int(self.y2)
        invalid = (x1 < 0 or x2 > image_size[0] or
                   y1 < 0 or y2 > image_size[1])

        if invalid and padding:
            copy = True
        else:
            # no padding: resize bounding box to become valid
            x1, x2 = max(x1, 0), min(x2, image_size[0])
            y1, y2 = max(y1, 0), min(y2, image_size[1])
            invalid = False
        width, height = x2 - x1, y2 - y1

        if copy:
            shape = (width, height) + ((channels, ) if channels > 1 else ())
            box = np.zeros(shape, type=image.dtype)
            box[-min(y1, 0):height-max(y2, image_size[1]),
                -min(x1, 0):height-max(x2, image_size[0])] = \
                image[max(y1, 0):min(y2, image_size[1]),
                      max(x1, 0):min(x2, image_size[0])]
        else:
            box = image[y1:y2, x1:x2]

        return box

    def __str__(self) -> str:
        # return (f"BoundingBox at ({self.x}, {self.y})"
        #         f" of size {self.width} x {self.height}")
        return (f"BoundingBox from ({self.x1}, {self.y1})"
                f" to ({self.x2}, {self.y2})")


class Region:
    """A region in an image, optionally annotated with attributes.

    Attributes
    ----------
    _location:
        The location of the region. This can be a :py:class:`BoundingBox`
        or any other description of a location (a contour, etc.).

    _attributes: dict
        A dictionary with further attributes describing the region,
        e.g., a label.
    """

    _location = None
    _atributes = None

    def __init__(self, location, **attributes):
        self._location = location
        self._attributes = attributes

    @property
    def location(self):
        return self._location

    def mark_image(self, image, color=None):
        self._location.mark_image(image, color=color)

    def scale(self, factor) -> None:
        """Scale this region by a given factor.

        Arguments
        ---------
        factor:
            The scaling factor. This can either be a float, or a pair
            of floats in which case the first number is the horizontal (x)
            scaling factor and the second numger is the vertical (y)
            scaling factor.
        """
        if self._location is not None:
            self._location.scale(factor)
