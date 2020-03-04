import sys
import logging

try:
    from imageio import imread, imwrite
except ImportError:
    try:
        from scipy.misc import imread, imsave as imwrite
    except ImportError:
        try:
            from matplotlib.pyplot import imread, imsave as imwrite
        except ImportError:
            # FIXME[hack]: better strategy to inform on missing modules
            explanation = ("Could not find any module providing 'imread'. "
                           "At least one such module is required "
                           "(e.g. imageio, scipy, matplotlib).")
            logging.fatal(explanation)
            sys.exit(1)
        # maybe also cv2, but convert the colorchannels

# FIXME[todo]: imresize:
#
# The documentation of scipy.misc.imresize says that imresize is
# deprecated! Use skimage.transform.resize instead. But it seems
# skimage.transform.resize gives different results from
# scipy.misc.imresize.
#  https://stackoverflow.com/questions/49374829/scipy-misc-imresize-deprecated-but-skimage-transform-resize-gives-different-resu
#
# -> Try using scipy.ndimage.interpolation.zoom()
#
# * cv2.resize(image,(width,height))
# * mahotas.imresize(img, nsize, order=3)
#   This function works in two ways: if nsize is a tuple or list of
#   integers, then the result will be of this size; otherwise, this
#   function behaves the same as mh.interpolate.zoom
#
# * mahotas.interpolate.zoom
# * imutils.resize

try:
    # rescale, resize, downscale_local_mean
    # image_rescaled = rescale(image, 0.25, anti_aliasing=False)
    # image_resized = resize(image, (image.shape[0] // 4, image.shape[1] // 4), anti_aliasing=True)
    # image_downscaled = downscale_local_mean(image, (4, 3))
    from skimage.transform import resize as imresize
except ImportError:
    try:
        from scipy.misc import imresize
    except ImportError:
        # FIXME[hack]: better strategy to inform on missing modules
        explanation = ("Could not find any module providing 'imresize'. "
                       "At least one such module is required "
                       "(e.g. skimage, scipy, cv2).")
        logging.fatal(explanation)
        sys.exit(1)

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
        
import numpy as np

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

    def mark_image(self, image, color=(1,0,0)):
        image[(self.y1,self.y2),self.x1:self.x2] = color
        image[self.y1:self.y2,(self.x1,self.x2)] = color

        
class Region:
    """A region in an image.

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

    def mark_image(self, image, color=(1.0, 0.0, 0.0)):
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
