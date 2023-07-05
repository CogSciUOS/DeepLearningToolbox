"""Additional types that require `Image` functionality

This separate modules was introduced to avoid circular imports.
 - image requires types like 'Size'
 - specific image operations in BoundingBox require Image.as_array

A better design may allow to make this module superflous.
"""
# standard imports
from typing import Union, Tuple, Optional, Any

# third-party imports
import numpy as np

# toolbox imports
from ._base import LOG
from .types import Size, Color
from .image import Imagelike, Image


class Location:
    """A :py:class:`Location` identifies an area in a two-dimensional
    space.  A typical location is a bounding box (realized by the
    subclass :py:class:`BoundingBox`), but this abstract definition
    also allows for alternative ways to describe a location.

    """

    def __init__(self, points) -> None:
        pass

    def __contains__(self, point) -> bool:
        """Checks if the given point lies in this :py:class:`Location`.

        To be implemented by subclasses.
        """

    def mark_image(self, image: Imagelike, color=(1, 0, 0)):
        """Mark this :py:class:`Location` in some given image.

        Arguments
        ---------
        image:
        """
        raise NotImplementedError(f"Location {self.__class__.__name__} "
                                  f"does not provide a method for marking "
                                  f"an image.")

    def extract_from_image(self, image: Imagelike) -> np.ndarray:
        """Extract this :py:class:`Location` from a given image.

        Arguments
        ---------
        image:
            The image from which the location is to be extracted.
        """
        raise NotImplementedError(f"Location {self.__class__.__name__} "
                                  f"does not provide a method for extraction "
                                  f"from an image.")

    def scale(self, factor: Union[float, Tuple[float, float]],
              reference: str = 'origin') -> None:
        """Scale this :py:class:`location` by the given factor.
        All coordinates  will be multiplied by this value.
        """
        raise NotImplementedError(f"Location {self.__class__.__name__} "
                                  f"does not provide a method for scaling.")


class PointsBasedLocation:
    """A :py:class:`PointsBasedLocation` is a :py:class:`Location`
    that can be described by points, like a polygon area, or more
    simple: a bounding box.

    Attributes
    ----------
    _points: np.ndarray
        An array of shape (n, 2), providing n points in form of (x, y)
        coordinates.
    """

    def __init__(self, points: np.ndarray) -> None:
        super().__init__()
        self._points = points

    def __contains__(self, point) -> bool:
        return ((self._points[:, 0].min() <= point[0] <=
                 self._points[:, 0].max()) and
                (self._points[:, 1].min() <= point[1] <=
                 self._points[:, 1].max()))

    def __getitem__(self, idx):
        return self._points[idx]

    def mark_image(self, image: np.ndarray, color=(1, 0, 0)):
        """Mark this :py:class:`PointsBasedLocation` in an image.
        """
        for point in self._points:
            image[max(point[1]-1, 0):min(point[1]+1, image.shape[0]),
                  max(point[0]-1, 0):min(point[0]+1, image.shape[1])] = color

    def extract_from_image(self, image: Imagelike) -> np.ndarray:
        """Extract this :py:class:`Location` from a given image.

        Arguments
        ---------
        image:
            The image from which this :py:class:`PointsBasedLocation`
            is to be extracted.
        """
        array = Image.as_array(image)
        height, width = array.shape[:2]
        point1_x, point1_y = self._points.min(axis=0)
        point2_x, point2_y = self._points.max(axis=0)
        point1_x, point1_y = max(0, int(point1_x)), max(0, int(point1_y))
        point2_x, point2_y = \
            min(width, int(point2_x)), min(height, int(point2_y))
        return array[point1_y:point2_y, point1_x:point2_x]

    def scale(self, factor: Union[float, Tuple[float, float]],
              reference: str = 'origin') -> None:
        """Scale the :py:class:`Location`.

        Arguments
        ---------
        factor:
            The scaling factor. This can either be a float, or a pair
            of floats in which case the first number is the horizontal (x)
            scaling factor and the second numger is the vertical (y)
            scaling factor.
        """
        if reference == 'origin':
            reference = np.ndarray((0, 0))
        elif reference == 'center':
            reference = self._points.mean(axis=0)
        else:
            reference = np.asarray(reference)

        self._points = (self._points - reference) * factor + reference

    @property
    def points(self) -> np.ndarray:
        """The points specifying this :py:class:`PointsBasedLocation`.
        This is an array of shape (n, 2), providing n points in form of (x, y)
        coordinates.
        """
        return self._points

    def __len__(self):
        return len(self._points)


class Landmarks(PointsBasedLocation):
    """Landmarks are an ordered list of points.
    """

    def __len__(self) -> int:
        return 0 if self._points is None else len(self._points)

    def __str__(self) -> str:
        return f"Landmarks with {len(self)} points."


class BoundingBox(PointsBasedLocation):
    # pylint: disable=invalid-name
    """A bounding box describes a rectangular arae in an image.
    """

    def __init__(self,
                 x1: Optional[float] = None, y1: Optional[float] = None,
                 x2: Optional[float] = None, y2: Optional[float] = None,
                 x: Optional[float] = None, y: Optional[float] = None,
                 width: Optional[float] = None,
                 height: Optional[float] = None) -> None:
        # pylint: disable=too-many-arguments
        super().__init__(np.ndarray((2, 2)))
        if x1 is not None:
            self.x1 = x1
        elif x is not None:
            self.x1 = x
        else:
            raise ValueError("No x coordinate supplied.")

        if y1 is not None:
            self.y1 = y1
        elif y is not None:
            self.y1 = y
        else:
            raise ValueError("No y coordinate supplied.")

        if x2 is not None:
            self.x2 = x2
        elif width is not None:
            self.width = width

        if y2 is not None:
            self.y2 = y2
        elif height is not None:
            self.height = height

    @property
    def x1(self) -> float:
        """The horizontal position of the left border of this
        :py:class:`BoundingBox`.

        """
        return self._points[0, 0]

    @x1.setter
    def x1(self, x1: float) -> None:
        self._points[0, 0] = x1

    @property
    def y1(self) -> float:
        """The vertical position of the upper border of this
        :py:class:`BoundingBox`.

        """
        return self._points[0, 1]

    @y1.setter
    def y1(self, y1: float) -> None:
        self._points[0, 1] = y1

    @property
    def x2(self) -> float:
        """The horizontal position of the right border of this
        :py:class:`BoundingBox`.

        """
        return self._points[1, 0]

    @x2.setter
    def x2(self, x2: float) -> None:
        self._points[1, 0] = max(x2, self.x1)  # Avoid negative width

    @property
    def y2(self) -> float:
        """The vertical position of the lower border of this
        :py:class:`BoundingBox`.

        """
        return self._points[1, 1]

    @y2.setter
    def y2(self, y2: float) -> None:
        self._points[1, 1] = max(y2, self.y1)  # Avoid negative height

    @property
    def x(self) -> float:
        """The horizontal position of the left border of this
        :py:class:`BoundingBox`.

        """
        return self.x1

    @x.setter
    def x(self, x: float) -> None:
        self.x1 = x

    @property
    def y(self) -> float:
        """The vertical position of the upper border of this
        :py:class:`BoundingBox`.

        """
        return self.y1

    @y.setter
    def y(self, y: float) -> None:
        self.y1 = y

    @property
    def width(self) -> float:
        """The width of the :py:class:`BoundingBox`.
        """
        return self.x2 - self.x1

    @width.setter
    def width(self, width: float) -> None:
        self.x2 = self.x1 + width

    @property
    def height(self) -> float:
        """The height of the :py:class:`BoundingBox`.
        """
        return self.y2 - self.y1

    @height.setter
    def height(self, height: float) -> None:
        self.y2 = self.y1 + height

    @property
    def size(self) -> Size:
        """The :py:class:`Size` of this :py:class:`BoundingBox`.
        """
        return Size(self.width, self.height)

    def mark_image(self, image: np.ndarray, color=None) -> None:
        """Mark this bounding box in the given image.
        """
        color = color or (0, 255, 0)
        size = image.shape[1::-1]
        thickness = max(1, max(size)//300)
        t1 = thickness//2
        t2 = (thickness+1)//2
        x1 = max(int(self.x1), t2)
        y1 = max(int(self.y1), t2)
        x2 = min(int(self.x2), size[0]-t1)
        y2 = min(int(self.y2), size[1]-t1)
        # print(f"mark_image[{self}]: image size={size}"
        #       f"shape={image.shape}, {image.dtype}:"
        #       f"{image.min()}-{image.max()}, box:({x1}, {y1}) - ({x2}, {y2})")
        for offset in range(-t2, t1):
            image[(y1+offset, y2+offset), x1:x2] = color
            image[y1:y2, (x1+offset, x2+offset)] = color

    def extract_from_image(self, image: Imagelike,
                           padding: bool = True,
                           copy: Optional[bool] = None) -> np.ndarray:
        """Extract the region described by the bounding box from an image.

        Arguments
        ---------
        padding:
            A flag indicating how to proceed in case this `BoundingBox`
            refers to points outside the image boundaries.  If `False`,
            only the valid part of the `BoundingBox` will be extracted,
            that is, the result is smaller than the `BoundingBox`.
            If `True`, the parts outside the image will be padded
            (currently only 0-padding is supported).
        copy:
            A flag indicating if the extracted region should be
            copied (`True`) or if just a view on the selected
            regien should be returned (`False`).  The view technique
            is faster and requires less memory, but will only work,
            if the result does not include parts outside the image.
            If `None`, the copy mode is automatically selected to
            be `False` (no copy) )if possible, otherwise a copy
            will be returned.
        """
        image = Image.as_array(image)
        image_size = image.shape[1::-1]

        x1, x2 = int(self.x1), int(self.x2)
        y1, y2 = int(self.y1), int(self.y2)
        invalid = (x1 < 0 or x2 > image_size[0] or
                   y1 < 0 or y2 > image_size[1])

        if invalid and padding:
            if copy is False:
                raise ValueError("Cannot apply padding if copy is False.")
            copy = True
        else:
            # no padding: resize bounding box to become valid
            x1, x2 = max(x1, 0), min(x2, image_size[0])
            y1, y2 = max(y1, 0), min(y2, image_size[1])
            invalid = False

        if copy:
            box = self._copy_box(image, x1, y1, x2, y2)
        else:
            box = image[y1:y2, x1:x2]

        return box

    @staticmethod
    def _copy_box(image:np.ndarray, x1:int , y1:int , x2:int , y2:int) -> np.ndarray:
        """Copy a box from an image.  The box coordinates may be outside
        of the image, in which case the corresponding parts in the box
        will be filled with zeros (black).
        """
        image_size = image.shape[1::-1]
        channels = 1 if image.ndim < 3 else image.shape[2]
        width, height = x2 - x1, y2 - y1

        shape = (height, width) + ((channels, ) if channels > 1 else ())
        box = np.zeros(shape, dtype=image.dtype)
        slice_box0 = slice(max(-y1, 0), height-max(y2-image_size[1], 0))
        slice_box1 = slice(max(-x1, 0), width-max(x2-image_size[0], 0))
        slice_image0 = slice(max(y1, 0), min(y2, image_size[1]))
        slice_image1 = slice(max(x1, 0), min(x2, image_size[0]))
        LOG.debug("Extracting boxfrom image: image[%s, %s] -> box[%s, %s]",
                  slice_image0, slice_image1, slice_box0, slice_box1)
        box[slice_box0, slice_box1] = image[slice_image0, slice_image1]
        return box

    def __str__(self) -> str:
        """String representation of this :py:class:`BoundingBox`.
        """
        # return f"({self.x1},{self.y1})-({self.x2},{self.y2})"
        # return (f"BoundingBox at ({self.x}, {self.y})"
        #         f" of size {self.width} x {self.height}")
        return (f"BoundingBox from ({self.x1}, {self.y1})"
                f" to ({self.x2}, {self.y2})")

    def __add__(self, other: 'BoundingBox') -> 'BoundingBox':
        """Adding two bounding boxes means to create a new bounding box
        that bounds both of them.
        """
        return BoundingBox(x1=min(self.x1, other.x1),
                           y1=min(self.y1, other.y1),
                           x2=max(self.x2, other.x2),
                           y2=max(self.y2, other.y2))

    def __mul__(self, other: 'BoundingBox') -> 'BoundingBox':
        """Multiplying two bounding boxes means to form the intersection.

        """
        return BoundingBox(x1=max(self.x1, other.x1),
                           y1=max(self.y1, other.y1),
                           x2=min(self.x2, other.x2),
                           y2=min(self.y2, other.y2))

    def area(self):
        """Compute the area of this :py:class:`BoundingBox`.
        """
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """The center of this bounding box as an (x,y) pair.
        """
        return ((self.x1 + self.x2)/2, (self.y1 + self.y2)/2)


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

    _location: Location
    _atributes = None

    color_min_confidence: np.ndarray = np.asarray((255., 0., 0.))  # red
    color_max_confidence: np.ndarray = np.asarray((0., 255., 0.))  # green

    def __init__(self, location: Location, **attributes):
        self._location = location
        self._attributes = attributes

    def __str__(self) -> str:
        return f"{self._location} with {len(self._attributes)} attributes"

    def __contains__(self, point) -> bool:
        return point in self._location

    def __getattr__(self, name: str) -> Any:
        if name in self._attributes:
            return self._attributes[name]
        raise AttributeError(f"Region has no attribute '{name}'. Valid "
                             f"attributes are: {self._attributes.keys()}")

    def __len__(self) -> int:
        return len(self._attributes)

    @property
    def location(self):
        """The :py:class:`Location` describing this :py:class:`Region`.
        """
        return self._location

    def mark_image(self, image: Imagelike,
                   color: Optional[Color] = None,
                   copy: Optional[bool] = None) -> Image:
        """Mark this :py:class:`Region` in a given `Image`.

        Arguments
        ---------
        image:
            The image in which the `Region` is to be marked.
        color:
            The color to be used for marking. If `None`, and
            the `Region` is annotated with a confidence score,
            the color will be choosen based on that confidence
            score.
        copy:
            A flag indicating if the marks should be made in
            the original image (`False`) or in a copy of the
            image (`True`).

        Result
        ------
        marked_image:
            The image including the marks.
        """
        if color is None and 'confidence' in self._attributes:
            confidence = max(0, min(1.0, self._attributes['confidence']))
            mark_color = ((1-confidence) * self.color_min_confidence +
                          confidence * self.color_max_confidence)
            color = tuple(mark_color.astype(np.uint8))
        the_image = Image(image, copy=copy)
        self._location.mark_image(the_image.array, color=color)
        return the_image

    def extract_from_image(self, image: Imagelike,
                           **kwargs) -> np.ndarray:
        """Extract this :py:class:`Region` from a given image.

        Arguments
        ---------
        image:
            The image from the the region is to be extracted.

        Result
        ------
        patch:
            A numpy array (`dtype=np.uint8`) containing the extracted
            region.
        """
        return self._location.extract_from_image(image, **kwargs)

    def scale(self, factor: Union[float, Tuple[float, float]],
              reference: str = 'origin') -> None:
        """Scale this region by a given factor.

        Arguments
        ---------
        factor:
            The scaling factor. This can either be a float, or a pair
            of floats in which case the first number is the horizontal (x)
            scaling factor and the second numger is the vertical (y)
            scaling factor.

        reference (currently not implemented):
            The reference point.  The default is `'origin'`, meaning
            all coordinates are scaled with respect to the origin.
            Another special value is `'center'`, meaning that
            the center of the region should be taken as reference
            point.

            Question: What would be an application scenario for this feature?
        """
        del reference
        if self._location is not None:
            self._location.scale(factor)
