"""Segmentation refers to the process of identifying segments in given
data.  Most prominently, image segmentation aims at identifying
(sematically) coherent regions in an image.
"""
# standard imports
from typing import Iterable, Optional, Tuple, Union, Any
import math

# thirdparty
import numpy as np

# toolbox imports
from ..base.image import Image, Imagelike
from ..base.implementation import Implementable
from ..util.image import imshow
from .image import ImageTool


Bitmasklike = Union[np.ndarray, Any]


class Bitmask:
    """A bitmask.  The bitmask is intended to mark regions in an image.

    """

    def __init__(self, bitmask: Bitmasklike) -> None:
        if isinstance(bitmask, np.ndarray):
            self._bitmask = np.array(bitmask, dtype=bool)
        elif hasattr(bitmask, 'bitmask'):
            self._bitmask = bitmask.bitmask
        else:
            raise TypeError(f"Invalid bitmap type: {type(bitmap)}")

    @property
    def shape(self) -> Tuple[int, int]:
        return self._bitmask.shape

    @property
    def array(self) -> np.ndarray:
        return self._bitmask
    
    def mark_image(self, image: Imagelike, color=(0,1,0),
                   alpha: float = 0.3) -> np.ndarray:
        """Mark the region selected by this `Bitmask` in a given
        image.
        """
        img = Image.as_array(image)
        print(img.shape, id(img), color, image.dtype)
        color = np.array(color) * 255
        for channel in range(3):
            img[self._bitmask] = \
                (img[self._bitmask] * (1-alpha) +
                 alpha * color).astype(np.uint8)
        return img



# FIXME[old]:
class ShieldBitMask(dict):
    """The Bit Mask class for a consistent bit mask representation

    Needs a bit Mask as an input with just bool values, otherwise returns
    an TypeError.

    Attributes:
        bitmask: The bitmask the BitMask object contains

    Attributes:
        bitmask: The bitmask as list
    """

    def __init__(self, bitmask):
        self.bitmask = self.check_bit_mask(bitmask)
        dict.__init__(self, bitmask=self.bitmask)

    def check_bit_mask(self, bitmask):
        """Check if bitmask is in the right format(bool)

        Args:
           bitmask: An array that should be checked

        Returns:
            The controlled array

        Raise:
            TypeError: if the bitmask was not bool
            ValueError: if the bitmask was empty
        """
        if not isinstance(bitmask, np.ndarray):
            bitmask = np.asarray(bitmask)
        else:
            pass

        try:
            if bitmask.dtype == bool:
                return bitmask.tolist()

            raise TypeError("The bit mask has the wrong dtype. "
                            f"Is {bitmask.dtype} not bool")
        except Exception as exc:
            raise ValueError("The bit mask is empty.") from exc

    def get_bitmask(self) -> np.ndarray:
        """Returns the bit mask as np.ndarry

        Returns:
            np.ndarray: the actual bitmask of this BitMask object
        """
        return np.asarray(self.bitmask)

    def __str__(self) -> str:
        """Function for printing methods"""
        npmask = np.array(self.bitmask)
        return (f"BitMask with {np.count_nonzero(npmask is True)} TRUE"
                f" and {np.count_nonzero(npmask is False)} FALSE")

    def __repr__(self) -> str:
        """Function for printing methods"""
        npmask = np.array(self.bitmask)
        return (f"BitMask with {np.count_nonzero(npmask is True)} TRUE"
                f" and {np.count_nonzero(npmask is False)} FALSE")


class Segmentation:
    """A segmentation describes the segmentation of an image by assigning
    a unique label to each pixel of the image.
    """

    def __init__(self, segmentation: Optional[np.ndarray] = None,
                 bitmasks: Iterable[Bitmasklike] = None) -> None:
        if segmentation is None and bitmasks is None:
            raise TypeError("Require either labels or bitmasks to initialize "
                            "a Segmentation.")

        if segmentation is not None and bitmasks is not None:
            raise TypeError("Cannot initialize Segmentation from both, "
                            "labels and bitmasks")

        if segmentation is not None:
            self._segmentation = np.ndarray(segmentation, dtype=int)
            self._max_label = np.max(segmentation)
        elif bitmasks is not None:
            self._from_bitmasks(bitmasks)

    def _from_bitmasks(self, bitmasks: Iterable[Bitmasklike]) -> None:
        bitmask_iterator = iter(bitmasks)
        bitmask = next(bitmask_iterator)
        segmentation = np.array(Bitmask(bitmask).array, dtype=int)
        label = 1
        for label, bitmask in enumerate(bitmask_iterator, start=2):
            segmentation[Bitmask(bitmask).array] = label
        self._segmentation = segmentation
        self._max_label = label

    def bitmasks(self) -> Iterable[Bitmask]:
        """Iterate over the bitmasks for this :py:class:`Segmentation`.
        """
        for label in range(1, self._max_label + 1):
            yield Bitmask(self._segmentation == label)


class ImageSegmenter(ImageTool, Implementable):
    """A segmentation tool for images.
    """

    #
    # Detector
    #

    def _process(self, data, **kwargs) -> Any:
        """Processing data with a :py:class:`Detector` means detecting.
        """
        return self._segment(data, **kwargs)

    def _segment(self, image: np.ndarray, **kwargs) -> Segmentation:
        raise NotImplementedError("To be implemented by subclasses")

    def segment_image(self, image: Imagelike, **_kwargs) -> Segmentation:
        """Apply the detector to the given image.

        Arguments
        ---------
        image:
            The image to be processed by this :py:class:`ImageSegmenter`.

        Result
        ------
        segmentation:
            The `Segmentation` obtained from the `ImageSegmenter`.
        """
        return self._segment(Image.as_array(image))

    def segment_and_show(self, image: Imagelike, **_kwargs) -> None:
        """Apply the `ImageSegmenter` to the given `image` and show
        the result.

        Arguments
        ---------
        image:
            The image to be processed by this :py:class:`ImageDetector`.
        """
        imshow(self.mark_image2(image))

    #
    # Marking segmentations
    #

    def mark_image2(self, image: Imagelike,
                    segmentation: Optional[Segmentation] = None,
                    copy: bool = True) -> np.ndarray:
        """Mark the given detections in an image.

        Arguments
        ---------
        image: Imagelike
            The image into which the detections are to be drawn.
        detections: Detections
            The detections to draw.
        copy: bool
            A flag indicating if detections should be marked in
            a copy of the image (`True`) or into the original
            image object (`False`).

        Result
        ------
        marked_image: np.ndarray
            An image in which the given `Segmentation` are visually marked.
        """
        array = Image.as_array(image, copy=copy)
        # array.setflags(write=True)
        array = array.copy()  # FIXME[why]: was alread copied above
        if segmentation is None:
            segmentation = self.segment_image(Image(array))
        for label, bitmask in enumerate(segmentation.bitmasks()):
            color = (1,0,0) if label else (0,0,1)
            bitmask.mark_image(array, color=color)
        return array
