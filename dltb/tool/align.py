"""Face alinment tool.

A face alignment tool takes an image of a face an aligns it to obtain
a standardized version of the image, which may improve subsequent
tasks like verification, recognition, or attribute estimation.

"""
# standard imports
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple

# thirdparty imports
import numpy as np

# toolbox imports
from ..base.implementation import Implementable
from ..base.image import Imagelike, Landmarks
from .face.landmarks import Detector as LandmarksDetector

LandmarksType = TypeVar('LandmarksType', bound=Landmarks)


class LandmarkAligner(ABC, Implementable, Generic[LandmarksType]):
    """A :py:class:`LandmarkAligner` aligns images based on specific
    landmarks.  It employs a :py:class:`LandmarksDetector` to detect
    the landmarks in an image and then computes a transformation
    to move these landmarks to their reference points.
    """

    _detector: LandmarksDetector[LandmarksType]
    _reference: LandmarksType
    _size: Tuple[int, int]

    def __init__(self, detector: LandmarksDetector[LandmarksType],
                 reference: LandmarksType,
                 size: Tuple[int, int], **kwargs) -> None:
        """
        """
        super().__init__(**kwargs)
        self._detector = detector
        self._reference = reference
        self._size = size

    @abstractmethod
    def compute_transformation(self, landmarks: LandmarksType) -> np.ndarray:
        """Compute an (affine) transformation to map the given landmarks
        to the reference landmarks of the :py:class:`LandmarkAligner`.
        """
        # to be implemented by subclasses

    @abstractmethod
    def apply_transformation(self, image: Imagelike,
                             transformation: np.ndarray) -> np.ndarray:
        """Apply a Transformation to an image.

        Arguments
        ---------
        image:
            The image to be transformed.
        transformation:
            The transformation to be applied
        """
        # to be implemented by subclasses

    def align(self, image: Imagelike) -> np.ndarray:
        """Align an image by applying an (affine) transformation that maps
        source points to target points.

        Arguments
        ---------
        image:
            The image to align.
        points:
            A list of points to be mapped onto the reference points,
            given as (x,y) coordinates

        Result
        ------
        aligned:
            The aligned image.
        """
        landmarks = self._detector(image)
        transformation = self.compute_transformation(landmarks)
        return self.apply_transformation(image, transformation)
