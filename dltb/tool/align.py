"""Alinment tool.

An alignment tool takes an image of an object and aligns it to obtain
a standardized version of the image, which may improve subsequent
tasks like verification, recognition, or attribute estimation.

"""
# standard imports
from typing import TypeVar, Generic, Optional

# thirdparty imports
import numpy as np

# toolbox imports
from ..base.implementation import Implementable
from ..base.image import Imagelike, Image, ImageWarper
from ..base.image import Landmarks, Size, Sizelike
from .face.landmarks import Detector as LandmarksDetector

LandmarksType = TypeVar('LandmarksType', bound=Landmarks)


class LandmarkAligner(Generic[LandmarksType]):
    """A :py:class:`LandmarkAligner` aligns images based on specific
    landmarks.  It employs a :py:class:`LandmarksDetector` to detect
    the landmarks in an image and then computes a transformation
    to move these landmarks to their reference points.

    The goal is to transform the original image in a way that the
    positions of the landmarks are moved to predefined standard
    positions.

    A :py:class:`LandmarkAligner` uses a specific landmarking scheme
    that specifies the number and meaning of the landmarks.  The
    :py:class:`LandmarkDetector` defines standard positions for these
    landmarks.

    The :py:class:`LandmarkAligner` relies either on a compatible
    :py:class:`Landmarker` to obtain the landmarks for a given input
    image, or these landmarks have to be provided expilitly.
    """

    _detector: LandmarksDetector[LandmarksType] = None
    _reference: LandmarksType = None
    _size: Size = None
    _image_warper: ImageWarper = None

    def __init__(self, detector: LandmarksDetector[LandmarksType],
                 size: Optional[Sizelike] = None,
                 warper: Optional[ImageWarper] = None, **kwargs) -> None:
        """
        """
        super().__init__(**kwargs)
        self._detector = detector
        self.size = size
        self._image_warper = ImageWarper() if warper is None else warper

    @property
    def size(self) -> Size:
        return self._size

    @size.setter
    def size(self, size: Sizelike) -> None:
        self._size = \
            self._detector._reference_size if size is None else Size(size)
        self._reference = self._detector.reference(self._size)

    @property
    def reference(self) -> LandmarksType:
        return self._reference

    @property
    def detector(self) -> LandmarksDetector:
        return self._detector

    @property
    def warper(self) -> ImageWarper:
        return self._image_warper

    def compute_transformation(self, landmarks: LandmarksType) -> np.ndarray:
        """Compute an (affine) transformation to map the given landmarks
        to the reference landmarks of the :py:class:`LandmarkAligner`.
        """
        transformation = \
            self._image_warper.compute_transformation(landmarks.points,
                                                      self._reference.points)
        return transformation

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
        aligned = self._image_warper.warp(image, transformation, self._size)
        return aligned

    def __call__(self, image: Imagelike,
                 landmarks: Optional[LandmarksType] = None) -> np.ndarray:
        """Align an image by applying an (affine) transformation that maps
        source points to target points.

        Arguments
        ---------
        image:
            The image to align.
        landmarks:
            A list of points to be mapped onto the reference points,
            given as (x,y) coordinates.  If `None`, then the detector
            will be used to obtain landmarks, and all detections
            will be aligned.

        Result
        ------
        aligned:
            The aligned image (if landmarks were given) or a batch
            of aligned images.
        """
        image = Image.as_array(image)
        if landmarks is None:
            detections = self._detector.detect_landmarks(image)
            if not detections:
                return None
            result = np.nparray((len(detections),
                                 self._size.height, self._size.width,
                                 image.shape[2]))
            for idx, landmarks in enumerate(detections):
                result[idx] = self(image, landmarks)
            return result
        else:
            transformation = self.compute_transformation(landmarks)
            return self.apply_transformation(image, transformation)
