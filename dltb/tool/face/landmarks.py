"""Facial landmarking.  Facial landmarks are salient points in the face,
like eyes, nose, mouth, etc.  Detecting such points can be interesting
in itself, but it can also support further processing, like aligning
the face.

"""

# standard imports
from typing import TypeVar, Protocol, Generic, Tuple, Iterable

# thirdparty imports
import numpy as np

# toolbox imports
from .detector import Detector as FaceDetector
from ..detector import ImageDetector
from ...base.busy import busy
from ...base.meta import Metadata
from ...base.image import Landmarks, Imagelike
from ...base.implementation import Implementable


class FacialLandmarks(Protocol):

    def eyes(self) -> np.ndarray:
        """Points describing the eyes.
        """

    def mouth(self) -> np.ndarray:
        """Points describing the mouth.
        """

    @staticmethod
    def reference(size: Tuple[int, int]) -> 'FacialLandmarks':
        """Reference position for the landmarks in a standardized
        face.
        """

LandmarksType = TypeVar('LandmarksType', bound=FacialLandmarks)


# pylint: disable=abstract-method
class FacialLandmarksBase(Landmarks):
    """Facial landmarks describe salient points in a face, like eyes,
    nose, mouth, etc.  There exist several annotations schemes for
    facial landmarks.

    """

    def eyes(self):
        """Points describing the eyes.
        """
        raise NotImplementedError()

    def mouth(self):
        """Points describing the mouth.
        """
        raise NotImplementedError()

    @classmethod
    def reference(cls, size: Tuple[int, int], padding: Tuple,
                  keep_aspect_ratio: bool = True) -> 'FacialLandmarks':
        """Reference position for the landmarks in a standardized
        face.
        """
        reference_size, reference_landmarks = cls._reference()
        reference_ratio = reference_size[0] / reference_size[1]
        ratio = size[0] / size[1]
        if reference_ratio < ratio:  # target is wider => shift right
            delta_x = size[0] * (ratio-reference_ratio) / 2
        elif reference_ratio > ratio:  # target is taller => shift upwards
            delta_y = size[1] * (reference_ratio-ratio) / 2
        # FIXME[todo]: implementation incomplete
        return reference_landmarks

    @classmethod
    def _reference(cls) -> Tuple['FacialLandmarks', Tuple[float, float]]:
        """Reference position for the landmarks in a standardized
        face.
        """
        raise NotImplementedError()
        

class FacialLandmarks68(FacialLandmarksBase):
    """An 68-point facial landmark annotation scheme.
    """


class Detector(Protocol, Generic[LandmarksType]):
    """Interface for a :py:class`LandmarksDetector`.  The essential method
    is :py:meth:`detect_landmarks` that detects landmarks in an image.

    """

    def detect_landmarks(self, image: Imagelike) -> Iterable[LandmarksType]:
        """Detect facial landmarks for one or multiple faces depicted on an
        image.
        """


class DetectorBase(Implementable, ImageDetector):  # , Generic[LandmarksType]
    """Base implementation for a facial :py:class:`LandmarksDetector`.

    Most :py:class:`LandmarksDetector` will derive from this base
    class, which adds some further functionality.
    """
    _face_detector: FaceDetector = None

    @staticmethod
    def create(name: str, prepare: bool = True):
        """Create a facial landmark detector.
        """
        if name == 'dlib':
            from .dlib import FacialLandmarkDetector
            detector = FacialLandmarkDetector()
        else:
            raise ValueError(f"Unknown detector name '{name}'.")

        if prepare:
            detector.prepare()
        return detector

    def __init__(self, face_detector: FaceDetector = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.face_detector = face_detector

    @property
    def face_detector(self):
        """The face detector employed by this facial landmarks detector.
        """
        return self._face_detector

    @face_detector.setter
    def face_detector(self, face_detector):
        self._face_detector = face_detector

    def _detect_regions(self, image: np.ndarray, regions):
        """Apply the facial landmark detector to all specified regions
        in the given image.
        """
        metadata = Metadata(
            description='Facial landmarks detected by the dlib detctor')
        for region in regions:
            # FIXME[hack]: suppose region.location is a BoundingBox
            detection = self._predictor(image, region.location)
            metadata.add_region(self._detection_landmarks(detection))

        return metadata

    def _detect_all(self, image: np.ndarray,
                    face_detector: FaceDetector = None) -> Metadata:
        """Detect facial landmarks for all faces depicted on the given
        image.

        This method operates in two steps: it first applies the face
        detector to locate faces in the image and then applies the
        landmark detection to each of these faces.
        """
        if face_detector is None:
            face_detector = self._face_detector
        if face_detector is None:
            raise ValueError("No face detector was provided for "
                             "face landmark detection.")
        faces = face_detector.detect(image)
        return self._detect_all(image, faces.regions)

    #
    # Precessing
    #

    def process_all(self, data):
        """Process the given data.

        """
        self._next_data = data
        if not self.busy:
            self._process()

    @busy("processing")
    def _process_all(self):
        """Do the actual processing.
        """
        while self._next_data is not None:

            self._data = self._next_data
            self._next_data = None
            self.change(data_changed=True)

            self._detections = self.detect_all(self._data)
            self.change(detection_finished=True)
