"""Facial landmarking.  Facial landmarks are salient points in the face,
like eyes, nose, mouth, etc.  Detecting such points can be interesting
in itself, but it can also support further processing, like aligning
the face.

"""

# standard imports
# typing.Protocol is only introduced in Python 3.8. In prior versions,
# it is available from thirdparty package 'typing_extensions'.
from typing import TypeVar, Generic, Tuple, Iterable, Optional
import logging

# thirdparty imports
import numpy as np

# toolbox imports
from .detector import Detector as FaceDetector
from ..detector import ImageDetector
from ...typing import Protocol
from ...base.busy import busy
from ...base.metadata import Metadata
from ...base.data import Data
from ...base.image import Image, Imagelike, ImageDisplay
from ...base.image import BoundingBox, Landmarks, Sizelike, Size
from ...base.implementation import Implementable
from ...util.image import imwrite
from ...util.keyboard import SelectKeyboardObserver
from ...util.canvas import canvas_create, canvas_add_image

# logging
LOG = logging.getLogger(__name__)


class FacialLandmarks(Protocol):
    """Interface for facial landmarks.
    """

    def eyes(self) -> np.ndarray:
        """Points describing the eyes.
        """

    def mouth(self) -> np.ndarray:
        """Points describing the mouth.
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

        Arguments
        ---------
        size:
            The size to which the reference landmarks should be scaled.
        padding:
            unclear? - need documentation and implementation!
        keep_aspect_ratio:
            How to proceed if aspect ratio of reference points does not
            agree the aspect ratio of the target size.  If `True`,
            the original aspect size (of the reference points) will be kept,
            meaning that the wider axis will be padded.  If `False`,
            the reference points will be scaled along both axis with
            a different factor along each axis.
        """
        reference_size, reference_landmarks = cls._reference()
        reference_ratio = reference_size[0] / reference_size[1]
        ratio = size[0] / size[1]
        if reference_ratio < ratio:  # target is wider => shift right
            _delta_x = size[0] * (ratio-reference_ratio) / 2
        elif reference_ratio > ratio:  # target is taller => shift upwards
            _delta_y = size[1] * (reference_ratio-ratio) / 2
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


class DetectorBase(ImageDetector, Implementable):  # , Generic[LandmarksType]
    """Base implementation for a facial :py:class:`LandmarksDetector`.

    Most :py:class:`LandmarksDetector` will derive from this base
    class, which adds some further functionality.
    """
    _face_detector: FaceDetector = None

    # FIXME[hack]: should be deducible from type hints
    _LandmarksType: type = None   # the landmarks type
    _reference_landmarks = None   # :LandmarksType - the reference landmarks
    _reference_size: Size = None  # design size for the reference landmarks

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

    def reference(self, size: Optional[Sizelike] = None,
                  keep_aspect_ratio: bool = True) -> 'FacialLandmarks':
        """Reference position for the landmarks in a standardized
        face.
        """
        size = self._reference_size if size is None else Size(size)

        if size == self._reference_size:
            return self._reference_landmarks

        # scale the landmarks to the target size
        scale_x = size.width / self._reference_size.width
        scale_y = size.height / self._reference_size.height

        if keep_aspect_ratio:
            if scale_x > scale_y:
                scale_x = scale_y
                delta_x = (size.width - self._reference_size.width) / 2
                delta_y = 0
            elif scale_y > scale_x:
                scale_y = scale_x
                delta_y = (size.height - self._reference_size.height) / 2
                delta_x = 0
            else:
                delta_x = delta_y = 0
        else:
            # don't keep the aspect ratio, scale both axes differently
            delta_x = delta_y = 0
        points = np.zeros_like(self._reference_landmarks)
        for idx, point in enumerate(self._reference_landmarks):
            points[idx] = (point[0] * scale_x + delta_x,
                           point[1] * scale_y + delta_y)
        return self._LandmarksType(points=points)


def landmark_face_hack(image: Image, detector) -> \
        Tuple[BoundingBox, Landmarks, bool]:
    """Obtain (best) bounding box and landmarks for a given image using
    a combined bounding box and landmark detector.

    Arguments
    ---------

    Result
    ------
    bounding_box:
        The bounding box for the (best) detection.
    landmarks:
        The facial landmarks for the (best) detection.
    unique:
        A flag indication if the detector returned a singe detection (`True`)
        or if the best of multiple detection was choosen using some
        heuristics (`False`)

    Raises
    ------
    ValueError
        No (best) detection could be identified.  Either there was no
        detection or the heuristics could not decided of a best detection.
    """
    # image = Image(image)
    detections = detector(image)
    #detections = detector.detections(image)

    if not detections.has_regions:
        raise ValueError("No regions returned by detector for "
                         f"image '{image.filename}'.")

    regions = len(detections.regions)
    if regions == 0:
        raise ValueError("There were 0 regions returned by detector for "
                         f"image '{image.filename}'.")

    if regions % 2 != 0:
        raise ValueError("The box-and-landmark-detector returned "
                         f"an odd number ({regions}) of regions "
                         f"for image '{image.filename}'.")

    faces = regions // 2
    if faces == 1:
        best = 0
    else:
        best = detector.select_best(detections, image)
        LOG.info("%d faces detected in '%s' (%d is best).",
                 faces, getattr(image, 'filename', '?'), best)

    bbox = detections.regions[best*2].location
    landmarks = detections.regions[best*2+1].location

    detector.mark_data(image, detections, best=best, group_size=2)

    return bbox, landmarks, faces == 1


def align_face_hack(aligner, image, landmarks) -> np.ndarray:
    """Align a face image given an aligner.
    """
    # ./dltb/tool/face/landmarks.py
    #  - definition of a LandmarkDetector (called "DetectorBase")
    #    all subclasses should provide
    #     _reference_landmarkds: Landmarks
    #     _reference_size: Size
    #    the method reference can scale reference landmarks to a
    #    desired size
    #
    # ./dltb/tool/align.py
    #  - definition of abstract class LandmarkAligner
    #
    # ./dltb/thirdparty/opencv/__init__.py
    #  - definition of class OpenCVLandmarkAligner(LandmarkAligner):
    #    implementing the methods compute_transformation and
    #    apply_transformation
    #
    # ./dltb/thirdparty/face_evolve/mtcnn.py
    #   - Implementation of a LandmarkDetector providing
    #     reference landmarks
    #
    # ./dltb/thirdparty/datasource/lfw.py
    #
    #

    # The reference points used by the aligner
    _reference = aligner.reference

    # The transformation computed by the aligner
    transformation = aligner.compute_transformation(landmarks)

    # The aligned image computed by the aligner
    aligned_image = aligner.apply_transformation(image, transformation)

    return aligned_image


def apply_single_hack(data: Data, detector, aligner,
                      display: Optional[ImageDisplay] = None) -> None:
    """Apply a transformation to a data objects.

    State: immature - hard-wired transformation

    Arguments
    ---------
    data:
        The :py:class:`Data` object to which the operation is to be
        applied.
    input_directory:
        The base directory of the data.  Relative paths are determined
        relative to the directory.
    output_directory:
        The output directory.  Relative output filenames are interpreted
        relative to this directory.
    """

    # assume data is Imagelike: process_image transforms Imagelike -> Image
    image = detector.process_image(data)

    # bounding box and landmark for the (best) detection
    bbox, landmarks, _unique = landmark_face_hack(image, detector)

    # a marked version or the image
    marked_image = detector.marked_image(image)

    # an aligned (and cropped) version of the marked image
    aligned_image_marked = align_face_hack(aligner, marked_image, landmarks)

    # an aligned version of the original image
    aligned_image = aligner(image, landmarks)

    # print(data.filename, result[0], result[1], result[2])
    # output_tuple = (image.filename,
    #                result[0].x1, result[0].y1, result[0].x2, result[0].y2
    #                )  # FIXME[todo]:  result[1]. ...
    # result[0].mark_image(image.array)
    # result[1].mark_image(image.array)
    if display is not None:
        shape = image.shape
        canvas = canvas_create(size=(shape[0]*2, shape[1]*2),
                               channels=shape[2])

        # 1. the original image
        canvas_add_image(canvas, marked_image, rows=2, columns=2, index=1)

        # 2. the image cropped to the bounding box
        cropped_image = \
            marked_image[int(bbox.y1):int(bbox.y2), int(bbox.x1):int(bbox.x2)]
        canvas_add_image(canvas, cropped_image, rows=2, columns=2, index=2)

        # 3. the aligned image (marked version)
        canvas_add_image(canvas, aligned_image_marked,
                         rows=2, columns=2, index=3)

        # 4. the aligned image
        canvas_add_image(canvas, aligned_image, rows=2, columns=2, index=4)

        # display.show(canvas, blocking=not unique, wait_for_key=not unique)
        display.show(canvas, blocking=False)

    return aligned_image


def apply_multi_hack(datasource: Iterable[Data], detector, aligner,
                     input_directory=None, output_directory=None,
                     display: Optional[ImageDisplay] = None,
                     progress: Optional[Iterable] = None) -> None:
    """Apply a transformation to multiple data objects.

    State: immature - hard-wired transformation
    """
    if progress is not None:
        datasource = progress(datasource)
        output = datasource.set_description
    else:
        output = print

    stop_on_key_press = SelectKeyboardObserver()

    input_directory = input_directory and str(input_directory)

    try:
        for data in stop_on_key_press(datasource):
            output(f"{data.filename} [{data.shape}]")
            aligned_image = apply_single_hack(data, detector, aligner,
                                              display=display)
            if display is not None and display.closed:
                LOG.error("Display was closed: stopping operation.")
                break

            if input_directory and output_directory:
                # Write result into a file in the output directory.
                # Relative filenames in input and output directory
                # will be the same.
                if not data.has_attribute('filename'):
                    LOG.warning("No filename for data object. No output "
                                "file will be written.")
                elif not data.filename.startswith(input_directory):
                    LOG.warning("File will not be written: filename '%s' "
                                "is outside the input directory '%s'.",
                                data.filename, input_directory)
                else:
                    output_filename = output_directory / \
                        data.filename[len(input_directory) + 1:]
                    if output_filename.exists():
                        LOG.warning("Output file '%s' exist. "
                                    "Will not overwrite.", output_filename)
                    else:
                        output_filename.parent.mkdir(parents=True,
                                                     exist_ok=True)
                        imwrite(output_filename, aligned_image)
                    # print(f"{data.filename} -> {output_filename} "
                    #       f"[{aligned_image.shape}]")

    except KeyboardInterrupt:
        LOG.error("Keyboard interrupt: stopping operation.")
