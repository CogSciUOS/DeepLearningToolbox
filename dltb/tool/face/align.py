"""Face alinment tool.

A face alignment tool takes an image of a face an aligns it to obtain
a standardized version of the image, which may improve subsequent
tasks like verification, recognition, or attribute estimation.

"""

# toolbox imports
from . import Tool
from .landmarks import FacialLandmarks
from ..align import PointAligner
from ..base.implementation import Implementable
from ..base.image import Image, Imagelike


class Transformation:
    """
    """

    def transform(self, image: Imagelike) -> Image:
        """
        """


class Aligner(Tool):
    """
    """

    def align(self, image: Imagelike) -> Image:
        """Align the given face image.
        """


class LandmarkAligner(PointAligner):
    """A :py:class:`LandmarkAligner` aligner uses a set of facial
    landmarks as basis for the alignment.  The goal is to transform
    the original image in a way that the positions of the landmarks
    are moved to predefined standard positions.

    A :py:class:`LandmarkAligner` uses a specific landmarking scheme
    that specifies the number and meaning of the landmarks.  The
    :py:class:`LandmarkAligner` defines standard positions for these
    landmarks.  The :py:class:`LandmarkAligner` relies either on a
    compatible :py:class:`Landmarker` to obtain the landmarks for a
    given input image, or these landmarks have to be provided
    expilitly.

    """

    _aligner = None

    _landmarker = None

    def align(self, image: Imagelike, size) -> Image:
        """Align the given face image.
        """
        landmarks = self._landmarker.landmarks(image)
        return self._aligner.align_points(image, landmarks, size)
