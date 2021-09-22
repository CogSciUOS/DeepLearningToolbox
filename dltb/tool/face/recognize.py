"""Definition of face recognizers.
"""

# standard imports
from typing import Any
from abc import ABC, abstractmethod

# third party imports
import numpy as np

# Toolbox imports
from ..image import ImageTool
from ...base.implementation import Implementable


class FaceEmbedding(ImageTool, Implementable, ABC):
    # pylint: disable=abstract-method
    """A :py:class:`FaceEmbedding` embeds an image of a face into some
    metric space.  The ratio is, that images of the same person should
    be mapped to points close to each other, while images of different
    persons should result in points far away from each other.  These
    embedding then allows to do face verification (checking if two
    images depict the same person) and face identification (select a
    matching person from a gallery of face images).
    """

    external_result = ('embedding', )

    def distance(self, embedding1, embedding2) -> float:
        """Compute the distance between two embeddings.
        """
        return self._distance(self._internal_embedding(embedding1),
                              self._internal_embedding(embedding2))

    def _internal_embedding(self, embedding) -> Any:
        """Translate the embedding into the format used for internal
        processing
        """
        return embedding

    @abstractmethod
    def _distance(self, embedding1, embedding2) -> float:
        """To be implemented by subclasses.
        """


class ArcFace(FaceEmbedding, Implementable, ABC):
    # pylint: disable=abstract-method
    """:py:class:`ArcFace` is a special type of :py:class:`FaceEmbedding`
    that maps faces to points on a hypersphere and uses angular
    distance for comparison.  It is judged as one of the
    state-of-the-art face recognizers.
    """
