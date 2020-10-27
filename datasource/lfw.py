"""The Labeled Faces in the Wild (LFW) dataset.
"""

# standard imports
import os
import logging

# third party imports
import numpy as np

# toolbox imports
from util.image import BoundingBox, Region, Landmarks
from dltb.base.data import Data
from dltb.tool.classifier import ClassScheme
from dltb.datasource import Imagesource, DataDirectory

# logging
LOG = logging.getLogger(__name__)


class LabeledFacesInTheWild(DataDirectory, Imagesource):
    # pylint: disable=too-many-ancestors
    """"Labeled Faces in the Wild" (LFW) has for a long time been
    a standard benchmarking  dataset in the domain of face recognition.
    It consists of 13.233 images (each of size 250x250 pixels)
    of 5.749 people, with 1.680 people having two or more images.

    The images are contained in a directory called `lfw`, with
    one subdirectory per person (i.e., 5.749 subdirectories), named
    after that person (e.g., `Aaron_Eckhart`).
    Each of these subdirectories contains images depicting this
    person.

    References
    ----------
    [1] http://vis-www.cs.umass.edu/lfw/
    """

    def __init__(self, key: str = None, lfw_data: str = None,
                 **kwargs) -> None:
        """Initialize the Labeled Faces in the Wild (LFW) dataset.

        Parameters
        ----------
        lfw_data: str
            The path to the LFW root directory. This directory
            should contain the 5.749 subdirectories holding images
            of the known persons.
        """
        directory = '/net/projects/data/lfw/lfw' # FIXME[hack]
        description = "Labeled Faces in the Wild"
        super().__init__(key=key or "lfw",
                         directory=directory,
                         description=description,
                         label_from_directory='name',
                         **kwargs)
