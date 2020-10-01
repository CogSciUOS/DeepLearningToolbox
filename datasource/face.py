"""Collection of face datasets.
"""

# standard imports
from typing import Tuple
import os
import errno
from glob import glob

# third party imports
import numpy as np

# toolbox imports
from util.image import Landmarks, Region
from dltb.base.data import Data
from .datasource import Imagesource
from .directory import DataDirectory


class MSCeleb1M(DataDirectory, Imagesource):
    # pylint: disable=too-many-ancestors
    """The MS-Celeb-1M is a large scale dasataset for face
    recognition.

    There exists a preprocessed version of this dataset with
    all images cropped to 112x112 pixel. The dataset circulates
    in the net under the name `ms1m_align_112.zip` (with MD5 checksum
    `950f941021f5fd6bfa9da8eb8efc7f9d`).
    It contains 5,822,653 images of 85,742 persons.
    The images are aranged in subdirectories, one directory
    per person.
    """

    def __init__(self, key: str = "ms-celeb-1m", **kwargs) -> None:
        """Initialize the MS-Celeb-1M Face recognition dataset.
        """
        # os.getenv('MSCELEB1M_DATA', '.')
        msceleb1m_data = '/net/projects/data/MS-Celeb-1M/align_112'
        super().__init__(key=key, directory=msceleb1m_data,
                         description=f"MS-Celeb-1M Faces", **kwargs)

    def __str__(self):
        return 'MS-Celeb-1M'

    def _prepare(self, **kwargs) -> None:
        # pylint: disable=arguments-differ
        """Prepare the MS-Celeb-1M Face recognition dataset. This
        will arange a list of all images provided by the dataset,
        either by reading in a cache file, or by traversing the directory.
        """
        cache = f"msceleb1m_filelist.p"
        super()._prepare(filenames_cache=cache, **kwargs)
