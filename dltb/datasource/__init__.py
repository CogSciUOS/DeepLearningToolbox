"""The :py:mod:`dltb.datasource` module provides the abstract base
class :py:class:`Datasource` for createing datasources.
"""
from .datasource import (Datasource, Labeled, Random, Sectioned,
                         Livesource, Indexed, Imagesource, Imagesourcelike)
from .fetcher import Datafetcher

from .array import DataArray, LabeledArray
from .directory import DataDirectory, ImageDirectory
from .file import DataFile
from .files import DataFiles
from .noise import Noise
from .video import Video, Thumbcinema
from .webcam import DataWebcam
