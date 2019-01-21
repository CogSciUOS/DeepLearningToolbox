""".. moduleauthor:: Rasmus Diederichsen, Ulf Krumnack

.. module:: datasources

This module includes the various ways in which data can be
loaded. This includes loading individual files (e.g. images) from a
directory, files containing complete datasets (e.g., numpy or matlab
arrays), databases, grabbing images from the webcam.

The module also has some knowledge of a small collection of predefined
data sets (like "mnist", "keras", "imagenet", ...) more to be added.
For these datasets, it (should) know how to download the dataset, how
to locate it (e.g., by means of environment variables or some standard
locations), and how access it (i.e., which DataSource class to use).

"""
from .source import DataSource, InputData, Predefined
from .array import DataArray
from .file import DataFile
from .files import DataFiles
from .directory import DataDirectory
from .webcam import DataWebcam
from .video import DataVideo
from .keras import KerasDataSource
from .imagenet import ImageNet

import logging
logger = logging.getLogger(__name__)
del logging

import sys

for d in KerasDataSource.KERAS_IDS:
    try:
        KerasDataSource(d)
    except ValueError as err:
        print(f"Error instantiating keras data source '{d}': {err}",
              file=sys.stderr)
ImageNet(section='train')

logger.info(f"Predefined data sources: {Predefined.get_data_source_ids()}")

