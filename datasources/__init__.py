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

Datasources can have different
* memory: the complete data is in memory
* random_access: data in file(s) or in some database
* sequential: data is read from some stream (movie, webcam)

"""
from .source import Datasource, InputData, Predefined
from .source import Datasource as DataSource, Labeled, Random, Loop
from .controller import View, Controller
from .array import DataArray, LabeledArray
from .file import DataFile
from .files import DataFiles
from .directory import DataDirectory
from .webcam import DataWebcam
from .video import DataVideo
from .keras import KerasDatasource
from .imagenet import ImageNet
from .dogsandcats import DogsAndCats
from .widerface import WiderFace
from .noise import DataNoise

import logging
logger = logging.getLogger(__name__)
del logging

import sys

for d in KerasDatasource.KERAS_IDS:
    try:
        KerasDatasource(d)
    except ValueError as err:
        print(f"Error instantiating keras data source '{d}': {err}",
              file=sys.stderr)
#ImageNet(section='train')
ImageNet(section='val')
DataNoise(shape=(100,100,3))
DogsAndCats()
WiderFace()

if DataWebcam.check_availability():
    DataWebcam()

logger.info(f"Predefined data sources: {Predefined.get_data_source_ids()}")

