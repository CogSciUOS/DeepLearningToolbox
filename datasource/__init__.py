""".. moduleauthor:: Rasmus Diederichsen, Ulf Krumnack

.. module:: datasource

This module includes the various ways in which data can be
loaded. This includes loading individual files (e.g. images) from a
directory, files containing complete datasets (e.g., numpy or matlab
arrays), databases, grabbing images from the webcam.

The module also has some knowledge of a small collection of predefined
data sets (like "mnist", "keras", "imagenet", ...) more to be added.
For these datasets, it (should) know how to download the dataset, how
to locate it (e.g., by means of environment variables or some standard
locations), and how access it (i.e., which Datasource class to use).

Datasources can have different

* memory: the complete data is in memory

* random_access: data in file(s) or in some database

* sequential: data is read from some stream (movie, webcam)

"""

# standard imports
import sys
import logging

# toolbox imports
# FIXME[todo]: lazy import
# from util.lazy import lazy_begin, lazy_end
from . import predefined
from .data import Data, ClassScheme, ClassIdentifier
from .meta import Metadata
from .datasource import (Datasource, Labeled, Random, Sectioned,
                         Loop, Snapshot, Indexed, Imagesource)
from .fetcher import Datafetcher
from .controller import View, Controller
from .array import DataArray, LabeledArray
from .file import DataFile
from .files import DataFiles
from .directory import DataDirectory

# lazy_begin(): lazy import
from .video import Video
from .webcam import DataWebcam
# lazy_end(): lazy import
# FIXME[todo]:
# lazy_import('datasource.webcam', 'DataWebcam')
# lazy_import('datasource.video', 'Video')

# logging
LOG = logging.getLogger(__name__)
del logging

# FIXME[hack]: we do not really want to import that
# - we just want the code to be executed ...
del predefined

LOG.info("Predefined data sources: %r", list(Datasource.register_keys()))

# FIXME[old]: may be used at some other place ...
# for keras_id in KerasDatasource.KERAS_IDS:
#    try:
#        KerasDatasource(keras_id)
#    except ValueError as err:
#        LOG.error("Error instantiating keras data source '%s': %s",
#                  keras_id, err)
