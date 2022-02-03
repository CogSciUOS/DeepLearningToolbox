""".. moduleauthor:: Rasmus Diederichsen, Ulf Krumnack

.. module:: datasource

The :py:mod:`dltb.datasource` module provides the abstract base
class :py:class:`Datasource` for creating datasources.
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
import logging

# Toolbox imports
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

ABC = 3

# logging
LOG = logging.getLogger(__name__)
del logging


Datasource.register_instance('Noise', __name__ + '.noise', 'Noise',
                             shape=(100, 100, 3))

Datasource.register_instance('Webcam', __name__ + '.webcam', 'DataWebcam')
Datasource.register_instance('Webcam2', __name__ + '.webcam', 'DataWebcam',
                             device=1)


# FIXME[todo]: only for debugging ...
Datasource.register_instance('dummy', 'dltb.datasource.dummy', 'Dummy')


LOG.info("Predefined data sources: %r",
         list(Datasource.instance_register.keys()))


__all__ = ['Datasource', 'Labeled', 'Random', 'Sectioned',
           'Datafetcher', 'Livesource', 'Indexed', 'Imagesource',
           'Imagesourcelike', 'DataArray', 'LabeledArray', 'DataFile',
           'DataFiles', 'DataDirectory', 'ImageDirectory', 'Noise',
           'Video', 'Thumbcinema', 'DataWebcam']
