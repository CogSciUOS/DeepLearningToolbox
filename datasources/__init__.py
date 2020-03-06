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

# FIXME[todo]: lazy import
#from util.lazy import lazy_begin, lazy_end


from .meta import Metadata
from .source import (Datasource, InputData, Labeled, Random,
                     Loop, Snapshot, Indexed, Imagesource)
from .source import Datasource as DataSource
from .predefined import Predefined
from .controller import View, Controller
from .array import DataArray, LabeledArray
from .file import DataFile
from .files import DataFiles
from .directory import DataDirectory

#lazy_begin(): lazy import
from .video import DataVideo
from .webcam import DataWebcam
#lazy_end(): lazy import
# FIXME[todo]
#lazy_import('datasources.webcam', 'DataWebcam')
#lazy_import('datasources.video', 'DataVideo')

import logging
logger = logging.getLogger(__name__)
del logging

Predefined.register_id('mnist-train', 'datasources.keras',
                       'KerasDatasource', name='mnist', section='train')
Predefined.register_id('mnist-test', 'datasources.keras',
                       'KerasDatasource', name='mnist', section='test')
Predefined.register_id('imagenet-val', 'datasources.imagenet',
                       'ImageNet', section='val')  # section='train'
Predefined.register_id('dogsandcats', 'datasources.dogsandcats',
                       'DogsAndCats')
Predefined.register_id('widerface', 'datasources.widerface', 'WiderFace')
Predefined.register_id('Helen', 'datasources.helen', 'Helen')
Predefined.register_id('Noise', 'datasources.noise', 'DataNoise',
                       shape=(100,100,3))
Predefined.register_id('Webcam', 'datasources.webcam', 'DataWebcam')


# FIXME[old]: may be used at some other place ...
import sys
if False:
    for d in KerasDatasource.KERAS_IDS:
        try:
            KerasDatasource(d)
        except ValueError as err:
            print(f"Error instantiating keras data source '{d}': {err}",
                  file=sys.stderr)

logger.info(f"Predefined data sources: {Predefined.get_data_source_ids()}")

