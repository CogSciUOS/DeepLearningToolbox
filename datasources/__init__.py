'''.. moduleauthor:: Rasmus Diederichsen, Ulf Krumnack

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

'''
from .source import DataSource, InputData
from .array import DataArray
from .file import DataFile
from .files import DataFiles
from .directory import DataDirectory
from .webcam import DataWebcam
from .video import DataVideo
from .set import DataSet
from .imagenet import ImageNet

#
# Predefined Datasources
#

def get_datasource(public_id):
    if public_id == "imagenet":
        return ImageNet
    raise ValueError(f"Unknown datasource name '{public_id}'")

datasources = ["mnist", "cifar10", "cifar100", "imagenet"]
