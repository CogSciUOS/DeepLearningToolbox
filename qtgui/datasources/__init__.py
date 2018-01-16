'''
.. moduleauthor:: Rasmus Diederichsen

.. module:: datasources

This module includes the various ways in which data can be loaded.
'''
from .source import DataSource, InputData
from .array import DataArray
from .file import DataFile
from .set import DataSet
from .directory import DataDirectory
