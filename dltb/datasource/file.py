"""Definition of datasources that provide data from a single file.
"""

# standard imports
import os

# third party imports
import numpy as np

# toolbox imports
from .array import DataArray


class DataFile(DataArray):
    """Data source for reading from a file.

    Attributes
    ----------
    filename: str
        The name of the file from which the data are read.
    """

    def __init__(self, filename: str):
        """Create a new data file.

        Parameters
        ----------
        filename: str
            Name of the file containing the data
        """
        super().__init__()
        self._filename = filename

    def __str__(self):
        return f'<DataFile "{self._filename}"'

    @property
    def filename(self) -> str:
        """Get the file name"""
        return self._filename

    @filename.setter
    def filename(self, filename: str) -> None:
        """Set the data file to be used. Changing the filename
        invalidates the current data and requires preparing the
        Datasource again.

        Parameters
        ----------
        filename: str
            Name of the file containing the data.
        """
        self._filename = filename
        self.unprepare()

    #
    # Preparation
    #

    def _prepare(self) -> None:
        self._array = np.load(self.filename, mmap_mode='r')
        self._description = os.path.basename(self.filename)
