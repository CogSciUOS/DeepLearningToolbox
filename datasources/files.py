from datasources import DataSource, InputData
from util import imread

from os.path import join
import numpy as np

# FIXME[todo]: maybe combined with DataDirectory to profit from
# common features, like prefetching, caching, etc.


class DataFiles(DataSource):
    """Data source for reading from a collection of files.

    Attributes
    ----------
    _filenames   :  list of str
                    The names of the files from which the data are read.
    """

    _dirname: str = None
    _filenames: list = None
    _current_index: int = 0
    _current_data: np.ndarray = None

    def __init__(self, filenames: list, dirname=None):
        """Create a new DataFiles data source.

        Parameters
        ----------
        filename    :   list of str
                        Names of the files containing the data
        """
        super().__init__()
        self._dirname = dirname
        if filenames is not None:
            self.setFilenames(filenames)

    def setFilenames(self, filenames: list):
        """Set the data file to be used.

        Parameters
        ----------
        filename    :   list of str
                        Name of the files containing the data
        """
        self._filenames = filenames
        self._current_index = 0

    def __getitem__(self, index: int):
        """Provide access to the records in this data source."""
        self._current_index = index
        filename = self.getFile()
        self._current_data = imread(filename)
        return InputData(self._current_data, filename)

    def __len__(self):
        """Get the number of entries in this data source."""
        return len(self._filenames)

    def getFile(self) -> str:
        """Get the underlying file name"""
        filename = self._filenames[self._current_index]
        if self._dirname is not None:
            filename = join(self._dirname, filename)
        return filename

    def __str__(self):
        return f'<DataFiles "{self._filename[self._current_index]}"'
