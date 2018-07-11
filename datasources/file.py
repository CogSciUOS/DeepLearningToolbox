import numpy as np
from os.path import isfile, isdir, join, basename

from datasources import DataArray

class DataFile(DataArray):
    '''Data source for reading from a file.

    Attributes
    ----------
    _filename   :   str
                    The name of the file from which the data are read.
    '''

    _filename: str = None

    def __init__(self, filename: str):
        '''Create a new data file.

        Parameters
        ----------
        filename    :   str
                        Name of the file containing the data
        '''
        super().__init__()
        if filename is not None:
            self.setFile(filename)

    def setFile(self, filename: str):
        '''Set the data file to be used.

        Parameters
        ----------
        filename    :   str
                        Name of the file containing the data
        '''
        self._filename = filename
        data = np.load(filename, mmap_mode='r')
        self.setArray(data, basename(self._filename))

    def getFile(self) -> str:
        '''Get the underlying file name'''
        return self._filename

    def __str__(self):
        return f'<DataFile "{self._filename}"'
