from datasources import DataSource, InputData
from scipy.misc import imread
from os.path import join, isdir, isfile
from os import listdir
from glob import glob

class DataDirectory(DataSource):
    '''A data directory contains data entries (e.g., images), in
    individual files. Each files is only read when accessed.

    Attributes
    ----------
    _dirname    :   str
                    A directory containing input data files. Can be None.
    _filenames  :   list
                    A list of filenames in the data directory. An empty list indicates that no
                    suitable files where found in the directory.
    '''
    _dirname: str = None
    _filenames: list = None

    def __init__(self, dirname: str = None):
        '''Create a new DataDirectory

        Parameters
        ----------
        dirname :   str
                    Name of the directory with files
        '''
        super().__init__(dirname)
        self.setDirectory(dirname)

    def setDirectory(self, dirname: str):
        '''Set the directory to load from

        Parameters
        ----------
        dirname :   str
                    Name of the directory with files
        '''
        if self._dirname != dirname:
            self._dirname = dirname
            self._filenames = None

    def prepare(self):
        if self._dirname is not None and self._filenames is None:
#            self._filenames = [f for f in listdir(self._dirname)
#                               if isfile(join(self._dirname, f))]
            self._filenames = glob(join(self._dirname, "**", "*.*"),
                                   recursive=True)
        

    def getDirectory(self) -> str:
        return self._dirname

    def __getitem__(self, index):
        if not self._filenames:
            return None, None
        else:
            # TODO: This can be much improved by caching and/or prefetching
            filename = self._filenames[index]
            data = imread(join(self._dirname, filename))
            return InputData(data, None)

    def __len__(self):
        if not self._filenames:
            return 0
        return len(self._filenames)

    def __str__(self):
        return f'<DataDirectory "{self._dirname}"'

    def getName(self, index=None) -> str:
        if index is None:
            return self._description
        elif self._targets is None:
            return self._filenames[index]
        else:
            return self._filenames[index] + ", target=" + str(self._targets[index])
