from datasources import Datasource, InputData
from scipy.misc import imread
from os.path import join, isdir, isfile
from os import listdir
from glob import glob


class DataDirectory(Datasource):
    """A data directory contains data entries (e.g., images), in
    individual files. Each files is only read when accessed.

    Attributes
    ----------
    _dirname:   str
        A directory containing input data files. Can be None.
    _filenames:   list
        A list of filenames in the data directory. An empty list
        indicates that no suitable files where found in the directory.
        None means that the list has not yet been initialized.
    """
    _dirname: str = None
    _filenames: list = None

    def __init__(self, dirname: str = None):
        """Create a new DataDirectory

        Parameters
        ----------
        dirname :   str
                    Name of the directory with files
        """
        super().__init__(dirname)
        self.directory = dirname

    @property
    def directory(self):
        return self._dirname

    @directory.setter
    def directory(self, dirname: str):
        """Set the directory to load from

        Parameters
        ----------
        dirname: str
            Name of the directory with files
        """
        
        if dirname and not isdir(dirname):
            FileNotFoundError(f"No such directory: {dirname}")
        if self._dirname != dirname:
            self._dirname = dirname
            self._filenames = None
            self.change('metadata_changed')

    @property
    def prepared(self):
        return self._filenames is not None

    def prepare(self):
        if self.prepared:
            return  # nothing to do ...

        if not self._dirname:
            raise RuntimeError("No directory was specificed for DataDirectory")

        if self._filenames is None:
            # self._filenames = [f for f in listdir(self._dirname)
            #                    if isfile(join(self._dirname, f))]
            self._filenames = glob(join(self._dirname, "**", "*.*"),
                                   recursive=True)
        self.change('state_changed')

    def __getitem__(self, index):
        if not self.prepared:
            return InputData(None, None)

        # TODO: This can be much improved by caching and/or prefetching
        filename = self._filenames[index]
        data = imread(join(self._dirname, filename))
        return InputData(data, None)

    def __len__(self):
        return self.prepared and len(self._filenames) or 0

    def __str__(self):
        return f'<DataDirectory "{self._dirname}">'

    def _description_for_index(self, index: int) -> str:
        return self.prepared and self._filenames[index] or ''
