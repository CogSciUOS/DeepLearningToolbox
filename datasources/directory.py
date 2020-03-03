from datasources import Datasource, InputData, Indexed
from util.image import imread

from os.path import join, isdir, isfile
from os import listdir
from glob import glob
import random


class DataDirectory(Datasource, Indexed):
    """A data directory contains data entries (e.g., images), in
    individual files. Each file is only read when accessed.

    Attributes
    ----------
    _dirname: str
        A directory containing input data files. Can be None.
    _filenames: list
        A list of filenames in the data directory (that is, filenames
        relative to the data directory). An empty list
        indicates that no suitable files where found in the directory.
        None means that the list has not yet been prepared.
    """
    _dirname: str = None
    _filenames: list = None

    def __init__(self, dirname: str = None, description: str=None, **kwargs):
        """Create a new DataDirectory

        Parameters
        ----------
        dirname :   str
                    Name of the directory with files
        """
        description = description or f"directory {dirname}"
        super().__init__(description=description, **kwargs)
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

        Raises
        ------
        FileNotFoundError:
            The directory does not exist.
        """
        
        if dirname and not isdir(dirname):
            FileNotFoundError(f"No such directory: {dirname}")
        if self._dirname != dirname:
            self._dirname = dirname
            self._filenames = None
            self.change('metadata_changed')

    @property
    def prepared(self):
        """Check if this Directory has been prepared.
        """
        return self._filenames is not None

    def _prepare_data(self):
        """
        """
        if not self._dirname:
            raise RuntimeError("No directory was specificed for DataDirectory")

        if self._filenames is None:
            self._prepare_filenames()

    def _prepare_filenames(self):
        """Prepare the list of filenames maintained by this
        :py:class:`DataDirectory`.

        The default behaviour is to collect all files in the
        directory. Subclasses may implement alternative methods to
        collect filenames.
        """
        # self._filenames = [f for f in listdir(self._dirname)
        #                    if isfile(join(self._dirname, f))]
        self._filenames = glob(join(self._dirname, "**", "*.*"),
                               recursive=True)

    def __len__(self):
        return self.prepared and len(self._filenames) or 0

    def __str__(self):
        return f'<DataDirectory "{self._dirname}">'

    def _fetch(self, index=None, **kwargs):
        if index is None:
            self._fetch_random(**kwargs)
        else:
            image_file = self._filenames[index]
            self._image = imread(image_file)
            self._index = index


    @property
    def fetched(self):
        return self._image is not None

    def _get_data(self):
        """The actual implementation of the :py:meth:`data` property
        to be overwritten by subclasses.

        It can be assumed that a data point has been fetched when this
        method is invoked.
        """
        return self._image

    def _fetch_index(self, index, **kwargs) -> None:
        if isinstance(index, int):
            filename = self._filenames[index]
        elif isinstance(index, str):
            # FIXME[todo]: This can be improved by reverse lookup table
            filename = index
            index = self._filenames.index(filename)

        self._data = self.load_data(filename)
        self._index = index

    def load_data(self, filename):
        return imread(join(self._dirname, filename))
        
    # numerical index of the currently selected file
    _index: int = 0

    @property
    def index(self):
        return self._index
