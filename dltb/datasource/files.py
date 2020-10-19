"""Datasource reading from a collection of files.
"""

# standard imports
from typing import List
import os
from abc import abstractmethod

# toolbox imports
from dltb.base.data import Data
from .datasource import Datasource, Indexed


# FIXME[todo]: maybe combined with DataDirectory to profit from
# common features, like prefetching, caching, etc.


class DataFiles(Indexed, Datasource):
    """Data source for reading from a collection of files.

    Attributes
    ----------
    filenames: List[str]
        The names of the files from which the data are read.
    directory: str
        Base directory relative to which filnames are to be
        interpreted (unless fully qualified).
    """

    def __init__(self, filenames: List[str] = None,
                 directory: str = None, **kwargs) -> None:
        """Create a new DataFiles data source.

        Parameters
        ----------
        filename    :   list of str
                        Names of the files containing the data
        """
        super().__init__(**kwargs)
        self._directory = directory
        self._filenames = filenames

    def __str__(self):
        return f'<DataFiles with "{self.prepared and len(self)} files>'

    def __len__(self):
        """The length of a :py:class:`DataFiles` is the number
        of files in the filelist.
        """
        return len(self._filenames) if self.prepared else 0

    @property
    def directory(self):
        """The base directory relative to which (relative) filenames
        are to be interpreted.
        """
        return self._directory

    @directory.setter
    def directory(self, directory: str):
        """Set the base directory relative to which (relative) filenames
        are to be interpreted.
        """
        if directory != self.directory:
            self._set_directory(directory)

    def _set_directory(self, directory: str):
        self._directory = directory

    #
    # Preparable
    #

    def _preparable(self) -> bool:
        """This :py:class:`DataFiles` object can only be prepared if
        the base directory exists.
        """
        return (self.directory is not None and
                os.path.isdir(self.dirname) and
                super()._preparable())

    def _prepared(self) -> bool:
        """Check if this :py:class:`DataFiles` object has been prepared.
        """
        return self._filenames is not None

    # FIXME[todo]: concept - provide some idea how the filelist can be
    # prepared (including cache files) - combine this with DataDirectory

    #
    # Data
    #

    def _get_meta(self, data: Data, filename: str = None, **kwargs) -> None:
        # pylint: disable=arguments-differ
        if filename is not None and not data.datasource_argument:
            data.datasource_argument = 'filename'
            data.datasource_value = filename
        data.add_attribute('filename', batch=True)
        if self._filenames is not None:
            data.add_attribute('index', batch=True)
        if self._loader_kind != 'array':
            data.add_attribute(self._loader_kind, batch=True)
        super()._get_meta(data, **kwargs)

    def _get_data(self, data: Data, filename: str = None, **kwargs) -> None:
        # pylint: disable=arguments-differ
        """

        Attributes
        ----------
        filename: str
            The filename (relative to this directory).
        index: int
            A numerical index of the file. This argument can only be
            used, if the :py:class:`DataFiles` has initialized
            a filename register.

        Raises
        ------
        ValueError:
            If the index is present, but the :py:class:`DataFiles`
            has no filename register.
        """
        if filename and not data:
            self._get_data_from_file(data, filename)
        super()._get_data(data, **kwargs)

    @abstractmethod
    def _get_data_from_file(self, data: Data, filename: str):
        """Get data from a given file.

        Arguments
        ---------
        filename: str
            The filename (relative to this directory).

        Notes
        -----
        Subclasses implementing this method may utilize
        :py:meth:`load_datapoint_from_file` to load the actual data.

        """
        abs_filename = os.path.join(self.directory, filename)
        setattr(data, self._loader_kind,
                self.load_datapoint_from_file(abs_filename))
        data.filename = abs_filename
        if self._filenames is not None and not hasattr(data, 'index'):
            # FIXME[todo]: This can be improved by reverse lookup table
            data.index = self._filenames.index(filename)

    def _get_index(self, data: Data, index: int, **kwargs) -> None:
        """Implementation of the :py:class:`Indexed` interface. Lookup
        up data point by its index in the file list.
        """
        if self._filenames is None:
            raise ValueError(f"Access by index ({index}) is disabled: "
                             "no filename register was provided.")
        data.index = index
        self._get_data_from_file(data, self._filenames[index])
