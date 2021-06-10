"""Base class for datasources that read data from files in a
directory.

"""

# standard imports
from typing import Union, Iterable
import os
import glob
import random
import logging

# toolbox imports
from dltb.base.data import Data
from dltb.tool.classifier import ClassScheme
from dltb.util import read_cache, write_cache, cache_path
from .files import DataFiles
from .datasource import Imagesource

# logging
LOG = logging.getLogger(__name__)


class DataDirectory(DataFiles):
    # pylint: disable=too-many-ancestors
    # Too many ancestors (13/7)
    # pylint: disable=abstract-method
    # Method 'load_datapoint_from_file' is abstract in class 'Datasource'
    """A data directory contains data entries (e.g., images), in
    individual files. Each file is only read when accessed.

    Attributes
    ----------
    _directory: str
        A directory containing input data files. Can be None.
    _filenames: list
        A list of filenames in the data directory (that is, filenames
        relative to the data directory). An empty list
        indicates that no suitable files where found in the directory.
        None means that the list has not yet been prepared.
    suffix: str
        The suffix or suffixes of filenames of interest (default: '`*`').
    """

    def __init__(self, directory: str = None,
                 suffix: Union[str, Iterable[str]] = '*',
                 description: str = None,
                 label_from_directory: Union[bool, str] = False,
                 scheme: ClassScheme = None, **kwargs) -> None:
        """Create a new DataDirectory

        Parameters
        ----------
        directory: str
            Name of the directory with files
        label_from_directory:
            Use the directory name as class labels. If True, the
            directory name is taken as class label. If a string,
            it should indicate a format in the py:class:`ClassScheme`
            of this :py:class:`DataDirectory` and the directory name
            will be looked up from that scheme.
        """
        description = description or f"directory {directory}"
        super().__init__(directory=directory, description=description,
                         **kwargs)
        self._label_from_directory = label_from_directory
        self._suffix = suffix
        self._scheme = scheme

    def __str__(self):
        """String representation of this :py:class:`DataDirectory`.
        """
        return f'<DataDirectory "{self._directory}">'

    def _set_directory(self, directory: str):
        """(Re)set the directory. Changing the directory requires
        to prepare this :py:class:`DataDirectory` again.
        """
        super()._set_directory(directory)
        self.unprepare()

    @property
    def suffix(self) -> str:
        """The suffix of files of interest in the directory. Only
        files with that suffix will be considered when listing the
        directory.  The suffix may be `None` to indicate that no suffix
        checks should be performed, or `'*'` to allows for arbitrary
        suffixes (but not files without any suffix).
        """
        return self._suffix

    #
    # Preparable
    #

    def _preparable(self) -> bool:
        """Check if this :py:class:`DataDirectory` can be prepared.
        """
        return (bool(self._directory) and
                os.path.isdir(self.directory) and
                super()._preparable())

    def _prepare(self, filenames_cache: str = None, **kwargs) -> None:
        # pylint: disable=arguments-differ
        """Prepare this :py:class:`DataDirectory`.
        This will build up the list of filenames, either by traversing
        the directory, or by using a cache file.

        Parameters
        ----------
        filenames_cache: str
            Name of a cache file to store the list of filenames. The
            `filenames_cache` is interpreted relative to the global
            cache directory, as determined by :py:func:`dltb.util.cache_path`.
        """
        LOG.debug("Preparing DataDirectory: %s", self.directory)
        super()._prepare(**kwargs)
        if filenames_cache is None:
            filenames_cache = \
                type(self).__name__.lower() + '-' + self.key + '-filenames.p'
        self._filenames = filenames_cache and read_cache(filenames_cache)
        if self._filenames is None:
            LOG.info("DataDirectory: creating new cache file '%s' for "
                     "data directory '%s'",
                     cache_path(filenames_cache), self.directory)
            self._prepare_filenames()
            write_cache(filenames_cache, self._filenames)
        else:
            LOG.debug("DataDirectory: read %d filenames from cache '%s' for "
                      "data directory '%s'", len(self._filenames),
                      cache_path(filenames_cache), self.directory)
        if self._scheme is None:
            directories = set()
            for name in self._filenames:
                directories.add(os.path.dirname(name))
            self._scheme = ClassScheme(len(directories))
            self._scheme.add_labels(directories,
                                    self._label_from_directory or 'text',
                                    lookup=True)

    def _unprepare(self) -> None:
        self._filenames = None
        super()._unprepare()

    def _prepare_filenames(self) -> None:
        """Prepare the list of filenames maintained by this
        :py:class:`DataDirectory`.

        The default behaviour is to collect all files in the
        directory. Subclasses may implement alternative methods to
        collect filenames.
        """
        # self._filenames = \
        #         [f for f in os.listdir(self._directory)
        #          if os.path.isfile(os.path.join(self._directory, f))]
        suffixes = \
            (self.suffix, ) if isinstance(self.suffix, str) else self.suffix
        filenames = []
        for suffix in suffixes:
            pattern = os.path.join(self._directory, "**", f"*.{suffix}")
            filenames.extend(glob.glob(pattern, recursive=True))
        # we need relative filenames!
        index = len(str(self._directory)) + 1
        self._filenames = [filename[index:] for filename in filenames]
        LOG.info("DataDirectory: prepared %d filenames from directory '%s'.",
                 len(self._filenames), self._directory)

    #
    # Data
    #

    def _get_meta(self, data: Data, **kwargs) -> None:
        # pylint: disable=arguments-differ
        """Get metadata for some data.
        """
        if self._label_from_directory:
            data.add_attribute('label', batch=True)
        super()._get_meta(data, **kwargs)

    def _get_data_from_file(self, data: Data, filename: str) -> None:
        """Get data from a given file.
        """
        super()._get_data_from_file(data, filename)
        self._get_label_for_filename(data, filename)

    def _get_label_for_filename(self, data: Data, filename: str) -> None:
        """Get a class label for the file from the directory name.
        """
        if self._label_from_directory is True:
            data.label = os.path.dirname(filename)
        elif self._label_from_directory:
            data.label = \
                self._scheme.identifier(os.path.dirname(filename),
                                        lookup=self._label_from_directory)

    # FIXME[todo]: currently not used - provide a way to work without filelist
    def _get_random_without_filelist(self, data: Data) -> None:
        """Get a random datapoint in a 2-level hierarchy in cases where
        no filelist was build up.
        """
        # FIXME[todo]: currently there is no list of subdirectories
        #              (self._subdirs)
        subdir = random.choice(self._subdirs)
        name = random.choice(os.listdir(os.path.join(self.directory, subdir)))
        self._get_data_from_file(data, os.path.join(subdir, name))


class ImageDirectory(DataDirectory, Imagesource):
    """Images read from a directory.
    """
