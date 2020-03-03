from base import View as BaseView, Controller as BaseController, change, run
from .source import Datasource, Labeled, Loop, Indexed
from .array import DataArray
from .directory import DataDirectory
from .file import DataFile

import logging
import numpy as np


class View(BaseView, view_type=Datasource):
    """Base view backed by a datasource. Contains functionality for
    viewing a :py:class:`Datasource`.

    Attributes
    ----------
    _datasource : Datasource
        The current :py:class:`datasources.Datasource`
    """
    _logger = logging.getLogger(__name__)


    def __init__(self, datasource: Datasource=None, **kwargs):
        super().__init__(observable=datasource, **kwargs)

    
class Controller(View, BaseController):
    """Base controller backed by a datasource. Contains functionality for
    manipulating input data from a data source.

    Attributes
    ----------
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __len__(self) -> int:
        """Returns the number of elements in the currently selected
        datasource.

        Returns
        -------
        len: int
            The number of data points in this Datasource or None
            if no datsources is viewed or the number cannot be determined.
        """
        return 0 if self._datasource is None else len(self._datasource)

    @property
    def description(self):
        kwargs = {}
        if self.isinstance(Labeled):
            kwargs['with_label'] = True
        if self.isinstance(DataArray):
            kwargs['index'] = self._datasource.index
        return self._datasource.get_description(**kwargs) if self else "No data"

    @property
    def index(self) -> int:
        """Get the current index in the :py:class:`Datasource`.

        Result
        ------
        index: int
            The current index of this
            :py:class:`DatasourceController`.
        """
        if not self.isinstance(Indexed):
            raise TypeError("No index available for datasource of type "
                            f"{type(self._datasource)}")
        return self._datasource.index

    @run
    def set_index(self, index: int) -> None:
        """Set the current index in the :py:class:`Datasource`.

        Parameters
        ----------
        index: int
            The index to become the current index in this
            :py:class:`DatasourceController`.
        """
        self._logger.info(f"DatasourceController.set_index(index)")
        if index is None:
            index = 0
        elif self._datasource is None or len(self._datasource) < 1:
            index = None
        elif index < 0:
            index = 0
        elif index >= len(self._datasource):
            index = len(self._datasource) - 1

        if index != self._datasource.index:
            self._datasource.fetch(index=index)

    @run
    def random(self):
        """Select a random index into the dataset."""
        self._datasource.fetch(random=True)

    def advance(self):
        """Advance data index to end."""
        if self.isinstance(Indexed) and self.prepared:
            n_elems = len(self)
            self.set_index(n_elems - 1)

    def advance_one(self):
        """Advance data index by one."""
        if self.isinstance(Indexed) and self.prepared:
            n_elems = len(self)
            self.set_index(min(self._datasource.index + 1, n_elems))

    def rewind(self):
        """Reset data index to zero."""
        if self.isinstance(Indexed) and self.prepared:
            self.set_index(0)

    def rewind_one(self):
        """Rewind data index by one."""
        if self.isinstance(Indexed) and self.prepared:
            self.set_index(max(0, self._datasource.index - 1))

    def edit_index(self, index):
        """Set the current dataset index.  Index is left unchanged if out of
        range.

        Parameters
        ----------
        index   :   int

        """
        if index is not None:
            try:
                index = int(index)
                if index < 0:
                    raise ValueError('Index out of range')
            except ValueError:
                index = self._index
        self.set_index(index)

    def set_data_array(self, data: np.ndarray=None):
        """Set the data array to be used.

        Parameters
        ----------
        data:
            An array of data. The first axis is used to select the
            data record, the other axes belong to the actual data.
        """
        self(DataArray(data))

    def set_data_file(self, filename: str):
        """Set the data file to be used."""
        self(DataFile(filename))

    def set_data_directory(self, dirname: str=None):
        """Set the directory to be used for loading data."""
        self(DataDirectory(dirname))

    def loop(self, looping: bool=None):
        """Start or stop looping through the Datasource.
        This will fetch one data point after another.
        This is mainly intended to display live input like
        movies or webcam, but it can also be used for other Datasources 
        """
        if not isinstance(self._datasource, Loop):
            return
        if looping is not None and (looping == self.looping):
            return
        
        if self.looping:
            self._logger.info("Stopping datasource loop")
            self.stop_loop()
        else:
            self._logger.info("Starting datasource loop")
            self.start_loop()
            self._runner.runTask(self._datasource.run_loop)
