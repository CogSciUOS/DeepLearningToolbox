from base import View as BaseView, Controller as BaseController, change
from .source import Datasource
from .array import DataArray
from .directory import DataDirectory
from .file import DataFile

import logging
import threading
from threading import Event
from random import randint
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
    _index : int
        The current index in the :py:class:`datasources.Datasource`
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._index = 0
        self._loop_running = False
        self._loop_event = None

    # FIXME[old]: should be removed
    def get_observable(self) -> Datasource:
        return self._datasource

    def __len__(self) -> int:
        """Returns the number of elements in the currently selected
        datasource.

        Returns
        -------
        int

        """
        return 0 if self._datasource is None else len(self._datasource)

    # FIXME[old]: reimplemented by __call__() ...
    #@change
    def set_datasource(self, datasource: Datasource) -> None:
        """Set the :py:class:`Datasource` used by this DatasourceController.

        Parameters
        ----------
        datasource: Datasource
            The new Datasource.

        """
        # FIXME[async]: preparation may take some time - maybe do this
        # asynchronously
        datasource.prepare()
        self(datasource)
        self.set_index(0)
        self.change(datasource_changed=True)

    def get_datasource(self) -> Datasource:
        """Get the :py:class:`Datasource` used by this
        :py:class:`DatasourceController`.

        Result
        ------
        datasource: Datasource
            The :py:class:`Datasource` of this
            :py:class:`DatasourceController`.
        """
        return self._datasource

    #@change
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

        if index == self._index and len(self._datasource) > 0:
            return

        self._index = index
        self.change(index_changed=True, data_changed=True)

    @property
    def data(self) -> np.ndarray:
        data, _ = self._datasource[self._index] if self else None
        return data

    @property
    def label(self):
        _, label = self._datasource[self._index] if self else None
        return label

    @property
    def data_and_label(self):
        return self._datasource[self._index] if self else (None, None)

    @property
    def description(self):
        return (self._datasource.get_description(index=self._index,
                                                 target=True) if self
                else "No data")

    def get_index(self) -> int:
        """Get the current index in the :py:class:`Datasource`.

        Result
        ------
        index: int
            The current index of this
            :py:class:`DatasourceController`.
        """
        return self._index

    def onSourceSelected(self, source: Datasource):
        """Set a new :py:class:`datasources.Datasource`.

        Parameters
        ----------
        source  :   datasources.Datasource
        """
        self._runner.runTask(self.set_datasource, source)

    def random(self):
        """Select a random index into the dataset."""
        n_elems = len(self)
        index = randint(0, n_elems)
        self._runner.runTask(self.set_index, index)
        #self.set_index(index)

    def advance(self):
        """Advance data index to end."""
        n_elems = len(self)
        self._runner.runTask(self.set_index, n_elems - 1)

    def advance_one(self):
        """Advance data index by one."""
        self._runner.runTask(self.set_index, self._index + 1)

    def rewind(self):
        """Reset data index to zero."""
        self._runner.runTask(self.set_index, 0)

    def rewind_one(self):
        """Rewind data index by one."""
        self._runner.runTask(self.set_index, self._index - 1)

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
        self.set_datasource(DataArray(data))

    def set_data_file(self, filename: str):
        """Set the data file to be used."""
        self.set_datasource(DataFile(filename))

    def set_data_directory(self, dirname: str=None):
        """Set the directory to be used for loading data."""
        self.set_datasource(DataDirectory(dirname))


    def loop(self):
        if self._loop_running:
            self._logger.info("Stopping datasource loop")
            self._loop_running = False
        else:
            self._logger.info("Starting datasource loop")
            self._loop_event = Event()
            self._loop_running = True
            self._runner.runTask(self._loop)

    def _loop(self):
        self._logger.info("Running datasource loop")
        while self._loop_running:
            self.random()
            #self.advance_one()
            self._loop_event.clear()
            self._loop_event.wait(timeout=.2)
