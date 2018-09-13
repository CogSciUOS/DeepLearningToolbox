from random import randint
import numpy as np

# FIXME[problem]: circular import
#   DataSource -> Model -> controller -> DataSource
# from model import Model
from controller import BaseController
from observer import Observer, Observable, BaseChange
from datasources import DataSource, DataArray, DataDirectory, DataFile



class DataSourceChange(BaseChange):
    """.. :py:class:: DataSourceChange

    A class whose instances are passed to observers in
    :py:meth:`observer.Observer.datasource_changed` in order to inform them
    as to the exact nature of the model's change.


    Attributes
    ----------
    datasource_changed: bool
        Whether the underlying :py:class:`datasources.DataSource`
        has changed
    input_changed: bool
        Whether the input signal changed
    """
    
    ATTRIBUTES = ['datasource_changed', 'index_changed']



def datasourcechange(datasource_changed=False,
                     index_changed=False):
    # FIXME[hack]: we need a better notification concecpt
    """A decorator with arguments that will determine the DataSourceChange
    state. This decorator is intended to be used for methods of the
    Model class that may require informing some observer.

    This decorator tries to be cautious to prevent sending unnecessary
    notifications by checking for actual changes. Hence it requires
    some knowledge on the internals of the model class, to be able
    detect relevant changes.
    """
    def datasourcechange_decorator(function):
        def wrapper(self, *args, **kwargs):
            change, notify = self.changelog()
            import threading
            me = threading.current_thread().name
            print(f"in[{me}]({function.__name__}): {notify}")
            if datasource_changed:
                datasource = self._datasource
            if index_changed:
                index = self._index
            function(self, *args, **kwargs)
            if datasource_changed:
                change.datasource_changed = (datasource == self._datasource)
            if index_changed:
                change.index_changed = (index == self._index)
            print(f"out[{me}]({function.__name__}):{change}")
            if notify:
                self.notifyObservers()
        return wrapper
    return datasourcechange_decorator

class DataSourceObserver(Observer):

    def datasource_changed(self, datasource: 'DataSourceController',
                           change: DataSourceChange):
        pass
    
class DataSourceController(BaseController, Observable):
    """Base controller backed by a datasource. Contains functionality for
    manipulating input data from a data source.

    Attributes
    ----------
    _model: model.Model
        The model containing the network. If some data are
        selected by this DataSourceController, they will
        provided as input to the model.
    _datasource : DataSource
        The current :py:class:`datasources.DataSource`
    _index : int
        The current index in the :py:class:`datasources.DataSource`
    """

    def __init__(self, model: 'model.Model',
                 datasource: DataSource = None, **kwargs) -> None:
        super().__init__(**kwargs)
        Observable.__init__(self, DataSourceChange, 'datasource_changed')
        self._model = model
        self._datasource = datasource
        self._index = 0

    def get_observable(self) -> Observable:
        return self

    def __len__(self) -> int:
        """Returns the number of elements in the currently selected
        datasource.

        Returns
        -------
        int

        """
        return 0 if self._datasource is None else len(self._datasource)

    @datasourcechange(datasource_changed=True)
    def set_datasource(self, datasource: DataSource) -> None:
        """Set the :py:class:`DataSource` used by this DataSourceController.

        Parameters
        ----------
        datasource: DataSource
            The new DataSource.

        """
        # FIXME[async]: preparation may take some time - maybe do this
        # asynchronously
        datasource.prepare()
        self._datasource = datasource
        self.set_index(0)

    def get_datasource(self) -> DataSource:
        """Get the :py:class:`DataSource` used by this
        :py:class:`DataSourceController`.

        Result
        ------
        datasource: DataSource
            The :py:class:`DataSource` of this
            :py:class:`DataSourceController`.
        """
        return self._datasource
    
    @datasourcechange(index_changed=True)
    def set_index(self, index: int) -> None:
        """Set the current index in the :py:class:`DataSource`.

        Parameters
        ----------
        index: int
            The index to become the current index in this
            :py:class:`DataSourceController`.
        """
        print(f"DataSourceController.set_index(index)")
        if (index is None or self._datasource is None or
            len(self._datasource) < 1):
            index = None
        elif index < 0:
            index = 0
        elif index >= len(self._datasource):
            index = len(self._datasource) - 1

        self._index = index
        if not self._datasource or index is None:
            pass
            # self.set_input_data(None, None) #  FIXME: Data cannot be None!
        else:
            data, target = self._datasource[index]
            description = self._datasource.get_description(index)
            self._model.set_input_data(data, target=target,
                                       description=description)

    def get_index(self) -> int:
        """Get the current index in the :py:class:`DataSource`.

        Result
        ------
        index: int
            The current index of this
            :py:class:`DataSourceController`.
        """
        return self._index

    def onSourceSelected(self, source: DataSource):
        """Set a new :py:class:`datasources.DataSource`.

        Parameters
        ----------
        source  :   datasources.DataSource
        """
        self._runner.runTask(self.set_datasource, source)

    def random(self):
        """Select a random index into the dataset."""
        n_elems = len(self)
        index = randint(0, n_elems)
        self._runner.runTask(self.set_index, index)

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
