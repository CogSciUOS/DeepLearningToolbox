from collections import namedtuple
import numpy as np

InputData = namedtuple('Data', ['data', 'name'])

from base import BusyObservable

class Datasource(BusyObservable, method='datasource_changed',
                 changes=['state_changed', 'data_changed',
                          'index_changed']):
    """.. :py:class:: Datasource

    An abstract base class for different types of data sources.

    There are different APIs for navigating in a datasource, depending
    on what that datasource supports:

    array-like navigation: individual elements of a data source can be
    accessed using an array-like notation.

    random selection: select a random element from the dataset.

    Attributes
    ----------
    _description : str
        Short description of the dataset

    _targets: np.ndarray
        T
    _labels: list


    Changes
    -------
    state_changed:
        The state of the DataSource has changed (e.g. data were downloaded,
        loaded/unloaded, etc.)
    data_changed:
        The data have changed (e.g. new data from the camera).
    index_changed:
        For indexed datasources only: the index (current item) has changed.
    """

    def __init__(self, description=None):
        """Create a new Datasource.

        Parameters
        ----------
        description :   str
                        Description of the dataset
        """
        super().__init__()
        self._description = (self.__class__.__name__ if description is None
                             else description)
        self._targets = None
        self._labels = None

    def __getitem__(self, index: int):
        """Provide access to the records in this data source."""
        pass

    def __len__(self):
        """Get the number of entries in this data source."""
        pass

    def add_target_values(self, target_values: np.ndarray):
        if len(self) != len(target_values):
            raise ValueError('Wrong number of target values. expect={}, got={}'
                             .format(len(self), len(target_values)))
        self._targets = target_values

    def add_target_labels(self, labels: list):
        """Set labels (string representations) for given target values.

        Parameters
        ----------
        labels:
            A list of labels (string representations) for the target
            values of this data set.
        """
        self._labels = labels

    def get_label(self, target: int) -> str:
        """Get a label (string representation) for a given target value.

        Parameters
        ----------
        target: int
            The target value for which the label is requested.

        Returns
        -------
        label: str
            The label for the target. If no label information are
            available, this is simply be a string representation
            of the target value.
        """
        return str(target) if self._labels is None else self._labels[target]

    def getName(self, index=None) -> str:
        if index is None:
            return self._description
        elif self._targets is None:
            return self._description + "[" + str(index) + "]"
        else:
            target = int(self._targets[index])
            target = (f'[{target}]' if self._labels is None
                      else self._labels[target])
            return self._description + ", target=" + target

    def getDescription(self) -> str:
        """Get the description for this Datasource"""
        return self._description

    def get_description(self, index: int=None) -> str:
        return 'Image ' + str(index) + ' from ' + self._description

    def prepare(self):
        """Prepare this Datasource for use.
        """
        pass  # to be implemented by subclasses

    def get_section_ids(self):
        """Get a list of sections provided by this data source.  A data source
        may for example be divided into training and test data.
        """
        return None

    def get_section(self):
        """Get the current section in the data source.
        """
        return None

    def set_section(self, section_id):
        """Set the current section in the data source.
        """
        pass


class Predefined:
    """An abstract base class for predefined data sources.
    """

    _id = None

    def __init__(self, id):
        self._id = id
        Predefined.datasources[id] = self

    def get_public_id(self):
        """Get the "public" ID that is used to identify this datasource.  Only
        predefined Datasource should have such an ID, other
        datasources should provide None.
        """
        return self._id

    def check_availability(self):
        """Check if this Datasource is available.

        Returns
        -------
        True if the Datasource can be instantiated, False otherwise.
        """
        return False

    def download(self):
        raise NotImplementedError("Downloading this datasource is "
                                  "not implemented yet.")

    #
    # Static data and methods
    #

    datasources = {}

    @staticmethod
    def get_data_source_ids():
        return list(Predefined.datasources.keys())

    @staticmethod
    def get_data_source(id):
        return Predefined.datasources[id]
