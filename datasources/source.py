from collections import namedtuple
import numpy as np

InputData = namedtuple('Data', ['data', 'name'])

class DataSource:
    '''An abstract base class for different types of data sources.  The
    individual elements of a data source can be accessed using an array-like
    notation.

    Attributes
    ----------
    _description    :   str
                        Short description of the dataset

    '''
    _description: str = None
    _targets: np.ndarray = None
    _labels: list = None

    @staticmethod
    def check_availability():
        '''Check if this Datasource is available.
        
        Returns
        -------
        True if the DataSource can be instantiated, False otherwise.
        '''
        return True

    
    def __init__(self, description=None):
        '''Create a new DataSource

        Parameters
        ----------
        description :   str
                        Description of the dataset
        '''
        self._description = self.__class__.__name__ if description is None else description 

    def __getitem__(self, index: int):
        '''Provide access to the records in this data source.'''
        pass

    def __len__(self):
        '''Get the number of entries in this data source.'''
        pass

    def add_target_values(self, target_values: np.ndarray):
        if len(self) != len(target_values):
            raise ValueError('Wrong number of target values. expect={}, got={}'.format(len(self), len(target_values)))
        self._targets = target_values
        
    def add_target_labels(self, labels: list):
        self._labels = labels
        
    def getName(self, index=None) -> str:
        if index is None:
            return self._description
        elif self._targets is None:
            return self._description + "[" + str(index) + "]"
        else:
            target = int(self._targets[index])
            target = f'[{target}]' if self._labels is None else self._labels[target]
            return self._description + ", target=" + target

    def getDescription(self) -> str:
        '''Get the description for this DataSource'''
        return self._description


    @staticmethod
    def get_public_id():
        '''Get the "public" ID that is used to identify this datasource.  Only
        predefined DataSource should have such an ID, other
        datasources should provide None.

        Actually, it is the result of this method that determines if a
        DataSource is considered as predefined. If you return a value
        different from None here, you should make sure that you
        provide functionality to download and initialize this
        DataSource.

        '''
        return None

    @staticmethod
    def download():
        raise NotImplementedError("Downloading ImageNet is not implemented yet")
