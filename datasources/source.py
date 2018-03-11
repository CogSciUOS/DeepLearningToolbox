from collections import namedtuple

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

    def __init__(self, description=None):
        '''Create a new DataSource

        Parameters
        ----------
        description :   str
                        Description of the dataset
        '''
        self._description = description

    def __getitem__(self, index: int):
        '''Provide access to the records in this data source.'''
        pass

    def __len__(self):
        '''Get the number of entries in this data source.'''
        pass

    def getDescription(self) -> str:
        '''Get the description for this DataSource'''
        return self._description
