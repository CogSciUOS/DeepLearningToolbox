import numpy as np
from datasources import DataSource, InputData


class DataArray(DataSource):
    """A ``DataArray`` stores all entries in an array (like the MNIST
    character data). That means that all entries will have the same sizes.

    Attributes
    ----------
    _array  :   np.ndarray
                An array of input data. Can be ``None``.
    """
    _array: np.ndarray = None

    def __init__(self, array: np.ndarray=None, description: str=None):
        """Create a new DataArray

        Parameters
        ----------
        array   :   np.ndarray
                    Numpy data array
        description :   str
                        Description of the data set

        """
        super().__init__(description)
        if array is not None:
            self.setArray(array, description)

    def setArray(self, array, description='array'):
        """Set the array of this DataSource.

        Parameters
        ----------
        array   :   np.ndarray
                    Numpy data array
        description :   str
                        Description of the data set
        """
        self._array = array
        self._description = description

    def __getitem__(self, index: int) -> InputData:
        """

        Result
        ------
        data: np.ndarray
            The input data.
        label:
            The associated label, if known, None otherwise.

        Raises
        ------
        IndexError:
            The index is out of range.
        """
        if self._array is None or index is None:
            return None, None
        data = self._array[index]
        target = None if self._targets is None else self._targets[index]

        return InputData(data, target)

    def __len__(self):
        if self._array is None:
            return 0
        return len(self._array)

    def __str__(self):
        shape = None if self._array is None else self._array.shape
        return f'<DataArray "{shape}">'
