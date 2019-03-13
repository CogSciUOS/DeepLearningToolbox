import numpy as np
from datasources import Datasource, InputData


class DataArray(Datasource):
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

    @property
    def prepared(self) -> bool:
        """A :py:class:`DataArray` is prepared once the array
        has been initialized.
        """
        return self._array is not None

    def prepare(self) -> None:
        if not self.prepared:
            raise NotImplementedError("prepare() should be implemented by "
                                      "subclasses of DataArray.")

    def unprepare(self) -> None:
        """A :py:class:`DataArray` is reset in an unprepared state
        by releasing the array.
        """
        self._array = None
        self.change('state_changed')


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
