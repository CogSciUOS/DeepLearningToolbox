from . import Datasource, Labeled, Indexed, InputData, Metadata

from random import randint
import numpy as np


class DataArray(Indexed):
    """A ``DataArray`` stores all entries in an array (like the MNIST
    character data). That means that all entries will have the same sizes.

    Attributes
    ----------
    _array  :   np.ndarray
                An array of input data. Can be ``None``.
    """
    _array: np.ndarray = None
    _index: int = None
    _batch_indices: np.ndarray = None

    def __init__(self, array: np.ndarray=None, **kwargs):
        """Create a new DataArray

        Parameters
        ----------
        array   :   np.ndarray
                    Numpy data array
        description :   str
                        Description of the data set

        """
        super().__init__(**kwargs)
        if array is not None:
            self.set_data_array(array, description)

    @property
    def prepared(self) -> bool:
        """A :py:class:`DataArray` is prepared once the array
        has been initialized.
        """
        return self._array is not None

    def _prepare_data(self) -> None:
        super().prepare()
        self._index = 0
        self._batch_indices = np.arange(1)

    def _unprepare(self) -> None:
        """A :py:class:`DataArray` is reset in an unprepared state
        by releasing the array.
        """
        self._array = None

    def set_data_array(self, array, description='array'):
        """Set the array of this DataSource.

        Parameters
        ----------
        array: np.ndarray
            Numpy data array
        description: str
            Description of the data set
        """
        self._array = array
        self._description = description
        self.change('data_changed')

    @property
    def index(self) -> int:
        return self._index

    # FIXME[old]!
    def fetch_batch(self, batch_size: int=1) -> None:
        self._batch_indices = \
            np.arange(self._index + 1, self._index + 1 + batch_size) % len(self)
        self._index = (self._index + batch_size) % len(self)
        self.change('batch_changed')

    def _fetch_index(self, index: int=None,
                     metadata: Metadata=None) -> None:
        if index is not None:
            self._index = index 
        elif self._index is None:
            self._index = 0
        if metadata is not None:
            metadata.set_attribute('index', self._index)
        # print(f"Array:fetch({self._index})")

    @property
    def fetched(self):
        return self._index is not None

    def _get_data(self):
        return self._array[self._index]

    @property
    def batch(self):
        return self._array[self._batch_indices]

    def __len__(self):
        if self._array is None:
            return 0
        return len(self._array)

    def __getitem__(self, index: int) -> InputData:
        """

        Result
        ------
        data: np.ndarray
            The current data point.

        Raises
        ------
        IndexError:
            The index is out of range.
        """
        return (None if self._array is None or index is None
                else self._array[index])

    def __str__(self):
        shape = None if self._array is None else self._array.shape
        return f'<DataArray "{shape}">'

    def get_description(self, index: int=None, **kwargs) -> str:
        """Provide a description of the Datasource or one of its
        elements.

        Attributes
        ----------
        index: int
            Provide a description of the datapoint for that index.
        """
        description = super().get_description(**kwargs)
        if index is not None:
            description = 'Image ' + str(index) + ' from ' + description
        return description


class LabeledArray(DataArray, Labeled):
    """An array with labels for its entries.

    Attributes
    ----------
    _labels: np.ndarray
        An array mapping indices (of the data array) to (numeric) labels.
    """

    _labels: np.ndarray = None
    _number_of_labels: int = None

    @property
    def number_of_labels(self) -> int:
        """The number of different labels for this dataset.
        """
        return self._number_of_labels

    @property
    def labels_prepared(self) -> bool:
        """Check if labels for this dataset have been prepared.
        """
        return self._labels is not None


    def _prepare_labels(self, labels: np.ndarray=None) -> None:
        """Set the labels for for this labeled Array datasource.

        Arguments
        ---------
        labels: np.ndarray
            An array containing the labels for this datasource. Should
            have the same length as the data array.
        """
        if labels is None:
            raise ValueError("You have to provide a labels array when "
                             "preparing a LabeledArray.")
        if len(self) != len(labels):
            raise ValueError("Wrong number of target values: "
                             f"expect={len(self)}, got={len(labels)}")
        self._labels = labels
        self._number_of_labels = 1 + labels.max()

    def _get_label(self) -> int:
        """Get the (numeric) label for the current data point.
        This presupposes, that a current data point has been selected
        before by calling :py:meth:`fetch()`
        """
        return self._labels[self._index]

    def _get_batch_labels(self) -> int:
        """Get the (numeric) label for the current data point.
        This presupposes, that a current data point has been selected
        before by calling :py:meth:`fetch()`
        """
        return self._labels[self._batch_indices]

    def __getitem__(self, index: int) -> InputData:
        """

        Result
        ------
        data: np.ndarray
            The input data.
        label: int
            The associated label, if known, None otherwise.

        Raises
        ------
        IndexError:
            The index is out of range.
        """
        data = super().__getitem__(index)
        label = self._labels[index] if self.labels_prepared else None
        return InputData(data, label)

    def get_description(self, index: int=None, short: bool=False,
                        with_label: bool=False, **kwargs) -> str:
        """Provide a description of the Datasource or one of its
        elements.

        Attributes
        ----------
        index: int
            In case of an indexed Datasource, provide a description
            of the given element.
        """
        if index and with_label:
            description = super().get_description(index=index, **kwargs)
            if self.labels_prepared:
                description += " with label "
                label = self._labels[index]
                if self.has_text_for_labels():
                    text = self.get_text_for_label(label)
                    description += f"'{text}' "
                description += f"({label})"
            else:
                description += ", no label available"
        else:
            description = super().get_description(with_label=with_label,
                                                  **kwargs)
        return description
