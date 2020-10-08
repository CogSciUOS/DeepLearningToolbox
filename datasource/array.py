"""A :py:class:`Datasource` providing data from an array.
"""

# third party imports
import numpy as np

# toolbox imports
from dltb.base.data import Data
from .datasource import Labeled, Indexed


class DataArray(Indexed):
    # pylint: disable=too-many-ancestors
    """A ``DataArray`` stores all entries in an array (like the MNIST
    character data). That means that all entries will have the same sizes.

    Attributes
    ----------
    _array: np.ndarray
        An array of input data. Can be ``None``.
    """

    def __init__(self, array: np.ndarray = None, description: str = None,
                 **kwargs):
        """Create a new DataArray

        Parameters
        ----------
        array: np.ndarray
            Numpy data array
        description: str
            Description of the data set

        """
        super().__init__(**kwargs)
        self._array = array
        self._description = description

    def __len__(self):
        return 0 if self._array is None else len(self._array)

    def __str__(self):
        shape = None if self._array is None else self._array.shape
        return f'<DataArray "{shape}">'

    def _get_description(self, index: int = None, **kwargs) -> str:
        """Provide a description of the Datasource or one of its
        elements.

        Attributes
        ----------
        index: int
            Provide a description of the datapoint for that index.
        """
        description = super()._get_description(**kwargs)
        if index is not None:
            description = 'Image ' + str(index) + ' from ' + description
        return description

    #
    # Preparation
    #

    def _prepared(self) -> bool:
        """A :py:class:`DataArray` is prepared once the array
        has been initialized.
        """
        return super()._prepared() and (self._array is not None)

    def _unprepare(self) -> None:
        """A :py:class:`DataArray` is reset in an unprepared state
        by releasing the array.
        """
        self._array = None
        super()._unprepare()

    #
    # Data
    #

    def _get_meta(self, data: Data, **kwargs) -> None:
        """Get data from this :py:class:`Datasource`\\ .
        """
        data.add_attribute('shape', value=self._array.shape[1:])
        super()._get_meta(data, **kwargs)

    def _get_batch(self, data: Data, index: int = None, **kwargs) -> None:
        if index is not None:
            data.array = self._array[index:index+len(data)]
        super()._get_batch(data, **kwargs)

    def _get_index(self, data: Data, index: int, **kwargs) -> None:
        data.array = self._array[index]
        super()._get_index(data, index, **kwargs)


class LabeledArray(DataArray, Labeled):
    # pylint: disable=too-many-ancestors
    """An array with labels for its entries.

    Attributes
    ----------
    _labels: np.ndarray
        An array mapping indices (of the data array) to (numeric) labels.
    """

    _labels: np.ndarray = None

    @property
    def labels_prepared(self) -> bool:
        """Check if labels for this dataset have been prepared.
        """
        return self._labels is not None

    def _prepare_labels(self, labels: np.ndarray = None) -> None:
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

    def _get_meta(self, data: Data, **kwargs) -> None:
        """Get data from this :py:class:`Datasource`\\ .
        """
        data.add_attribute('label', batch=True)
        super()._get_meta(data, **kwargs)

    def _get_index(self, data, index: int, **kwargs) -> None:
        """

        Raises
        ------
        IndexError:
            The index is out of range.
        """
        super()._get_index(data, index, **kwargs)
        label = self._labels[index] if self.labels_prepared else None
        data.label = label

    def _get_description(self, index: int = None, short: bool = False,
                         with_label: bool = False, **kwargs) -> str:
        """Provide a description of the Datasource or one of its
        elements.

        Attributes
        ----------
        index: int
            In case of an indexed Datasource, provide a description
            of the given element.
        """
        if index and with_label:
            description = super()._get_description(index=index, **kwargs)
            if self.labels_prepared:
                description += " with label {self._labels[index]}"
            else:
                description += ", no label available"
        else:
            description = super()._get_description(with_label=with_label,
                                                   **kwargs)
        return description
