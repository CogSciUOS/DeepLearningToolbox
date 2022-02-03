"""A :py:class:`Datasource` providing data from an array.
"""

# FIXME[todo]: add the option to work with numpy memmap
#  np.memmap(filename, dtype='float32', mode='w+',
#            shape=(samples,) + network[layer].output_shape[1:])
#
# Notice: the filen referred to by filename has to by an
# (uncompressed) `.npy` file. There is not way to use compressed `.npz`
# files.


# third party imports
from typing import Union
import numpy as np

# toolbox imports
from ..base.data import Data
from ..tool.classifier import ClassScheme, ClassIdentifier
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
        # pylint: disable=arguments-differ
        """Get data from this :py:class:`Datasource`\\ .
        """
        data.add_attribute('shape', value=self._array.shape[1:])
        super()._get_meta(data, **kwargs)

    def _get_batch(self, data: Data, index: int = None, **kwargs) -> None:
        # pylint: disable=arguments-differ
        if index is not None:
            if isinstance(index, np.ndarray):
                data.array = self._array[index]
            else:
                data.array = self._array[index:index+len(data)]
        super()._get_batch(data, index=index, **kwargs)

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
    _one_hot: bool = False
    _scheme: ClassScheme = None

    def __init__(self, labels: np.ndarray = None, one_hot: bool = False,
                 scheme: ClassScheme = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._labels = labels
        self._one_hot = one_hot
        self._scheme = scheme

    @property
    def labels_prepared(self) -> bool:
        """Check if labels for this dataset have been prepared.
        """
        return self._labels is not None

    @property
    def label_scheme(self) -> ClassScheme:
        """The :py:class:`ClassScheme` in case the `Datasource` is a
        classification dataset.
        """
        return self._scheme

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

    def _set_labels(self, labels: np.ndarray,
                    scheme: ClassScheme = None) -> None:
        """Set the labels for this :py:class:`LabeledArray`. The labels
        will be stored as numpy array, allowing for some of the
        convenient numpy indexing techniques.

        Arguments
        ---------
        labels:
            The labels for this :py:class:`LabeledArray`. Depending
            on the type of datasource, this could be (numeric) class
            indices, but could also be other kind of labels, like
            list of bounding boxes. The crucial point is that there
            should be one label for each data point in the datasource,
            and the order of labels has to correspond to the order
            of datapoints.
        scheme:
            If not `None`, `labels` are considered as class labels,
            that is identifiers of classes in that :py:class:`ClassScheme`.
            Such labels will be stored as an numpy array of
            :py:class:`ClassIdentifier`.
        """
        if scheme is None:
            self._labels = labels
        else:
            self._scheme = scheme
            self._labels = np.empty(len(self._lfw_people.target), dtype=object)
            for index, label in enumerate(labels):
                self._labels[index] = \
                    ClassIdentifier(label, scheme=scheme)

    def _get_meta(self, data: Data, **kwargs) -> None:
        """Get data from this :py:class:`Datasource`\\ .
        """
        data.add_attribute('label', batch=True)
        super()._get_meta(data, **kwargs)

    def _get_batch(self, data: Data, **kwargs) -> None:
        """Get data from this :py:class:`Datasource`\\ .
        """
        super()._get_batch(data, **kwargs)
        if self.labels_prepared:
            label = self._labels[data.index]
            if self._one_hot and self._scheme is not None:
                data.label = self.label_to_one_hot(label)
            else:
                data.label = label

    def _get_index(self, data, index: int, **kwargs) -> None:
        """Get a single data element for a given index.

        Raises
        ------
        IndexError:
            The index is out of range.
        """
        super()._get_index(data, index, **kwargs)
        label = self._labels[index] if self.labels_prepared else None
        if self._one_hot:
            data.label = self.label_to_one_hot(label)
        else:
            data.label = label

    def label_to_one_hot(self, label: Union[int, np.ndarray]) -> np.ndarray:
        """Get a one-hot representation for a numerical class label.
        """
        if isinstance(label, int):
            one_hot = np.zeros(len(self._scheme))
            one_hot[label] = 1
            return one_hot

        return np.eye(len(self._scheme))[label]

    def one_hot_to_label(self, one_hot: np.ndarray) -> Union[int, np.ndarray]:
        """Obtain numerical label(s) for a (single or batch of)
        one-hot vector(s).
        """
        if one_hot.ndims == 1:
            return np.argmax(one_hot)

        # batch of one-hot vectors, shape (batch, n_labels)
        return np.argmax(one_hot, axis=1)

    def _get_description(self, index: int = None, short: bool = False,
                         with_label: bool = False, **kwargs) -> str:
        # pylint: disable=arguments-differ
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
            if not short:
                if self.labels_prepared:
                    description += " with label {self._labels[index]}"
                else:
                    description += ", no label available"
        else:
            description = super()._get_description(with_label=with_label,
                                                   **kwargs)
        return description
