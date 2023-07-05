"""A structure representing data (and metadata).
"""

# standard imports
from typing import Any, Union, Iterable, Optional
from collections.abc import Sized

# third party imports
import numpy as np

# toolbox imports
from . import Observable


# Datalike is intended to be everything that can be used as data.
#
# np.ndarray:
#    The raw data array.
# str:
#    A URL.
Datalike = Union[np.ndarray, str]


class Data(Observable, method='data_changed', changes={'data_changed'}):
    """Abstract base class for :py:class:`Data` objects.
    A :py:class:`Data` object represents a single data point or a
    batch of such point, that can for example be obtained from
    a :py:class:`Datasource`.
    """

    def __new__(cls, **kwargs) -> 'Data':
        new_cls = DataDict if cls is Data else cls
        # error: Argument 2 for "super" not an instance of argument 1
        # probably a bug in mypy? https://github.com/python/mypy/issues/7595
        # type: ignore[misc]
        __new__ = super(Data, new_cls).__new__
        return (__new__(new_cls) if __new__ is object.__new__ else
                __new__(new_cls, **kwargs))

    # def __getattr__(self, attr) -> Any:
    # def __setattr__(self, attribute: str, value: Any) -> None:

    @classmethod
    def as_array(cls, data: Datalike, copy: Optional[bool] = False,
                 dtype: Optional[type] = None) -> np.ndarray:
        """Get a data-like object as nump array.

        Arguments
        ---------
        data: Datalike
            A data-like object to turn into an array.
        copy: bool
            A flag indicating if the data should be copied or
            if the original data is to be returned (if possible).
        dtype:
            Numpy datatype, e.g., numpy.float32.
        """
        if isinstance(data, Data):
            array = data.array
        elif isinstance(data, np.ndarray):
            array = data
        else:
            raise TypeError(f"Cannot convert {type(data)} to array.")

        if dtype is not None and dtype != array.dtype:
            array = array.astype(dtype)
            copy = False

        if copy:
            array = array.copy()

        return data


class DataDict(Data):
    # pylint: disable=no-member
    # _attributes, data
    """A :py:class:`Data` object describes piece of data, either single
    datum or a batch of data.  The central part of a :py:class:`Data`
    object are the actual values, usually in form of a numpy
    array. The values may be present in different versions, e.g.,
    different sizes of an image, normalized versions of the data.
    Such additional representation may be added on demand. The main
    idea is to (1) initially store not more data than necessary, and
    (2) avoid doing same processing steps multiple times, that is
    store transformed version of the data, once they have been
    computed. Each version of the data will be available in

    In addition to the actual data, optional secondary values like
    manually annotations (e.g., class labels for classification tasks,
    bounding boxes for detection task, etc.) and derived values like
    statistics or model predictions can be present or added in form of
    data attributes.  Data attributes are immutable once they have
    been added to the :py:class:`Data` object. That means, that the
    object can grow, but established attributes are not supposed to
    change.

    A :py:class:`Data` is :py:class:`Observable`, allowing a data view
    to be informed once new attributes have been added.

    Batches
    -------

    A :py:class:`Data` object can be created as a batch, meaning that
    it does not represent a single datum, but rather a batch of
    data. The ratio behind this is that many frameworks actually
    expect and operate on batched data.

    The :py:class:`Data` interface provides means to switch between single
    value and batched values: given a :py:class:`Data` representing a
    single datum, one can create a batch of size 1 using the `[]` operator.
    Given a batch :py:class:`Data` one can obtain a view on a single
    value from that batch by providing a numeric index.

    A data attribute can be qualified as batch attribute (meaning that
    there are different attribute values for each element of the
    batch, e.g., the size for a batch of images with different
    individual sizes) or as global attribute (being just one value for
    the complete batch, e.g., if all images in a batch of images have
    the same size). Whether a data attribute is batch or global has to
    be specified when the attribute is added to the data object.


    Attributes
    ----------
    _batch: int

    FIXME[old]:
    Labels
    ------
    * scheme: e.g., ClassLabel (like, MNISTClasses or ImagnetClasses)
      of RegionLabel (like BoundingBox, FacialLandmark168)
    * single or multi:

    """

    def __init__(self, array: np.ndarray = None,
                 datasource=None, batch: int = None, **kwargs) -> None:
        """
        Arguments
        ---------

        """
        super().__init__(**kwargs)
        super().__setattr__('_attributes', {})
        if batch is True:
            if array is None:
                raise ValueError("If no data is given, argument `batch` "
                                 "should specify the batch size.")
            batch = len(array)
        if batch is not None and batch is not False:
            super().__setattr__('_batch', batch)
        if datasource is not None:
            self.add_attribute('datasource', datasource)
        self.add_attribute('array', value=array, batch=True)
        self.add_attribute('type')
        if array is not None:
            self.add_attribute('shape', array.shape)

    def __bool__(self) -> bool:
        """Check if data has been assigned to this :py:class:`Data`
        structure.
        """
        return hasattr(self, 'array') and self.array is not None

    def __str__(self) -> str:
        """Provide an informal string representation of this
        :py:class:`Data` item.
        """
        # pylint: disable=no-member
        result = f"Batch({len(self)})" if self.is_batch else "Data"
        if hasattr(self, 'datasource'):
            result += f"[{self.datasource}]"
        if not self.is_batch:
            if not self:
                result += "[no data]"
            else:
                result += f"shape={self.array.shape}, dtype={self.array.dtype}"
        result += f" with {len(self._attributes)} attributes"
        result += ": " + ", ".join(self._attributes.keys())
        return result

    def __len__(self) -> int:
        if not self.is_batch:
            raise TypeError("Data has no length (not a batch)")
        return self._batch

    def __getitem__(self, index: int):
        if not self.is_batch:
            raise TypeError("Data object is not subscriptable (not a batch)")
        length = len(self)
        if not -length <= index < length:
            raise IndexError("batch index out of range")
        return BatchDataItem(self, index if index >= 0 else index + length)

    def __iter__(self):
        """Iterate over all batch items.
        """
        if not self.is_batch:
            raise TypeError("Data object is not iterable (not a batch)")
        for i in range(len(self)):
            yield self[i]

    def __setattr__(self, attr: str, val: Any) -> None:
        """
        """
        if attr[0] == '_':
            super().__setattr__(attr, val)
            return

        if attr not in self._attributes:
            raise AttributeError(f"Data has no attribute '{attr}'.")
        if self._attributes[attr]:  # batch attribute
            if not isinstance(val, Sized):
                raise ValueError("Cannot assign non-Sized value "
                                 f"of type {type(val)} "
                                 f"to batch attribute '{attr}'")
            if len(val) != len(self):
                raise ValueError("Cannot assign value of length {len(val)}"
                                 f"to batch attribute '{attr}' "
                                 f"of length {len(self)}")
        super().__setattr__(attr, val)
        self.notify_observers(self.Change('data_changed'), attribute=attr)

    @property
    def is_batch(self) -> bool:
        """Check if this :py:class:`Data` represents a batch of data.
        """
        return hasattr(self, '_batch')

    def has_attribute(self, name: str) -> bool:
        """Check if this :py:class:`Data` has the given attribute.
        """
        return name in self._attributes

    def is_batch_attribute(self, name: str) -> bool:
        """Check if the given attribute is a batch attribute.
        """
        return self._attributes.get(name, False)

    def add_attribute(self, name: str, value: Any = None, batch: bool = False,
                      initialize: bool = False) -> None:
        """Add an attribute to this :py:class:`Data`. Only attributes
        added by this method can be set.

        Parameters
        ----------
        name: str
            Name of the new attribute.
        initialize:
            A flag indicating if the attribute should be initialized.
        value: Any (optional)
            Value for the new attribute. If not None, this implies
            initialize=True.
        batch: bool
            A flag indicating if the attribute is a batch attribute
            (`True` = different value for each batch item) or a global
            attribute (`False` = each batch item has the same value).
        """
        batch = batch and self.is_batch
        self._attributes[name] = batch

        if initialize or value is not None or not batch:
            if batch and (not isinstance(value, Sized) or
                          len(value) != len(self)):
                # batch attribute but non-batch value
                value = [value] * len(self)
            setattr(self, name, value)

    def attributes(self, batch: bool = None) -> Iterable[str]:
        """A view on the attributes of this :py:class:`Data`.

        Arguments
        ---------
        batch: bool
            A flag indicating what kind of attributes should be iterated:
            True = batch attribute, False = non-batch attribute,
            None = both types of attributes (default).
        """
        for name, is_batch in self._attributes.items():
            if batch is None or batch is is_batch:
                yield name

    def initialize_attributes(self, attributes: Iterable[str] = None,
                              batch: bool = None) -> None:
        """Initialize attributes of this :py:class:`Data`.

        Arguments
        ---------
        batch: bool
            A flag indicating what kind of attributes should be initialized:
            True = batch attribute, False = non-batch attribute,
            None = both types of attributes (default).
        """
        if attributes is None:
            attributes = self.attributes(batch=batch)
        for name in attributes:
            if not hasattr(self, name):
                if self.is_batch_attribute(name):
                    setattr(self, name, [None] * len(self))
                else:
                    setattr(self, name, None)

    def debug(self) -> None:
        """Debugging a data object. Output types of all attributes.
        """
        print(f"Data object ({id(self)})")
        for name, is_batch in self._attributes.items():
            value = getattr(self, name, None)
            if value is None:
                info = 'None' if hasattr(self, name) else '*missing*'
            elif isinstance(value, np.ndarray):
                info = f"ndarray{value.shape} [{value.dtype}]"
            elif isinstance(value, int):
                info = f"{value} (int)"
            elif isinstance(value, str):
                info = f"'{value}' (str)"
            elif isinstance(value, tuple):
                info = f"{value} (tuple)"
            else:
                info = f"{type(value)}"
            print(f" - {name}[{'batch' if is_batch else 'global'}]: {info}")


class BatchDataItem(Data):
    """A single data item in a batch of data.
    """
    is_batch: bool = False

    def __init__(self, data: Data, index: int) -> None:
        super().__init__()
        self._data = data
        self._index = index

    def __str__(self):
        """Provide an informal string representation of this
        :py:class:`BatchDataItem` item.
        """
        return f"Batch item {self._index} of {self._data}"

    def __bool__(self):
        """Check if data has been assigned to this :py:class:`Data`
        structure.
        """
        return hasattr(self._data, 'array') and self.array is not None

    def __getattr__(self, attr) -> Any:
        if attr.startswith('_'):
            raise AttributeError(f"BatchDataItem has no attribute '{attr}'")
        if self._data.is_batch_attribute(attr):
            return getattr(self._data, attr)[self._index]
        return getattr(self._data, attr)

    def __setattr__(self, attr, val) -> None:
        if attr.startswith('_'):
            super().__setattr__(attr, val)
        elif self._data.is_batch_attribute(attr):
            getattr(self._data, attr)[self._index] = val
        else:
            setattr(self._data, attr, val)

    def add_attribute(self, name: str, value: Any = None, batch: bool = True,
                      initialize: bool = True) -> None:
        """Add an attribute to this :py:class:`Data`. Only attributes
        added by this method can be set.

        Parameters
        ----------
        name: str
            Name of the new attribute.
        initialize:
            A flag indicating if the attribute should be initialized.
        value: Any (optional)
            Value for the new attribute. If not None, this implies
            initialize=True.
        batch: bool
            A flag indicating if the attribute is a batch attribute
            (`True` = different value for each batch item) or a global
            attribute (`False` = each batch item has the same value).
        """
        if not batch:
            raise ValueError(f"Cannot add non-batch attribute '{name}' "
                             "to a BatchDataItem.")
        if not initialize:
            raise ValueError("You have to initialize the whole batch when "
                             "adding the the batch attribute '{name}' "
                             "via a BatchDataItem.")
        self._data.add_attribute(name, value=value,
                                 batch=True, initialize=True)


class BatchWrapper(Data):
    """A wrapper letting a single (non-batch) :py:class:`Data` object
    looking like a batch.
    """
    is_batch: bool = True

    def __init__(self, data: Data) -> None:
        super().__init__()
        self._data = data

    def __getattr__(self, attr) -> Any:
        if attr.startswith('_'):
            raise AttributeError(f"BatchWrapper has no attribute '{attr}'")
        if self._data.is_batch_attribute(attr):
            value = getattr(self._data, attr)
            return (value[np.newaxis] if isinstance(np.ndarray, value)
                    else [value])
        return getattr(self._data, attr)

    def __setattr__(self, attribute: str, value: Any) -> None:
        if attribute.startswith('_'):
            super().__setattr__(attribute, value)
        elif self._data.is_batch_attribute(attribute):
            if len(value) != 1:
                raise AttributeError("BatchWrapper only accepts batches "
                                     f"of size 1 for attribute '{attribute}'")
            setattr(self._data, attribute, value[1])
        else:
            setattr(self._data, attribute, value)


#
# Utility functions
#

def add_noise(data: np.ndarray) -> np.ndarray:
    """Add (Gaussian) noise to a given data.
    """
    image = data.array
    noise = np.random.normal(0, 0.3, size=np.shape(image))
    data.add_attribute('noisy', image + noise, batch=True)
