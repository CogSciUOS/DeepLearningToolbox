"""A structure representing data (and metadata).
"""

# standard imports
from typing import Any, Iterable
from collections.abc import Sized

# third party imports
import numpy as np

# toolbox imports
from base import MetaRegister


class Data:
    # pylint: disable=no-member
    # _attributes, data
    """A piece of data, either single datum or a batch of data.

    Labels
    ------
    * scheme: e.g., ClassLabel (like, MNISTClasses or ImagnetClasses)
      of RegionLabel (like BoundingBox, FacialLandmark168)
    * single or multi:
    """
    TYPE_IMAGE = 1
    TYPE_FACE = 2 | TYPE_IMAGE

    def __init__(self, datasource=None, data=None, batch: int = None) -> None:
        super().__setattr__('_attributes', {})
        if batch is not None:
            super().__setattr__('_batch', batch)
        if datasource is not None:
            self.add_attribute('datasource', datasource)
        self.add_attribute('data', value=data, batch=True)
        self.add_attribute('type')
        # FIXME[hack]: should be removed as not all data has to be labeled
        # - but some parts of the program expect a label, e.g,
        # self.label = "no label!"
        #  File ".../qtgui/widgets/inputselector.py", line 225,
        #                                           in datasource_changed
        #    self._showInfo(data=data.data, label=data.label)

    def __bool__(self) -> bool:
        """Check if data has been assigned to this :py:class:`Data`
        structure.
        """
        return hasattr(self, 'data') and self.data is not None

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
                result += f"shape={self.data.shape}, dtype={self.data.dtype}"
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
        if attr not in self._attributes:
            raise AttributeError(f"Data has no attribute '{attr}'.")
        if self._attributes[attr]:  # batch attribute
            if not hasattr(val, '__len__'):
                raise ValueError("Cannot assign single value "
                                 f"to batch attribute '{attr}'")
            if len(val) != len(self):
                raise ValueError("Cannot assign value of length {len(val)}"
                                 f"to batch attribute '{attr}' "
                                 f"of length {len(self)}")
        super().__setattr__(attr, val)

    @property
    def is_batch(self) -> bool:
        """Check if this :py:class:`Data` represents a batch of data.
        """
        return hasattr(self, '_batch')

    @property
    def is_image(self) -> bool:
        """Check if this :py:class:`Data` represents an image.
        """
        return getattr(self, 'type', 0) == self.TYPE_IMAGE

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


class BatchDataItem:
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
        return hasattr(self._data, 'data') and self.data is not None

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


class ClassScheme(metaclass=MetaRegister):
    """A :py:class:`ClassScheme` represents a classification scheme.  This
    is essentially a collection of classes.  An actual class is
    referred to by a :py:class:`ClassIdentifier`.

    In addition to the collection of classes, a
    :py:class:`ClassScheme` can also provide one or multiple lookup
    tables that allow to map numerical class indices or textual class
    labels to classes and vice versa. There will always be a default
    table to be used, but all lookup operation will provide an
    optional argument to specify an alternative lookup table to be
    used. Lookup tables can be added to a :py:class:`ClassScheme`
    using the method :py:meth:`add_labels`.


    Attributes
    ----------

    _length: int
        The number of classes in this :py:class:`ClassIdentifier`.

    _lookup: Dict[str, Any]
        Lookup tables mapping labels to classes. Keys are names
        of lookup tables and values the actual tables. Each lookup
        table is either an array (in case of numerical labels)
        or `dict` in case that labels are strings.

    _labels: Dict[str, Any]
        Lookup tables for mapping classes to labels.
        Keys are the name of the table (the same as in _lookup) and
        values are mappings from classes to class labels.

    _no_class: ClassIdentifier
        The class representing the invalid class (no class)
        FIXME[todo]: currently not used!
    """

    def __init__(self, length: int = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._length = length
        self._labels = {}
        self._lookup = {}
        self._no_class = None

    def __len__(self):
        return self._length

    def identifier(self, index: Any, lookup: str = None) -> 'ClassIdentifier':
        """Get an :py:class:`ClassIdentifier` for the given index.

        Arguments
        ---------
        index:
            An index from which the :py:class:`ClassIdentifier` is
            constructed.
        lookup: str
            The name of a lookup table to get the canonical value
            for the given index.
        """
        return ClassIdentifier(self.get_label(index, lookup=lookup), self)

    def get_label(self, index: Any, name: str = 'default',
                  lookup: str = None) -> Any:
        """Look up a label from this :py:class:`ClassScheme`.
        """
        # FIXME[hack]:
        # 1) allow for more iterables than just lists
        #    (e.g., tuples, ndarrays)
        # 2) do it more efficently
        if isinstance(index, list):
            if lookup is not None:
                index = [self._lookup[lookup][i] for i in index]
            if name == 'default':
                return index
            return [self._labels[name][i] for i in index]
        elif isinstance(index, tuple):
            if lookup is not None:
                index = tuple(self._lookup[lookup][i] for i in index)
            if name == 'default':
                return index
            return tuple(self._labels[name][i] for i in index)

        if lookup is not None:
            index = self._lookup[lookup][index]
        return index if name == 'default' else self._labels[name][index]

    def has_label(self, name: str = 'default') -> bool:
        """Check if this :py:class:`ClassScheme` supports the given
        labeling format.

        Arguments
        ---------
        name: str
            The name of the labeling format.
        """
        return name == 'default' or name in self._labels

    def add_labels(self, values: Iterable, name: str = 'default',
                   lookup: bool = False) -> None:
        """Add a label set to this :py:class:`ClassScheme`.

        Arguments
        ---------
        values:
            Some iterable providing the labels in the order of this
            :py:class:`ClassScheme`.
        name: str
            The name of the label set. This name has to be used when
            looking up labels in this label set.
        lookup: bool
            A flag indicating if a reverse lookup table shall be created
            for this label.
        """
        if isinstance(values, np.ndarray):
            pass  # take the given numpy array
        else:
            values = list(values)
        if self._length is None:
            self._length = len(values)
        elif self._length != len(values):
            raise ValueError("Wrong number of class labels: "
                             f"expected {self._length}, got {len(values)}")

        self._labels[name] = values
        if lookup:
            if isinstance(values, np.ndarray) and values.max() < 2*len(self):
                self._lookup[name] = np.zeros(values.max()+1, dtype=np.int)
                self._lookup[name][values] = np.arange(0, values.size)
            else:
                self._lookup[name] = \
                    {val: idx for idx, val in enumerate(values)}


ClassScheme.register_key('ImageNet', 'datasource.imagenet',
                         'ImagenetScheme')
ClassScheme.register_key('WiderFace', 'datasource.widerface',
                         'WiderfaceScheme')


class ClassIdentifier(int):
    # pylint: disable=no-member
    # _scheme
    """An identifier for a class in a classification scheme.
    A class identifier is usually refered to as "class label".

    Arguments
    ---------
    value: Any
        A value identifying the class. This may be the numerical
        class index, or any label which can be mapped to such
        a value by a reverse lookup table of the
        :py:class:`ClassScheme`.
    scheme: ClassScheme
        The classification scheme according to which this
        identifier is used.

    Raises
    ------
    ValueError:
        If the value is not a valid label in the
        :py:class:`ClassScheme`.
    """

    def __new__(cls, value: Any, scheme: ClassScheme = None,
                lookup: str = None) -> None:
        """Create a new class number.

        """
        if lookup is not None:
            if scheme is None:
                raise ValueError(f"No scheme provided for lookup of {value}")
            self = scheme.identifier(value, lookup=lookup)
        else:
            self = super().__new__(cls, value)
            self._scheme = scheme
        return self

    def label(self, name: str = None) -> Any:
        """Get the label for this class number.

        Arguments
        ---------
        name: str
            The name of the labeling format.

        Raises
        ------
        KeyError:
            The given name is not a valid labeling for the ClassScheme.
        """
        return self._scheme.get_label(self, name)

    def has_label(self, name: str = None) -> bool:
        """Check if the :py:class:`ClassIdentifier` has a
        label of the given labeling format.

        Arguments
        ---------
        name: str
            The name of the labeling format.
        """
        return self._scheme and self._scheme.has_label(name)
