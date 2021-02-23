"""General definitions for classifiers.

"""

# FIXME[concept]: there should be a connection between network
#   and suitable datasets/labels, i.e., AlexNet should provide
#   class labels, even if not applied to ImageNet data

# standard imports
from typing import Union, Tuple, Iterable, Iterator, Any
from abc import abstractmethod, ABC
import logging

# third party imports
import numpy as np

# toolbox imports
from ..base.prepare import Preparable
from ..base.register import RegisterClass
from ..base.data import Datalike
from ..base.image import Imagelike, ImageExtension


# logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


class ClassScheme(metaclass=RegisterClass):
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

    def get_label(self, index: Any, name: str = None,
                  lookup: str = None) -> Any:
        """Look up a label for a given index for this
        :py:class:`ClassScheme`.

        Attributes
        ----------
        index:
            A single class index or a collection of indices (list or tuple).
            The
        name:
            The name of the label format. If `None`, the default
            format (the integer value) is returned.
        
        """
        # FIXME[todo]:
        # 1) allow for more iterables than just lists
        #    (e.g., tuples, ndarrays)
        # 2) do it more efficently
        if isinstance(index, list):
            if lookup is not None:
                index = [self._lookup[lookup][i] for i in index]
            if name is None:
                return index
            return [self._labels[name][i] for i in index]

        if isinstance(index, tuple):
            if lookup is not None:
                index = tuple(self._lookup[lookup][i] for i in index)
            if name is None:
                return index
            return tuple(self._labels[name][i] for i in index)

        if lookup is not None:
            index = self._lookup[lookup][index]
        return index if name is None else self._labels[name][index]

    def has_label(self, name: str) -> bool:
        """Check if this :py:class:`ClassScheme` supports the given
        labeling format.

        Arguments
        ---------
        name: str
            The name of the labeling format.
        """
        return (name is None) or (name in self._labels)

    def add_labels(self, values: Iterable, name: str,
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

    def labels(self) -> Iterator[str]:
        """Enumerate the label names that are currently registered with
        this :py:class:`ClassScheme`.
        """
        return self._labels.keys()


ClassScheme.register_instance('ImageNet',
                              'dltb.thirdparty.datasource.imagenet',
                              'ImagenetScheme')
ClassScheme.register_instance('WiderFace',
                              'dltb.thirdparty.datasource.widerface',
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

    def __getitem__(self, name: str = None) -> Any:
        """Get the label for this class number.

        Arguments
        ---------
        name: str
            The name of the labeling format.

        Raises
        ------
        KeyError:
            The given name is not a valid labeling for the
            :py:class:`ClassScheme`.
        """
        return self._scheme.get_label(self, name)

    def label(self, name: str = None) -> Any:
        # FIXME[old]: should be replaced by the shorter __getitem__()
        """Get the label for this class number.

        Arguments
        ---------
        name: str
            The name of the labeling format.

        Raises
        ------
        KeyError:
            The given name is not a valid labeling for the
            :py:class:`ClassScheme`.
        """
        return self[name]

    @property
    def scheme(self) -> ClassScheme:
        """The :py:class:`ClassScheme` in which this
        :py:class:`ClassIdentifier` identifies a class.
        """
        return self._scheme

    def has_label(self, name: str = None) -> bool:
        """Check if the :py:class:`ClassIdentifier` has a
        label of the given labeling format.

        Arguments
        ---------
        name: str
            The name of the labeling format.
        """
        return self._scheme and self._scheme.has_label(name)


class Classifier(ABC, Preparable):
    """A :py:class:`Classifier` is associated with a classification
    scheme, describing the set of classes ("labels").  Output units of
    the network have to be mapped to the class labels.


    Attributes
    ----------

    _scheme: ClassScheme
        The classification scheme applied by this classifier.

    _lookup: str
        The name of the lookup table used to map :py:class:`Classifier`
        outputs to classes of the :py:class:`ClassScheme`.
        The classification scheme has to support this lookup table.

    """

    def __init__(self, scheme: Union[ClassScheme, str, int],
                 lookup: str = None, **kwargs):
        LOG.debug("Classifier[scheme={%s}]: %s", scheme, kwargs)
        super().__init__(**kwargs)
        if isinstance(scheme, str):
            scheme = ClassScheme[scheme]
        elif isinstance(scheme, int):
            scheme = ClassScheme(10)
        self._scheme = scheme
        self._lookup = lookup

    @property
    def class_scheme(self) -> ClassScheme:
        """The :py:class:`ClassScheme` used by this :py:class:`Classifier`.
        """
        return self._scheme

    @property
    def class_scheme_lookup(self) -> str:
        """The lookup table to be used to map the (internal) results of this
        :py:class:`Classifier` to a class from the
        :py:class:`ClassScheme`.
        """
        return self._lookup

    def _prepare(self) -> None:
        """Prepare this :py:class:`Classifier`.

        Raises
        ------
        ValueError
            If the :py:class:`ClassScheme` does not fit to this
            :py:class:`Classifier`.
        """
        super()._prepare()
        self._scheme.prepare()

    @abstractmethod
    def classify(self, inputs: Datalike) -> ClassIdentifier:
        """Output the top-n classes for given batch of inputs.

        Arguments
        ---------
        inputs: np.ndarray
            A data point or a batch of input data to classify.

        Results
        -------
        classes:
            A list of class-identifiers or a
            list of tuples of class identifiers.
        """


class ImageClassifier(ImageExtension, base=Classifier):
    """
    """

    def classify_image(self, image: Imagelike) -> ClassIdentifier:
        """Classify the given image.
        """
        return self.classify(self.image_to_internal(image))


class SoftClassifier(Classifier):
    """A :py:class:`SoftClassifier` maps an input item to score vector
    describing the confidence with which the item belongs to the
    corresponding class of a classification scheme.

    """

    @abstractmethod
    def class_scores(self, inputs: Datalike) -> np.ndarray:
        """Compute all class scores. The output array will have one entry for
        each class of the classification scheme.

        Arguments
        ---------
        inputs: np.ndarray
            The input data, either a single data point or a batch of data.
        """
        # FIXME[concept]: we need a way to determine if inputs are single or
        # batch!

    def classify(self, inputs: Datalike, top: int = None,
                 confidence: bool = False) -> Union[ClassIdentifier,
                                                    Tuple[ClassIdentifier]]:
        # pylint: disable=arguments-differ
        """Output the top-n classes for given batch of inputs.

        Arguments
        ---------
        inputs: np.ndarray
            A batch of input data to classify.

        Results
        -------
        classes:
            A list of class-identifiers or a
            list of tuples of class identifiers.
        score (optional):
            If top is None, a one-dimensional array of confidence values or
            otherwise a two-dimension array providing the top highest
            confidence values for each input item.
        """
        scores = self.class_scores(inputs)
        top_classes, top_scores = self.top_classes(scores, top=top)
        return (top_classes, top_scores) if confidence else top_classes

    #
    # Utilities
    #

    def top_classes(self, scores: np.ndarray,
                    top: int = None) -> (list, np.ndarray):
        """Get the network's top classification results for the
        current input. The results will be sorted with highest ranked
        class first.

        Parameters
        ----------
        scores: np.ndarray
            The class scores ("probabilities").
        top: int
            The number of results to report.

        Returns
        -------
        classes:
            The class indices.
        scores:
            The corresponding class scores, i.e., the output value
            of the network for that class.
        """

        if scores is None:
            return (None, None)  # no results

        #
        # compute the top n class scores
        #
        unbatch = scores.ndim == 1
        if unbatch:
            scores = scores[np.newaxis]

        # scores has shape (batch, classes)
        batch = np.arange(len(scores))
        if top is None:
            top_indices = np.argmax(scores, axis=-1)
        else:
            # Remark: here we could use np.argsort(-class_scores)[:n]
            # but that may be slow for a large number classes,
            # as it does a full sort. The numpy.partition provides a faster,
            # though somewhat more complicated method.
            top_indices_unsorted = np.argpartition(-scores, top)[batch, :top]
            order = np.argsort((-scores)[batch, top_indices_unsorted.T].T)
            top_indices = top_indices_unsorted[batch, order.T].T

        # FIXME[hack]: best would be, if ClassIdentifier could deal with
        # numpy.int. Until this is possible, we will use lists of int.
        if scores.ndim == 1:  # no batch data
            class_identifiers = \
                (self._scheme.identifier(top_indices, lookup=self._lookup)
                 if top is None else
                 [self._scheme.identifier(cls, lookup=self._lookup)
                  for cls in top_indices])
        else:
            class_identifiers = []
            for indices in top_indices:
                if top is None:
                    identifiers = \
                        self._scheme.identifier(indices, lookup=self._lookup)
                else:
                    identifiers = \
                        [self._scheme.identifier(cls, lookup=self._lookup)
                         for cls in indices]
                class_identifiers.append(identifiers)

        scores = scores[batch, top_indices.T].T
        if unbatch:
            class_identifiers = class_identifiers[0]
            scores = scores[0]

        return class_identifiers, scores

    def class_rank(self, scores: np.ndarray,
                   label: ClassIdentifier) -> (int, float):
        """Check the rank (position) of a class identifier
        in the given score table.

        Arguments
        ---------
        scores: np.ndarray
            An array of class scores (higher score indicating higher
            confidence in the corresponding class).
        label: ClassIdentifier
            A class identifier for the class in question.

        Results
        -------
        rank: int
            Zero-based rank of the given class.  0 means that the
            given class is the most likely one, while larger rank
            indicates less confidence.
        """
        # FIXME[todo]: batch = np.arange(len(scores))
        score = scores[label[self._lookup]]
        rank = (scores > score).sum()

        return rank, score

    def print_classification(self, scores: np.ndarray, top: int = 5):
        """Output the classification scores.
        """
        top_indices, top_scores = self.top_classes(scores, top=top)
        for i, (index, score) in enumerate(zip(top_indices, top_scores)):
            print(f"  {i}: {index} ({score})")
