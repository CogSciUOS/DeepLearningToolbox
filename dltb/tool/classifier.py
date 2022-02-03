"""General definitions for classifiers.

"""

# FIXME[concept]: there should be a connection between network
#   and suitable datasets/labels, i.e., AlexNet should provide
#   class labels, even if not applied to ImageNet data

# standard imports
from typing import Union, Tuple, Iterable, Any, Sequence, Optional
from typing import TypeVar, overload
from abc import abstractmethod, ABC
import logging

# third party imports
import numpy as np

# toolbox imports
from ..typing import Protocol
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
            The result format will correspond to the format of this argument.
        name:
            The name of the label format (output value). If `None`,
            the default format (the integer value) is returned.
        lookup:
            The name of the lookup table for the index (input value).
            If `None`, no lookup table will be used.

        Result
        ------
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
            self.create_lookup_table(name)

    def create_lookup_table(self, name: str) -> None:
        """Add a (reverse) lookup table for label set.  This method
        can be called after a label set has been added to this
        :py:class:`ClassScheme`.  It is essentially equivalent to
        passing the `lookup=True` flag when adding the label set with
        :py:meth:`add_labels`.

        Arguments
        ---------
        name: str
            The name of the label set. This name has to be used when
            looking up labels in this label set.
        """
        values = self._labels[name]  # may raise KeyError
        if name in self._lookup:
            return  # lookup table already exists -> nothing to do
        if isinstance(values, np.ndarray) and values.max() < 2*len(self):
            self._lookup[name] = np.zeros(values.max()+1, dtype=int)
            self._lookup[name][values] = np.arange(0, values.size)
        else:
            self._lookup[name] = \
                {val: idx for idx, val in enumerate(values)}

    def labels(self, lookup: bool = None) -> Iterable[str]:
        """Iterate over the names of the label sets that are currently
        registered with this :py:class:`ClassScheme`.

        """
        for label in self._labels:
            if lookup is None:
                yield label
            elif lookup and label in self._lookup:
                yield label
            elif not lookup and label not in self._lookup:
                yield label

    def reindex(self, values: np.ndarray, axis: int = None,
                source: str = None, target: str = None) -> np.ndarray:
        """Reindex the given
        """
        if len(values) != len(self):
            raise ValueError("Invalid values for reindexing: length is "
                             f"{len(values)} but should be {len(self)}")
        if source == target:
            return values

        translation = self.translate(np.arange(len(self)),
                                     source=source, target=target)
        return values[translation]

    def translate(self, labels: np.ndarray,
                  source: str = None, target: str = None) -> np.ndarray:
        """Translate an array of indices from a given indexing format
        into another format.

        Arguments
        ---------
        labels:
            The array of indices.
        source:
            The name of the source index format.  If no source name
            is provided, the indices are assumed to be in the standard
            format of this :py:class:`ClassScheme`.
        target:
            The name of the target format. If no target name is provided,
            the standard indices of this :py:class:`ClassScheme` are used.

        Result
        ------
        translated
            An array containing the translated labels.
        """
        if source is not None:
            labels = self._lookup[source][labels]
        if target is not None:
            labels = self._labels[target][labels]
        return labels


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

    Use this class like this:

    .. code-block:: python

        label = classifier.classify(data)
        labels = classifier.classify(batch)


    Attributes
    ----------

    _scheme: ClassScheme
        The classification scheme applied by this classifier.

    _lookup: str
        Name of a lookup table for the given `ClassScheme`. This
        table is used to map (numeric) outputs of the classifier
        to classes of that `ClassScheme`.
    """

    def __init__(self, scheme: Optional[Union[ClassScheme, str, int]] = None,
                 lookup: Optional[str] = None, **kwargs):
        LOG.debug("Classifier[scheme={%s}]: %s", scheme, kwargs)
        super().__init__(**kwargs)
        if isinstance(scheme, str):
            scheme = ClassScheme[scheme]
        elif isinstance(scheme, int):
            scheme = ClassScheme(scheme)
        elif scheme is None:
            raise ValueError("No class scheme provided for Classifier.")
        self._scheme = scheme
        self._lookup = lookup

    @property
    def scheme(self) -> ClassScheme:
        """The :py:class:`ClassScheme` used by this :py:class:`Classifier`.
        """
        return self._scheme

    # FIXME[old]: why scheme and class_scheme?
    # -> maybe classes can have other schemes?
    @property
    def class_scheme(self) -> ClassScheme:
        """The :py:class:`ClassScheme` used by this :py:class:`Classifier`.
        """
        return self._scheme

    @property
    def lookup(self) -> Optional[str]:
        """The lookup table for the :py:class:`ClassScheme` used by
        this :py:class:`Classifier` to map internal class numbers
        to class identifiers and class labels.  If `None`, this
        `Classifier` uses the default indices of the `ClassScheme`.
        """
        return self._lookup

    def _prepared(self) -> None:
        """Prepare this :py:class:`Classifier`.
        """
        return super()._prepared and self._scheme.prepared

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

    def get_label(self, index: int, name: str = None) -> Any:
        """Look up a label for a given index for this
        :py:class:`ClassScheme`.

        Attributes
        ----------
        index:
            A single class index or a collection of indices (list or tuple).
            The result format will correspond to the format of this argument.
        name:
            The name of the label format (output value). If `None`,
            the default format (the integer value) is returned.
        """
        return self.scheme.get_label(index, name=name, lookup=self._lookup)

    def class_identifier(self, index: int) -> ClassIdentifier:
        """Get a :py:class:`ClassIdentifier` representing a class index
        for this class.
        """
        return (index if isinstance(index, ClassIdentifier) else
                self.scheme.identifier(index, lookup=self._lookup))


class ImageClassifier(ImageExtension, base=Classifier):
    """An :py:class:`ImageClassifier` is can classify images.
    """

    # FIXME[old]: not used (and probably not needed anymore), as
    # classes new classes should now how to convert arguments into
    # an internal format ...
    def classify_image(self, image: Imagelike) -> ClassIdentifier:
        """Classify the given image.
        """
        return self.classify(self.image_to_internal(image))


class SoftClassifier(Classifier):
    """A :py:class:`SoftClassifier` maps an input item to score vector
    describing the confidence with which the item belongs to the
    corresponding class of a classification scheme.

    """
    # FIXME[concept]: there may be different types of scores.  For
    # example, many deep network classifiers apply a softmax to the
    # last layer (probits) to obtain a normalized (sum 1) output
    # vector.  Depending on the application, one or the other may
    # be of interest.  One may argue, that the probits contain
    # more information than the normalized vector.
    # But which of them are the "class_scores"?

    @abstractmethod
    def class_scores(self, inputs: Datalike,
                     softmax: bool = False) -> np.ndarray:
        """Compute all class scores. The output array will have one entry for
        each class of the classification scheme.

        Arguments
        ---------
        inputs: 
            The input data, either a single data point or a batch of data.
        softmax:
            Usually the method will return unnormalized scores
            (if available). If this flag is set, the returned
            values are normalized using the softmax function.
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
            The class scores, either probit or normalized ("probabilities").
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

    def print_classification(self, inputs: Datalike, top: int = 5):
        """Output the classification scores.
        """
        scores = self.class_scores(inputs)
        top_indices, top_scores = self.top_classes(scores, top=top)
        for i, (index, score) in enumerate(zip(top_indices, top_scores)):
            print(f"  {i+1}: {self.scheme.get_label(index, self._lookup)} "
                  f"({self.scheme.get_label(index, 'text')}) "
                  f"[{score:.4f}]")


class BinaryClassifier(Classifier):
    """A binary classifier assigns input points to one of two classes.
    This is a very common form of classification with many applications.

    """

    def evaluate(self, datasource: Sequence):
        confusion_matrix = ConfusionMatrix(classes=2)
        for data in datasource:
            prediction = self(data)
            confusion_matrix[data.label, prediction] += 1


class ConfusionMatrix:
    def __init__(self, classes: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self._matrix = np.zeros(2, 2, dtype=np.int)

    def __getitem__(self, pos: Tuple[int, int]) -> int:
        correct, predicted = pos
        return self._matrix[correct, predicted]

    def __setitem__(self, pos: Tuple[int, int], value: int) -> None:
        correct, predicted = pos
        self._matrix[correct, predicted] = value

    def accuracy(self) -> float:
        tp, fp, tn, fn = 0, 0, 0, 0
        for idx, row in enumerate(self._matrix):
            correct = row[idx]
            error = row.sum() - correct


class PairClassifier(BinaryClassifier):
    """A pair classifier obtains a pair of input points and should
    determine if these are same or different (e.g. two images of the
    same cat).

    """

    def process_pair(data) -> bool:
        return data[0] == data[1]  # FIXME[todo]

    def evaluate_pairs(pairs) -> float:
        # taken from evaluate arcface

        print("Start evaluating on age group: " + current_age_group)

        for img1, img2, same in pairs:
            img1 = short_preprocessing(img1)
            img2 = short_preprocessing(img2)

            out_1 = l2_norm(arcface_model(img1))
            out_1 = out_1[0].numpy()

            out_2 = l2_norm(arcface_model(img2))
            out_2 = out_2[0].numpy()

            embeddings_array[idx, 0] = out_1
            embeddings_array[idx, 1] = out_2

            gt_array[idx] = int(gt_same_person)

        tpr, fpr, accuracy, best_thresholds = \
            calculate_roc(thresholds,
                          embeddings_array[:,0], embeddings_array[:,1],
                          gt_array, nrof_folds=10)

        print("Accuracies for age group "+current_age_group+":")
        print(accuracy)
        mean = accuracy.mean()
        print("Mean Accuracy for age group "+current_age_group+" is: "+
              str(mean))


class MetricPairVerifier:
    """A pair verifier that decides for a pair if both components are the
    same based on their distance: if the distance is below a given
    threshold, the components are judged as same, otherwise they are
    judged as different.

    A :py:class:`MetricPairVerifier` has to implement a
    :py:meth:`distance` method that allows to compute the distance for
    a pair of points

    """

    def threshold(self) -> float:
        """The threshold for the distance between the two components
        of a pair to be considered the same.
        """
        return self._threshold

    def is_same(self, data1, data2, threshold: float = None) -> bool:
        """Check for two data points (or batch of data), if they
        are to be considered the same (their distance is below the
        threshold) or different (their distance is above the threshold).
        """
        threshold = threshold or self.threshold
        distance = self.distance(data1, data2)
        is_same = distance < threshold
        return is_same

    def calculate_accuracy(self, threshold, dist, actual_issame):
        """Calculate accuracy and other metrics.

        based on arcface-tf2 code
        """
        threshold = threshold or self.threshold
        predict_issame = np.less(dist, threshold)
        # pylint: disable=invalid-name
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame,
                                   np.logical_not(actual_issame)))
        tn = np.sum(np.logical_and(np.logical_not(predict_issame),
                                   np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame),
                                   actual_issame))

        tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
        fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
        acc = float(tp + tn) / dist.size
        return tpr, fpr, acc

    def calculate_roc(self, thresholds, embeddings1, embeddings2,
                      actual_issame, nrof_folds=10):
        """Comput a ROC curve

        based on arcface-tf2 code
        """
        assert embeddings1.shape[0] == embeddings2.shape[0]
        assert embeddings1.shape[1] == embeddings2.shape[1]
        nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
        nrof_thresholds = len(thresholds)
        k_fold = KFold(n_splits=nrof_folds, shuffle=False)

        tprs = np.zeros((nrof_folds, nrof_thresholds))
        fprs = np.zeros((nrof_folds, nrof_thresholds))
        accuracy = np.zeros((nrof_folds))
        best_thresholds = np.zeros((nrof_folds))
        indices = np.arange(nrof_pairs)

        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

        for idx, (train_set, test_set) \
                in enumerate(k_fold.split(indices)):
            # Find the best threshold for the fold
            acc_train = np.zeros((nrof_thresholds))
            for threshold_idx, threshold in enumerate(thresholds):
                _, _, acc_train[threshold_idx] = \
                    self.calculate_accuracy(threshold, dist[train_set],
                                            actual_issame[train_set])
                best_threshold_index = np.argmax(acc_train)

            best_thresholds[idx] = thresholds[best_threshold_index]
            for threshold_idx, threshold in enumerate(thresholds):
                tprs[idx, threshold_idx], fprs[idx, threshold_idx], _ = \
                    self.calculate_accuracy(threshold, dist[test_set],
                                            actual_issame[test_set])
            _, _, accuracy[idx] = \
                self.calculate_accuracy(thresholds[best_threshold_index],
                                        dist[test_set],
                                        actual_issame[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        return tpr, fpr, accuracy, best_thresholds

    #
    # to be implemented by sublasses
    #

    _threshold: float = 1.


# ---------------------------------------------------------------------------
# FIXME[todo]: First ideas of ClassifierProtocols, not yet finished and
# not yet used.

DataType = TypeVar('DataType')
ClassSchemeType = TypeVar('ClassSchemeType')

# ClassIdentifierType could be ClassScheme.Identifer
ClassIdentifierType = TypeVar('ClassIdentifierType')


class Batch(Protocol[DataType]):
    """
    """


class ClassifierProtocol(Protocol[DataType, ClassSchemeType,
                                  ClassIdentifierType]):
    """
    """

    @property
    def class_scheme(self) -> ClassSchemeType:
        """The :py:class:`ClassScheme` used by this :py:class:`Classifier`.
        """

    @overload
    def classify(self, inputs: DataType) -> ClassIdentifierType:
        """Output the classes for the given inputs.

        Arguments
        ---------
        inputs: np.ndarray
            A data point or a batch of input data to classify.

        Results
        -------
        classes:
            A class-identifier or a batch of class identifiers.
        """

    @overload
    def classify(self, inputs: Batch[DataType]) \
            -> Batch[ClassIdentifierType]:
        """Classify a batch of data.
        """


class SoftClassifierProtocol(ClassifierProtocol[DataType, ClassSchemeType,
                                                ClassIdentifierType]):

    @property
    def class_scheme(self) -> ClassSchemeType:
        """The :py:class:`ClassScheme` used by this :py:class:`Classifier`.
        """

    @overload
    def classify(self, inputs: DataType, top: int) \
            -> Sequence[ClassIdentifierType]:
        """Output the classes for the given inputs.

        Arguments
        ---------
        inputs: np.ndarray
            A data point or a batch of input data to classify.

        Results
        -------
        classes:
            A class-identifier or a batch of class identifiers.
        """

    @overload
    def classify(self, inputs: Batch[DataType], top: int) \
            -> Batch[Sequence[ClassIdentifierType]]:
        """Classify a batch of data and return a batch of top classes.
        """

    def class_scores(self, inputs: DataType, softmax: bool = False) \
            -> Sequence[float]:
        """Provide the class scores for the given input data.

        Argument
        --------
        inputs:
            The input data to classify.
        softmax:
            Usually the method will return unnormalized scores
            (if available). If this flag is set, the returned
            values are normalized using the softmax function.
        """
