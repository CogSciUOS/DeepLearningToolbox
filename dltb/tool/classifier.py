"""General definitions for classifiers.

"""

# FIXME[concept]: there should be a connection between network
#   and suitable datasets/labels, i.e., AlexNet should provide
#   class labels, even if not applied to ImageNet data

# standard imports
from typing import Union, Tuple, Any
import logging

# third party imports
import numpy as np

# toolbox imports
from ..base.data import Data, ClassScheme, ClassIdentifier
from .image import ImageTool

# FIXME[hack]: instead of prepare_input_image use the network.resize
# API once it is finished!
from dltb.base.image import Imagelike, Image
from dltb.util.image import imread, imresize

# logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


class Classifier:
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

    _result: Tuple[str] = ('identifier')
    _internal_arguments: Tuple[str] = ('_inputs')
    _internal_result: Tuple[str] = ('identifier')

    def __init__(self, scheme: Union[ClassScheme, str, int],
                 lookup: str = None, **kwargs):
        LOG.debug("Classifier[scheme={%s}]: %s", scheme, kwargs)
        super().__init__(**kwargs)
        if isinstance(scheme, str):
            scheme = ClassScheme.register_initialize_key(scheme)
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

    def _prepare(self, **kwargs) -> None:
        """Prepare this :py:class:`Classifier`.

        Raises
        ------
        ValueError
            If the :py:class:`ClassScheme` does not fit to this
            :py:class:`Classifier`.
        """
        super()._prepare()
        self._scheme.prepare()

        
    def preprocess(self, data: Data) -> None:
        """Preprocess the given data to a format suitable to be processed by
        this :py:class:`Classifier`. The actual operations to be
        performed depend on the requirements of the
        :py:class:`Classifier` and have to be implemented by the
        classifier.

        """
        pass

    def classify(self, inputs: np.ndarray):
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
        """
        return self(inputs)

    #
    # Private processor API
    #

    def _preprocess(self, inputs: np.ndarray, *args, **kwargs) -> np.ndarray:
        """A :py:class:`SoftClassifier` maps input data to class scores.
        """
        data = super()._preprocess(inputs, *args, **kwargs)
        if inputs is not None:
            data.add_attribute('_inputs', Data.as_array(inputs))
        return data
    
    def _process(self, inputs: np.ndarray) -> np.ndarray:
        """A :py:class:`SoftClassifier` maps input data to class scores.
        """
        # return the class scores
        raise NotImplementedError()


class SoftClassifier(Classifier):

    _result: Tuple[str] = ('scores')
    _internal_arguments: Tuple[str] = ('_inputs')
    _internal_result: Tuple[str] = ('scores')

    def _process(self, inputs: np.ndarray) -> np.ndarray:
        """A :py:class:`SoftClassifier` maps input data to class scores.
        """
        # return the class scores
        raise NotImplementedError()
    
    def class_scores(self, inputs: np.ndarray) -> np.ndarray:
        """Compute all class scores. The output array will have one entry for
        each class of the classification scheme.

        Arguments
        ---------
        inputs: np.ndarray
            The input data, either a single data point or a batch of data.
        """
        # FIXME[concept]: we need a way to determine if inputs are single or
        # batch!
        return self(inputs)

    def classify(self, inputs: np.ndarray, top: int = None):
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
        score:
            If top is None, a one-dimensial array of confidence values or
            otherwise a two-dimension array providing the top highest
            confidence values for each input item.
        """
        return self.top_classes(self(inputs), top=top)

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

        return class_identifiers, scores[batch, top_indices.T].T

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
        score = scores[label.label(self._lookup)]
        rank = (scores > score).sum()

        return rank, score

    def print_classification(self, scores: np.ndarray, top: int = 5):
        top_indices, top_scores = self.top_classes()
        for i, (index, score) in enumerate(zip(top_indices, top_scores)):
            print(f"  {i}: {index} ({score})")


class ImageClassifier(Classifier, ImageTool):
    """An :py:class:`ImageClassifier` is a classifier for images.
    """

    def preprocess_image(self, image: Imagelike) -> Any:
        """Preprocess a single image to be in a format that
        can be used as input for this :py:class:`ImageClassifier`.
        This may include resizing the image, as well centering and
        standardization, or adding a batch dimension.
        """
        return self._preprocess(Image.as_array(image))

    def _preprocess(self, image, *arg, **kwargs) -> Data:
        data = super()._preprocess(image, *arg, **kwargs)
        if image is not None:
            image = np.expand_dims(image, axis=0)
            self.add_data_attribute(data, 'image', image)
        return data

    def _image_as_batch(self, image: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image, str):
            # FIXME[todo]: general resizing/preprocessing strategy for networks
            size = self.get_input_shape(include_batch=False,
                                        include_channel=False)
            image = imread(image)  # , size=size
            image = imresize(image, size=size)

        # add batch dimension
        image = np.expand_dims(image, axis=0)
        return image

    def classify_image(self, image: Union[str, np.ndarray],
                       top: int = None, confidence: bool = False
                       ) -> Union[ClassIdentifier, Tuple[ClassIdentifier]]:
        """Classify the given image.

        Arguments
        ---------
        image: Union[str, np.ndarray]
            The image to classify, either as image filename or as
            numpy array.
        top: int
            Number of top classification results to report. If not provided,
            the single best class will be returned

        Result
        ------
        class or classes:
            Either a single class or a tuple of the top best matches.
        confidence(s) (optional):
            If the confidence argument is True, the corresponding confidence
            value (or a tuple of confidence values).
        """
        image_batch = self._image_as_batch(image)
        classes, scores = self.classify(image_batch, top=top)

        return (classes[0], scores[0]) if confidence else classes[0]
