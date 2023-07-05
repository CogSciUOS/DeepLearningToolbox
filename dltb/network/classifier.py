"""Base class for networks that act as classifier.
"""
# standard imports

# third party imports
import numpy as np

# toolbox imports
from ..base.data import Datalike
from ..tool.classifier import SoftClassifier
from .network import Network
from .layer import Layer


class Classifier(SoftClassifier, Network):
    """A :py:class:`Network` to be used as classifier.

    _labeling: str
        The name of the label lookup table of the :py:class:`ClassScheme`
        by which the activation vector of the output layer(s), i.e.
        the probit/score is indexed.
    """

    def __init__(self, labeling: str = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._labeling = labeling

    @property
    def labeling(self) -> str:
        """The name of the label lookup table of the :py:class:`ClassScheme`
        by which the activation vector of the output layer(s), i.e.
        the probit/score is indexed.
        """
        return self._labeling

    @property
    def logit_layer(self) -> Layer:
        """The layer providing the class "logits".
        This is usually the prefinal layer that is passed
        in the softmax output layer.
        """
        return self.get_output_layer()  # FIXME[todo]

    @property
    def score_layer(self) -> Layer:
        """The layer computing the class scores ("probabilities").
        This is usually the output layer of the :py:class:`Classifier`.
        """
        return self.get_output_layer()

    def _prepare(self) -> None:
        """Prepare this :py:class:`Classifier`.

        Raises
        ------
        ValueError
            If the :py:class:`ClassScheme` does not fit to this
            :py:class:`Classifier`.
        """
        super()._prepare()

        # check if the output layer fits to the classification scheme
        output_shape = self.get_output_shape(include_batch=False)
        if len(output_shape) > 1:
            raise ValueError(f"Network '{self}' seems not to be Classifier "
                             f"(output shape is {output_shape})")
        if output_shape[0] != len(self._scheme):
            raise ValueError(f"Network '{self}' does not fit the the "
                             f"classification scheme: {output_shape} output "
                             f"units vs. {len(self._scheme)} classes")

    #
    # Implementation of the SoftClassifier API
    #

    def class_scores(self, inputs: Datalike,
                     probit: bool = False) -> np.ndarray:
        """Implementation of the :py:class:`SoftClassifier` interface.

        Arguments
        ---------
        inputs:
            The input data (either individual datum or batch of data).

        Results
        -------
        scores:
            The class scores obtained by classifying the input,
            indexed according to the :py:clas:`ClassScheme` of this
            :py:class:`Classifier`.
        """
        # obtain activation values for the score_layer
        activations = self.get_activations(inputs, self.score_layer)

        # convert scores from internal format into numpy array
        return self.to_class_scheme(activations)

    def to_class_scheme(self, activations: np.ndarray) -> np.ndarray:
        """Reindex a given activation vector acoording to the
        :py:class:`ClassScheme` of this :py:class:`Classifier`. The
        reindexed activations vector can be used to directly read out
        activation values, using the :py:class:`ClassIdentifier`s of
        the :py:class:`ClassScheme` as index.
        """
        return self._scheme.reindex(activations, source=self._labeling)
