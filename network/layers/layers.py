from typing import Tuple
import numpy as np


# ------------ Base layers ----------------------


class Layer:
    """A Layer encapsulates operations that transform a tensor."""
    def __init__(self, network):
        self._network = network

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """The shape of the tensor that is input to the layer."""
        raise NotImplementedError

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """The shape of the tensor that is output to the layer."""
        raise NotImplementedError


class NeuralLayer(Layer):
    """A NeuralLayer is assumed to encapsulated three operations:
    Calculating the net input by taking some inner product between weights and input.
    Adding a bias.
    Applying some (mostly non-linear) activation function.
    """

    @property
    def parameters(self) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError

    @property
    def num_parameters(self) -> int:
        return sum(param.size for param in self.parameters)

    @property
    def weights(self) -> np.ndarray:
        raise NotImplementedError
    @property
    def bias(self) -> np.ndarray:
        raise NotImplementedError



class StridingLayer(Layer):
    @property
    def strides(self) -> Tuple[int, int]:
        """

        Returns
        -------
        The strides in height and width direction.
        """
        raise NotImplementedError

    @property
    def padding(self) -> str:
        """The padding stragety used for striding operations.

        Returns
        -------
        Either 'valid' or 'same.
        """
        raise NotImplementedError


# -------------- Neural layers ---------------


class Dense(NeuralLayer):
    pass

class Conv2D(NeuralLayer, StridingLayer):

    @property
    def kernel_size(self) -> Tuple[int, int]:
        raise NotImplementedError

    # called filters in keras
    @property
    def num_channels(self) -> int:
        """

        Returns
        -------
        The number of filter maps created by the convolution layer.
        """
        raise NotImplementedError


# --------------- Transformation layers ------------

class MaxPooling2D(StridingLayer):
    @property
    def pool_size(self) -> Tuple[int, int]:
        """

        Returns
        -------
        The number of pixels pooled in height and width direction in each pooling step.
        """
        raise NotImplementedError


class Dropout(Layer):
    pass

class Flatten(Layer):
    pass






























