"""Base classes for network layers.
"""

# standard imports
from typing import Tuple, List
import numpy as np
from collections import OrderedDict

# toolbox imports
from base import Identifiable

# ------------ Base layers ----------------------

class Layer(Identifiable):
    """A Layer encapsulates operations that transform a tensor.

    Attributes
    ----------
    _network: Network
        The :py:class:`Network` this :py:class:`Layer` is part of.
    _predecessor: Layer
        The :py:class:`Layer` preceeding this :py:class:`Layer` in the
        network. May be `None`, meaning that this layer is an input
        layer.
    _successor: Layer
        The :py:class:`Layer` preceeding this :py:class:`Layer` in
        the network. May be `None`, meaning that this layer is an
        input layer.
    """

    def __init__(self, network, id: str = None):
        super().__init__(id)
        self._network = network
        self._predecessor = None
        self._sucessor = None

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """The shape of the tensor that is input to the layer."""
        raise NotImplementedError

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """The shape of the tensor that is output to the layer."""
        raise NotImplementedError

    @property
    def network(self) -> 'Network':
        return self._network

    @property
    def predecessor(self) -> 'Layer':
        """The :py:class:`Layer` preceeding this :py:class:`Layer` in the
        network. May be `None`, meaning that this layer is an input
        layer.

        """
        return self._predecessor

    @property
    def sucessor(self) -> 'Layer':
        """The suceeding Layer in a network."""
        return self._sucessor

    @staticmethod
    def compute_receptive_field(point1: Tuple[int, ...],
                                point2: Tuple[int, ...],
                                size: Tuple[int, ...],
                                padding: Tuple[int, ...],
                                strides: Tuple[int, ...]) \
            -> (Tuple[int, ...], Tuple[int, ...]):
        """The receptive field of a layer.

        Parameters
        ----------
        point1: the upper left corner of the region of interest.
        point2: the lower right corner of the region of interest.

        Returns
        ------
        q1: The upper left corner of the receptive field for the region
            of interest.
        q2: The lower right corner of the receptive field for the region
            of interest.
        """
        print(f"compute_receptive_field({point1},{point2},{size},"
              f"{padding},{strides})")
        if point2 is None:
            point2 = point1
        q1, q2 = (), ()
        for i, s in enumerate(strides):
            q1 = q1 + (point1[i] * s,)
            q2 = q2 + (point2[i] * s,)

        point1, point2 = (), ()
        for i, (s, p) in enumerate(zip(size, padding)):
            point1 = point1 + (q1[i] - p,)
            point2 = point2 + (q2[i] + s - p,)
        return point1, point2

    def receptive_field(self, point1: Tuple[int, ...],
                        point2: Tuple[int, ...] = None
                        ) -> (Tuple[int, ...], Tuple[int, ...]):
        """The receptive field of a layer.

        Parameters
        ----------
        point1:
            The upper left corner of the region of interest.
        point2:
            The lower right corner of the region of interest.
            If none is given, the region is assumed to consist of just
            one point.

        Returns
        -------
        point1, point2:
            The upper left corner and the lower right corner of the
            receptive field for the region of interest.
        """
        return ((point1, point2) if self.predecessor is None else
                self.predecessor.receptive_field(point1, point2))

    @property
    def info(self) -> OrderedDict:
        """Metadata of this :py:class:`Layer`.
        """
        info_dict = OrderedDict()
        info_dict['input_shape'] = self.input_shape
        info_dict['output_shape'] = self.output_shape
        return info_dict


class NeuralLayer(Layer):
    """A NeuralLayer is assumed to encapsulated three operations:
    Calculating the net input by taking some inner product between weights
    and input.
    Adding a bias.
    Applying some (mostly non-linear) activation function.
    """

    @property
    def parameters(self) -> Tuple[np.ndarray, ...]:
        """The parameters (connection weights and bias vector) of this
        :py:class:`NeuralLayer`.
        """
        raise NotImplementedError

    @property
    def num_parameters(self) -> int:
        """The total number of parameters of this :py:class:`NeuralLayer`.
        """
        return sum(param.size for param in self.parameters)

    @property
    def weights(self) -> np.ndarray:
        """The connection weight matrix of this :py:class:`NeuralLayer`.

        """
        raise NotImplementedError

    @property
    def bias(self) -> np.ndarray:
        """The bias vector of this :py:class:`NeuralLayer`.

        """
        raise NotImplementedError

    @property
    def info(self) -> OrderedDict:
        """A :py:class:`NeuralLayer` adds `'num_parameters'` to the metadata.

        """
        info_dict = super().info
        info_dict['num_parameters'] = self.num_parameters
        return info_dict


class StridingLayer(Layer):
    """A striding layer acts like a local filter, striding over the
    input map.
    """

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
        """The padding strategy used for striding operations.

        Returns
        -------
        Either 'valid' or 'same'.
        """
        raise NotImplementedError

    @property
    def filter_size(self) -> str:
        """The filter size of this striding layer.

        Returns
        -------
        Either 'valid' or 'same'.
        """
        raise NotImplementedError

    def receptive_field(self, point1: Tuple[int, ...],
                        point2: Tuple[int, ...] = None
                        ) -> (Tuple[int, ...], Tuple[int, ...]):
        """The receptive field of a layer.

        Parameters
        ----------
        point1: the upper left corner of the region of interest.
        point2: the lower right corner of the region of interest.

        Returns
        ------
        q1: The upper left corner of the receptive field for the region
            of interest.
        q2: The lower right corner of the receptive field for the region
            of interest.
        """
        point1, point2 = \
            self.compute_receptive_field(point1, point2, self.filter_size,
                                         self.padding, self.strides)
        return super().receptive_field(point1, point2)

    @property
    def info(self) -> OrderedDict:
        """A :py:class:`StridingLayer` adds `'strides'` and `'padding'` to the
        metadata.
        """
        info_dict = super().info
        info_dict['strides'] = self.strides
        info_dict['padding'] = self.padding
        return info_dict


# -------------- Neural layers ---------------


class Dense(NeuralLayer):
    """A dense layer (sometimes also called "fully connected layer").
    This layer connects every input unit with every output unit.
    
    A dense layer removes all spatial structure that may be present in
    the input activation map, as each output unit can be affected by
    all input units.
    """


class Conv2D(NeuralLayer, StridingLayer):
    """A :py:class:`Conv2D` applies a two-dimensional convolution
    operation to the (two-dimensional) input activation map.
    """

    @property
    def kernel_size(self) -> Tuple[int, int]:
        """The kernel size of this :py:class:`Conv2D` layer.
        """
        raise NotImplementedError

    @property
    def filter_size(self) -> Tuple[int, int]:
        """The filter size of a convolutional layer is the kernel size.

        Returns
        -------
        """
        return self.kernel_size

    @property
    def filters(self) -> int:
        """

        Returns
        -------
        The number of filter maps created by the convolution layer.
        """
        raise NotImplementedError

    @property
    def info(self) -> OrderedDict:
        """A :py:class:`Conv2D` layer adds `'kernel_size'` and `'filters'` to
        the metadata.

        """
        info_dict = super().info
        info_dict['kernel_size'] = self.kernel_size
        info_dict['filters'] = self.filters
        return info_dict


# --------------- Transformation layers ------------

class MaxPooling2D(StridingLayer):
    """A :py:class:`MaxPooling2D` summarizes multiple input values
    into a single output value.
    """

    @property
    def pool_size(self) -> Tuple[int, int]:
        """

        Returns
        -------
        The number of pixels pooled in height and width direction in each pooling step.
        """
        raise NotImplementedError

    @property
    def filter_size(self) -> Tuple[int, int]:
        """The filter size of a pooling layer is the pool size.

        Returns
        -------
        """
        return self.pool_size

    @property
    def info(self) -> OrderedDict:
        """A :py:class:`Conv2D` layer adds `'pool_size'` to the metadata.

        """
        info_dict = super().info
        info_dict['pool_size'] = self.pool_size
        return info_dict


class Dropout(Layer):
    """A :py:class:`Dropout` layer randomly drops some inputs.
    """

    @property
    def rate(self) -> float:
        """The dropout rate indicating the ratio of inputs to be dropped.
        This should be a value between `0.0` and `1.0`
        """
        raise NotImplementedError

class Flatten(Layer):
    """A :py:class:`Flatten` layer removes spatial structure from
    the input activation map, resulting in a long, one-dimensional
    feature vector.
    """
