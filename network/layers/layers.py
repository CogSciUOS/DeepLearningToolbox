from typing import Tuple
import numpy as np
from collections import OrderedDict

from util import Identifiable

# ------------ Base layers ----------------------

class Layer(Identifiable):
    """A Layer encapsulates operations that transform a tensor."""

    _predecessor: 'Layer' = None
    _sucessor: 'Layer' = None
    
    def __init__(self, network, id=None):
        super().__init__(id)
        self._network = network

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """The shape of the tensor that is input to the layer."""
        raise NotImplementedError

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """The shape of the tensor that is output to the layer."""
        raise NotImplementedError

    @property
    def predecessor(self) -> 'Layer':
        """The preceeding Layer in a network."""
        return self._predecessor

    @property
    def sucessor(self) -> 'Layer':
        """The suceeding Layer in a network."""
        return self._sucessor

    def receptive_field(self, p1: Tuple[int, ...], p2: Tuple[int, ...]=None
                        ) -> (Tuple[int, ...], Tuple[int, ...]):
        """The receptive field of a layer.

        Parameters
        ---------- 
        p1: the upper left corner of the region of interest.
        p2: the lower right corner of the region of interest.

        Result
        ------
        The upper left corner and the lower right corner of the
        receptive field for the region of interest.
        """
        if self.predecessor is None:
            return p1, p2
        return self.predecessor.receptive_field(p1,p2)

    @property
    def info(self) -> OrderedDict:
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

    @property
    def info(self) -> OrderedDict:
        info_dict = super().info
        info_dict['num_parameters'] = self.num_parameters
        return info_dict

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
        """The padding strategy used for striding operations.

        Returns
        -------
        Either 'valid' or 'same'.
        """
        raise NotImplementedError

    def receptive_field(self, p1: Tuple[int, ...], p2: Tuple[int, ...]=None
                        ) -> (Tuple[int, ...], Tuple[int, ...]):
        """The receptive field of a layer.

        Parameters
        ----------
        p1: the upper left corner of the region of interest.
        p2: the lower right corner of the region of interest.

        Result
        ------
        q1: The upper left corner of the receptive field for the region
            of interest.
        q2: The lower right corner of the receptive field for the region
            of interest.
        """
        if p2 is None:
            p2 = p1
        q1 = (), q2 = ()
        for i, s in enumerate(self.strides):
            q1 = q1 + (p[i] * s)
            q2 = q2 + (p[i] * s)

        return self.predecessor.receptive_field(q1,q2)

    @property
    def info(self) -> OrderedDict:
        info_dict = super().info
        info_dict['strides'] = self.strides
        info_dict['padding'] = self.padding
        return info_dict


# -------------- Neural layers ---------------


class Dense(NeuralLayer):
    pass

class Conv2D(NeuralLayer, StridingLayer):

    @property
    def kernel_size(self) -> Tuple[int, int]:
        raise NotImplementedError

    @property
    def filters(self) -> int:
        """

        Returns
        -------
        The number of filter maps created by the convolution layer.
        """
        raise NotImplementedError

    def receptive_field(self, p1: Tuple[int, ...], p2: Tuple[int, ...]=None
                        ) -> (Tuple[int, ...], Tuple[int, ...]):
        """The receptive field of a layer.

        Parameters
        ----------
        p1: the upper left corner of the region of interest.
        p2: the lower right corner of the region of interest.

        Result
        ------
        q1: The upper left corner of the receptive field for the region
            of interest.
        q2: The lower right corner of the receptive field for the region
            of interest.
        """
        p1, p2 = super().receptive_field(self, p1, p2)
        q1 = (), q2 = ()
        for i, s in enumerate(self.kernel_size):
            if self.padding == 'same':
                q1 = q1 + (p[i] - s//2)
                q2 = q2 + (p[i] + s//2)
            else:
                q1 = q1 + (p[i])
                q2 = q2 + (p[i] + s-1)

        return self.predecessor.receptive_field(q1,q2)

    @property
    def info(self) -> OrderedDict:
        info_dict = super().info
        info_dict['kernel_size'] = self.kernel_size
        info_dict['filters'] = self.filters
        return info_dict


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

    def receptive_field(self, p1: Tuple[int, ...], p2: Tuple[int, ...]=None
                        ) -> (Tuple[int, ...], Tuple[int, ...]):
        """The receptive field of a layer.

        Parameters
        ----------
        p1: the upper left corner of the region of interest.
        p2: the lower right corner of the region of interest.

        Result
        ------
        q1: The upper left corner of the receptive field for the region
            of interest.
        q2: The lower right corner of the receptive field for the region
            of interest.
        """
        p1, p2 = super().receptive_field(self, p1, p2)
        q1 = (), q2 = ()
        for i, s in enumerate(self.pool_size):
            if self.padding == 'same':
                q1 = q1 + (p[i] - s//2)
                q2 = q2 + (p[i] + s//2)
            else:
                q1 = q1 + (p[i])
                q2 = q2 + (p[i] + s-1)

        return self.predecessor.receptive_field(q1,q2)

    @property
    def info(self) -> OrderedDict:
        info_dict = super().info
        info_dict['pool_size'] = self.pool_size
        return info_dict


class Dropout(Layer):
    pass

class Flatten(Layer):
    pass
