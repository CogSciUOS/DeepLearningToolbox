"""Base classes for neural network definition.

Base classes are introduced in a separate file to avoid circular
definitions for typing.

Maybe these classes can be realized as Protocols.

"""
# standard imports
from typing import Tuple, Any, Union
from collections import OrderedDict

# toolbox imports
from ..base.register import RegisterEntry


class LayerBase(RegisterEntry):
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

    def __init__(self, network: 'Network', key: str = None) -> None:
        super().__init__(key=key)
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


class NetworkBase:
    """Interface for the `Network` class.
    """

    def __getitem__(self, key: Any) -> LayerBase:
        ...


# For passsing a layer, one can either pass the layer object, or
# just is (per Network) unique key.
Layerlike = Union[LayerBase, str]

def layer_key(layer: Layerlike) -> str:
    """The layer key for a given `Layerlike` object.
    """
    return layer if isinstance(layer, str) else layer.key

def as_layer(layer: Layerlike, network: NetworkBase) -> LayerBase:
    """The layer object for `Layerlike` object.
    """
    return network[layer] if isinstance(layer, str) else layer
