"""An engine accessing network activations.
"""

# standard imports
from typing import Union, List
import logging

# third party imports
import numpy as np

# toolbox imports
from network import Network, Classifier, ShapeAdaptor, ResizePolicy
from network.layers import Layer
from ..base.data import Data
from . import Tool, Worker

# logging
LOG = logging.getLogger(__name__)


# FIXME[todo]: this is essentially a wraper around Network.
# Check if we could make the Network itself an ActivationTool
class ActivationTool(Tool, Network.Observer):
    """.. :py:class:: Activation

    The :py:class:`Activation` class encompassing network, current
    activations, and the like.

    An :py:class:`Activation` tool is :py:class:`Observable`. Changes
    in the :py:class:`Activation` that may affect the computation of
    activation are passed to observers (e.g. workers) calling the
    :py:meth:`Observer.activation_changed` method in order to inform
    them as to the exact nature of the model's change.

    **Changes**

    network_changed:
        The underlying :py:class:`network.Network` has changed,
        or its preparation state was altered. The new network
        can be accessed vie the :py:attr:`network` property.
    layer_changed:
        The selected set of layers was changed.

    Attributes
    ----------

    _network: Network
        Currently active network

    _layers: List[Layer]
        the layers of interest

    _classification: bool
        If True, the model will consider the current model
        as a classifier and record the output of the output layer
        in addition to the current (hidden) layer.

    """

    def __init__(self, network: Network = None, **kwargs):
        """Create a new ``Engine`` instance.

        Parameters
        ----------
        network: Network
            Network providing activation values.
        """
        super().__init__(**kwargs)

        # adapters
        self._shape_adaptor = ShapeAdaptor(ResizePolicy.Bilinear())
        self._channel_adaptor = ShapeAdaptor(ResizePolicy.Channels())
        self._channel_axis = 'channels_first'

        # network related
        self._network = None
        self.network = network

    #
    # network
    #

    def network_changed(self, _network: Network, info: Network.Change) -> None:
        """React to changes of the :py:class:`Network`.
        The :py:class:`ActivationTool` is interested when the
        network becomes prepared (or unprepared). We just forward
        these notifications.
        """
        LOG.debug("Activation.network_changed(%s)", info)
        if info.state_changed:
            self.change('state_changed')

    @property
    def network(self) -> Network:
        """Get the currently selected network.

        Returns
        -------
        The currently selected network or None if no network
        is selected.
        """
        return self._network

    @network.setter
    def network(self, network: Network) -> None:
        if network is self._network:
            return  # nothing changed

        if self._network is not None:
            self.unobserve(self._network)
        self._network = network
        if network is not None:
            interests = Network.Change('state_changed')
            self.observe(network, interests)
            # FIXME[old]: what is this supposed to do?
            if network.prepared and self._shape_adaptor is not None:
                self._shape_adaptor.setNetwork(network)
                self._channel_adaptor.setNetwork(network)
        self.change('tool_changed')

    #
    # Tool interface
    #

    external_result = ('activations', )
    internal_arguments = ('inputs', 'layer_ids')
    internal_result = ('activations_list', )

    def _preprocess(self, inputs: np.ndarray, layer_ids: List[Layer] = None,
                    **kwargs) -> Data:
        # pylint: disable=arguments-differ
        data = super()._preprocess(**kwargs)
        array = inputs.array if isinstance(inputs, Data) else inputs
        data.add_attribute('inputs', array)
        unlist = False
        if layer_ids is None:
            layer_ids = list(self._network.layer_dict.keys())
        elif not isinstance(layer_ids, list):
            layer_ids, unlist = [layer_ids], True
        data.add_attribute('layer_ids', layer_ids)
        data.add_attribute('unlist', unlist)
        return data

    def _process(self, inputs: np.ndarray,
                 layers: List[Layer]) -> List[np.ndarray]:
        # pylint: disable=arguments-differ

        LOG.info("computing activations for data <%s>, layers=%s",
                 inputs.shape, layers)

        if self._network is None:
            return None

        if not layers:
            return layers

        return self._network.get_activations(inputs, layers)

    def _postprocess(self, data: Data, what: str) -> None:
        if what == 'activations':
            activations_dict = \
                dict(zip(data.layer_ids, map(lambda activations: activations,
                                             data.activations_list)))
            data.add_attribute(what, activations_dict)
        else:
            super()._postprocess(data, what)

    def data_activations(self, data: Data, layer: Layer = None,
                         unit: int = None) -> np.ndarray:
        """Get the precomputed activation values for the current
        :py:class:`Data`.
        """
        activations = self.get_data_attribute(data, 'activations')
        if layer is None:
            return activations
        if isinstance(layer, Layer):
            layer = layer.id
        activations = activations[layer]
        if unit is None:
            return activations
        if self._network.channel_axis == Network.CHANNEL_AXIS_FIRST:
            return activations[unit]
        return activations[..., unit]


class ActivationWorker(Worker):
    """A :py:class:`Worker` specialized to work with the
    :py:class:`ActivationTool`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._layer_ids = []
        self._fixed_layers = []
        self._classification = False

    def _ready(self) -> bool:
        # FIXME[hack]
        return (super()._ready() and
                self._tool.network is not None and
                self._tool.network.prepared)

    def set_network(self, network: Network,
                    layers: List[Layer] = None) -> None:
        """Set the current network. Update will only be published if
        not already selected.

        Parameters
        ----------
        network : str or int or network.network.Network
            Key for the network
        """
        LOG.info("Engine.set_network(%s): old=%s", network, self._network)
        if network is not None and not isinstance(network, Network):
            raise TypeError("Expecting a Network, "
                            f"not {type(network)} ({network})")

        if self._tool is None:
            raise RuntimeError("Trying to set a network "
                               "without having a Tool.")

        self._tool.network = network

        # set the layers (this will also trigger the computation
        # of the activations)
        self.set_layers(layers)
        self.change(network_changed=True)

    def set_layers(self, layers: List[Layer]) -> None:
        """Set the layers for which activations shall be computed.

        """
        self._fixed_layers = layers
        self._update_layers()

    def add_layer(self, layer: Union[str, Layer]) -> None:
        """Add a layer to the list of activation layers.
        """
        if isinstance(layer, str):
            self._fixed_layers.append(self.network[layer])
        elif isinstance(layer, Layer):
            self._fixed_layers.append(layer)
        else:
            raise TypeError("Invalid type for argument layer: {type(layer)}")
        self._update_layers()

    def remove_layer(self, layer: Layer) -> None:
        """Remove a layer from the list of activation layers.
        """
        self._fixed_layers.remove(layer)
        self._update_layers()

    def set_classification(self, classification: bool = True) -> None:
        """Record the classification results.  This assumes that the network
        is a classifier and the results are provided in the last
        layer.
        """
        if classification != self._classification:
            self._classification = classification
            self._update_layers()

    def _update_layers(self) -> None:

        # Determining layers
        layer_ids = list(map(lambda layer: layer.id, self._fixed_layers))
        if self._classification and isinstance(self._network, Classifier):
            class_scores_id = self._network.scores.id
            if class_scores_id not in layer_ids:
                layer_ids.append(class_scores_id)

        got_new_layers = layer_ids > self._layer_ids and self._data is not None
        self._layer_ids = layer_ids
        if got_new_layers:
            self.work(self._data)

    def _apply_tool(self, data: Data, **kwargs) -> None:
        self.tool.apply(self, data, layers=self._layer_ids, **kwargs)

    def activations(self, layer: Layer = None, unit: int = None) -> np.ndarray:
        """Get the precomputed activation values for the current
        :py:class:`Data`.
        """
        activations = self._tool.data_activations(self._data, layer, unit)
        print("activations:", len(activations) if layer is None else activations.shape)
        return self._tool.data_activations(self._data, layer, unit)
