from typing import Dict, Iterable

import numpy as np

from network import Network, ShapeAdaptor, ResizePolicy
from network.layers import Layer
from base.observer import Observable, change
from util import async

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Engine(Observable, method='modelChanged',
             changes=['network_changed',
                      'layer_changed',
                      'unit_changed',
                      'input_changed',
                      'activation_changed']):
    """.. :py:class:: Engine

    Engine class encompassing network, current activations, and the like.

    An model is Observable. Changes in the model are passed to observers
    calling the :py:meth:`Observer.modelChanged` method in order
    to inform them as to the exact nature of the model's change.

    Changes
    -------
    network_changed : bool
        Whether the underlying :py:class:`network.Network` has changed
    layer_changed : bool
        Whether the current :py:class:`network.layers.Layer` has changed
    unit_changed : bool
        Whether the selected unit changed
    input_changed : bool
        Whether the input signal changed
    activation_changed: bool
        Whether the network activation changed. This usually coincides
        with a change in input data, but may occur be delayed in case
        of complex computation and multi-threading.

    Attributes
    ----------
    _input: np.ndarray
        Current input data, suitable for the current network
        (this is an adapted version of _data)
    _current_activation: np.ndarray
        The last computed activations
    _layer: Layer
        Currently selected layer
    _classification: bool
        If True, the model will consider the current model
        as a classifier and record the output of the output layer
        in addition to the current (hidden) layer.
    _unit: int
        Currently selected unit in the layer
    _network: Network
        Currently active network
    _networks: Dict[str, Network]
        All available networks. FIXME[todo]: Move this out of the model
    _data: np.ndarray
        Current data provided by the data source
    _data_target: int
        A target value (i.e., label) for the current _data as provided by
        the data source, None if no such information is provided.
        If set, this integer value will indicate the index of the
        correct target unit in the classification layer (usually the
        last layer of the network).
    _data_description: str

    New:
    _layers: List[layer_ids]
        the layers of interest
    _activations: Dict[]
        mapping layer_ids to activations
    """

    def __init__(self, network: Network=None):
        """Create a new ``Engine`` instance.

        Parameters
        ----------
        network :   Network
                    Network instance backing the model
        """
        super().__init__()

        #
        # data related
        #
        self._data = None
        self._input = None
        self._data_target = None
        self._data_description = None
        self._shape_adaptor = ShapeAdaptor(ResizePolicy.Bilinear())
        self._channel_adaptor = ShapeAdaptor(ResizePolicy.Channels())

        #
        # network related
        #
        self._network = None
        self._layer = None
        self._unit = None
        self._classification = None
        self._current_activation = None
        self._layers = []
        self._activations = {}

        # FIXME[hack]: should be set from the outside, depending on
        # whether there is someone using the classification information!
        self._classification = True

        if network is not None:
            self.set_network(network)

    ##########################################################################
    #                          SETTING DATA                                  #
    ##########################################################################

    def get_input_data(self, raw: bool=False) -> np.ndarray:
        """Obtain the current input data.  This is the current data in a
        format suitable to be fed to the current network.

        Parameters
        ----------
        raw   :   bool
            If true, the method will return the raw data (as it was
            provided by the input source). Otherwise it will provide
            data in a format suitable for the current network.

        Returns
        -------
        np.ndarray

        """
        return self._data if raw else self._input

    @change
    def set_input_data(self, data: np.ndarray, target: int=None,
                       description: str = None):
        """Provide one data vector as input for the network.
        The input data must have 2, 3, or 4 dimensions.

        - 2 dimensions means a single gray value image
        - 3 dimensions means a single three-channel image. The channels will
          be repeated thrice to form a three-dimensional input
        - 4 dimensions are only supported insofar as the shape must be
          ``(1, A, B, C)``, meaning the fist dimension is singular and can be
          dropped. Actual batches are not supported.

        The input data may be adapted to match the input shape of
        the current network. Different adaptation strategies may be
        applied, which are provided by setting a
        :py:class::ShapeAdaptor for this Engine.

        Parameters
        ----------
        data: np.ndarray
            The data array
        target: int
            The data label. None if no label is available.
        description: str
            A description of the input data.
        """
        logger.info(f"Activation: set_input_data({data.shape},{target},{description})")
        #
        # do some sanity checks and corrections
        #
        if data is None or not data.ndim:
            raise ValueError('Data cannot be None.')

        if data.ndim > 4 or data.ndim < 2:
            raise ValueError(f'Data must have between 2 '
                             'and 4 dimensions (has {data.ndim}).')

        if data.ndim == 4:
            if data.shape[0] == 1:
                # first dimension has size of 1 -> remove
                data = data.squeeze(0)
            else:
                raise ValueError('Cannot visualize batch of images')

        #
        # set the data
        #
        self._data = data
        self._data_target = target
        self._data_description = description

        #
        # adapt the data to match the network input shape
        #
        self._update_input()
        self.change(input_changed=True)

        #
        # recompute the network activations
        #
        self._update_activation()

    @property
    def input_data(self):
        return self._input

    @input_data.setter
    def input_data(self, data):
        target = None
        description = None
        if isinstance(data, tuple):
            for d in data[1:]:
                if isinstance(d, int):
                    label = d
                elif isinstance(d, str):
                    description = d
            data = data[0]
        self.set_input_data(data, target, description)

    @property
    def raw_input_data(self):
        return self._data

    @property
    def input_data_description(self):
        return self._data_description

    def _update_input(self) -> None:
        """Update the input data from the current raw data.
        """
        data = self._data
        if data is not None:
            data = self._shape_adaptor(data)
            data = self._channel_adaptor(data)
        self._input = data

    ##########################################################################
    #                     SETTING THE NETWORK                                #
    ##########################################################################

    @change
    def set_network(self, network: Network) -> None:
        """Set the current network. Update will only be published if
        not already selected.

        Parameters
        ----------
        network : str or int or network.network.Network
            Key for the network
        """
        if not isinstance(network, Network):
            raise ValueError("Expecting a Network, "
                             f"not {type(network)} ({network})")
        print(f"Activation: network is now {network}")

        if self._network != network:
            self._network = network

            if self._shape_adaptor is not None:
                self._shape_adaptor.setNetwork(network)
                self._channel_adaptor.setNetwork(network)
                self._update_input()

            self.change(network_changed=True)

            # Finally unset the layer (this will also trigger
            # a computation of the activations)
            self.layer = None

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
        self.set_network(network)

    # FIXME[old]: check if this is still needed.
    # make a clear concept of layer, layer_id and layer_index!
    def idForLayer(self, layer_id: str) -> int:
        """Obtain the numeric id for a given layer identifier.
        .. note:: This operation is linear in the number of layers
        in the current network.

        Parameters
        ----------
        layer_id : str
            Identifier of the layer

        Returns
        -------
        int
            layer index

        Raises
        ------
        ValueError:
            The given layer_id does not identify a Layer in the
            current model.
        """
        if layer_id is None:
            return None
        try:
            layer_keys = list(self._network.layer_dict.keys())
            return layer_keys.index(layer_id)
        except ValueError:
            raise ValueError(f"Layer for string '{layer_id}' not found."
                             f" valid keys are: {layer_keys}"
                             f", current layer is '{self._layer}'"
                             f"/'{self.layer}'")

    @change
    def set_layer(self, layer: Layer):
        """Set the current layer to choose units from.

        Parameters
        ----------
        layer   :   Layer
                    Layer instance to display
        """
        # FIXME[hack]:
        if layer == "":
            layer = None
        if layer == "":
            raise ValueError("layer_id should not be '', but rather None")

        if self._layer != layer:
            self._unit = None
            self._layer = layer
            self.change(layer_changed=True, unit_changed=True)

            # FIXME[concept]: reconsider the update logic!
            #  should updating the layer_list automatically update the
            #  activation? or may there be another update to the layer list?
            if self._input is not None and layer is not None:
                self._update_activation()

    @property
    def layer_id(self):
        """Get the id of the currently selected network layer.

        Returns
        -------
        The currently selected layer id. None if no layer is selected.
        """
        return self._layer

    @layer_id.setter
    def layer_id(self, layer: Layer):
        self.set_layer(layer)

    @change
    def set_unit(self, unit: int):
        """Change the currently visualised channel/unit.

        Parameters
        ----------
        unit: int
            Index of the unit in the layer (0-based)
        """
        if self._layer is None:
            unit = None
        elif unit is not None:
            layer_shape = self._network.get_layer_output_shape(self._layer)
            if unit < 0 or unit >= layer_shape[-1]:
                unit = None
        if unit != self._unit:
            self._unit = unit
            self.change(unit_changed=True)

    @property
    def unit(self) -> int:
        """The currently selected unit.

        Result
        ------
        unit: int
            Index of the unit in the layer (0-based).
        """
        return self._unit

    @unit.setter
    def unit(self, unit: int):
        self.set_unit(unit)

    def set_classification(self, classication: bool=True):
        """Record the classification results.  This assumes that the network
        is a classifier and the results are provided in the last
        layer.
        """
        old_classification = self._classification
        self._classification = classication
        if old_classification != self._classification:
            self._update_activation()

    def _update_layer_list(self):
        if self._network is None:
            self._layers = []
        else:
            layers = set()
            if self._layer is not None:
                layers.add(self._layer)
            if self._classification and self._network.is_classifier():
                layers.add(self._network.output_layer_id())
            self._layers = list(layers)

    #@async
    @change
    def _update_activation(self):
        """Set the :py:attr:`_current_activation` property by loading
        activations for :py:attr:`_layer` and :py:attr:`_data`.
        This is a noop if no layers are selected or no data is
        set."""

        self._update_layer_list()
        logger.info(f"Activation: _update_activation: LAYERS={self._layers}")
        if self._layers and self._input is not None:
            layers = list(self._layers)

            # compute the activations for the layers of interest
            activations = self._network.get_activations(layers,
                                                        self._input)

            # FIXME[hack]: should be done in network!
            for i in range(len(activations)):
                if activations[i].ndim in {2, 4}:
                    if activations[i].shape[0] != 1:
                        raise RuntimeError('Attempting to visualise batch.')
                    activations[i] = np.squeeze(activations[i], axis=0)

            self._activations = {id: activations[i]
                                 for i, id in enumerate(layers)}

            # FIXME[debug]: if we work we multiple Threads, we have to
            # care for synchroization!
            if layers != self._layers:
                logger.info(f"Activation: update_activation(): "
                      "LAYERS CHANGED DURING UPDATE "
                      "{layers} vs. {self._layers}")

            # FIXME[old]
            self._current_activation = self._activations.get(self._layer, None)
        else:
            self._activations = {}
        self.change(activation_changed=True)

    ##########################################################################
    #                             UTILITIES                                  #
    ##########################################################################

    # FIXME[design]: some redundancy with Network.classify_top_n()
    def top_n_classifications(self, n=5, labels: bool=True):
        """Get the network's top n classification results for the
        current input. The results will be sorted with highest ranked
        class first.

        Parameters
        ----------
        n:
            The number of results to report.
        labels:
            If True, return class labels (str) instead of class indices.

        Returns
        -------
        classes:
            The classes, either indices (int) or labels (str).
        scores:
            The corresponding class scores, i.e., the output value
            of the network for that class.
        target:
            The target class, i.e. the "correct" answer. None if unknown.
        """
        #
        # determine target class
        #
        target = self._data_target
        if labels and target is not None:
            if self._network is not None:
                target = self._network.get_label_for_class(target)
            else:
                target = str(target)

        #
        # some sanity checks
        #
        no_result = (None, None, target)
        if not self._network:
            return no_result  # no network available

        if not self._classification:
            return no_result  # computation of classification is turned off

        if not self._network.is_classifier():
            return no_result  # network is not a classifier

        classification_layer_id = self._network.output_layer_id()
        if classification_layer_id not in self._activations:
            return no_result  # no classifiction values available

        #
        # compute the top n class scores
        #
        class_scores = self._activations[classification_layer_id]

        # Remark: here we could use np.argsort(-class_scores)[:n]
        # but that may be slow for a large number classes,
        # as it does a full sort. The numpy.partition provides a faster,
        # though somewhat more complicated method.
        top_n_indices_unsorted = np.argpartition(-class_scores, n)[:n]
        order = np.argsort((-class_scores)[top_n_indices_unsorted])
        top_n_indices = top_n_indices_unsorted[order]
        if labels:
            top_n = [self._network.get_label_for_class(i)
                     for i in top_n_indices]
        else:
            top_n = top_n_indices

        return top_n, class_scores[top_n_indices], target


# FIXME[old]
class OldModel:
    def __init__(self, network: Network=None):
        self._networks = {}

    # FIXME[hack]: add_network is seemingly not called via a Controller,
    # hence we have to explicitly notify the observers
    @change
    def add_network(self, network: Network, select: bool=True) -> None:
        """Add a model to visualise. This will add the network to the list of
        choices and make it the currently selcted one.

        Parameters
        ----------
        network : Network
            A network.
        select : bool
            A flag indicating whether the new network should
            automatically be selected as the active network in this
            model.
        """
        if network is not None:
            self._networks[network.get_id()] = network
        if select:
            self.network = network
        self.change(network_changed=True)

    @property
    def networks(self) -> Iterable[Network]:
        """Get the currently selected network.

        Returns
        -------
        The currently selected network or None if no network
        is selected.
        """
        return self._networks.values()

    def network_by_id(self, network_id: str) -> Network:
        """Get a network by an unique identifier.
        The network has to be registered with this Model
        (via :py:meth:`add_network`) before it can be retrieved
        by this method.
        """
        return self._networks.get(network_id)
   
