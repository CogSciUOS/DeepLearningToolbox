from toolbox import Toolbox
from datasources import Datasource

from typing import Dict, Iterable

import numpy as np

from network import Network, ShapeAdaptor, ResizePolicy
from network.layers import Layer
from base.observer import Observable, change
from util import async

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Engine(Observable, Toolbox.Observer, method='activation_changed',
             changes=['network_changed',
                      'layer_changed',
                      'unit_changed',
                      'input_changed',
                      'activation_changed']):
    """.. :py:class:: Engine

    Engine class encompassing network, current activations, and the like.

    An model is Observable. Changes in the model are passed to observers
    calling the :py:meth:`Observer.activation_changed` method in order
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
    _data: np.ndarray
        Current data provided by the data source
    _data_description: str

    New:
    _layers: List[layer_ids]
        the layers of interest
    _activations: Dict[layer_ids, np.ndarray]
        Mapping layer_ids to the current activation values.
    """

    _toolbox: Toolbox = None  # FIXME[question]: is this really needed?
    _network: Network = None
    
    def __init__(self, network: Network=None, toolbox: Toolbox=None):
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
        self._data_label = None
        self._data_description = None
        self._datasource = None
        self._shape_adaptor = ShapeAdaptor(ResizePolicy.Bilinear())
        self._channel_adaptor = ShapeAdaptor(ResizePolicy.Channels())

        #
        # network related
        #
        self._network = None
        self._layer = None
        self._unit = None
        self._classification = None
        self._layers = []
        self._activations = {}

        # FIXME[hack]: should be set from the outside, depending on
        # whether there is someone using the classification information!
        self._classification = True

        self.set_toolbox(toolbox)
        if network is not None:
            self.set_network(network)

    def set_toolbox(self, toolbox: Toolbox) -> None:
        if toolbox:
            interests = Toolbox.Change('input_changed')
            self.observe(toolbox, interests=interests)

    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        if change.input_changed:
            self.set_input_data(data=toolbox.input_data,
                                label=toolbox.input_label,
                                datasource=toolbox.input_datasource,
                                description=toolbox.input_description)
            
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

    #@change
    def set_input_data(self, data: np.ndarray, label: int=None,
                       datasource: Datasource=None,
                       description: str=None) -> None:
        """Provide one data vector as input for the network.
        The input data must have 2, 3, or 4 dimensions.

        - 2 dimensions means a single gray value image
        - 3 dimensions means a single three-channel image. The channels will
          be repeated thrice to form a three-dimensional input
        - 4 dimensions are only supported insofar as the shape must be
          ``(1, A, B, C)``, meaning the fist dimension is singular and can be
          dropped. Actual batches are not supported (yet).

        The input data may be adapted to match the input shape of
        the current network. Different adaptation strategies may be
        applied, which are provided by setting a
        :py:class::ShapeAdaptor for this Engine.

        Parameters
        ----------
        data: np.ndarray
            The data array
        label: int
            The data label. None if no label is available.
        description: str
            A description of the input data.
        """
        logger.info(f"Activation: set_input_data({data is not None and data.shape},{label},{description})")
        #
        # do some sanity checks and corrections
        #
        if data is None or not data.ndim:
            return
            #raise ValueError('Data cannot be None.')

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
        self._data_label = label
        self._datasource = datasource
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
        label = None
        description = None
        if isinstance(data, tuple):
            for d in data[1:]:
                if isinstance(d, int):
                    label = d
                elif isinstance(d, str):
                    description = d
            data = data[0]
        self.set_input_data(data, label, description)

    @property
    def raw_input_data(self):
        return self._data

    @property
    def input_label(self):
        return self._data_label

    @property
    def input_datasource(self):
        return self._datasource

    @property
    def input_data_description(self):
        return self._data_description

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
        if network is not None and not isinstance(network, Network):
            raise ValueError("Expecting a Network, "
                             f"not {type(network)} ({network})")

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
            print(f"Engine.set_layer(): layer changed: {self._layer} / {self._unit}")
            # FIXME[problem]: why does change does not work here?
            #self.change(layer_changed=True, unit_changed=True)
            self.notifyObservers(layer_changed=True, unit_changed=True)
            print(f"Engine.set_layer(): observers were notified ... {self._input is not None}, {layer}")


            # FIXME[concept]: reconsider the update logic!
            #  should updating the layer_list automatically update the
            #  activation? or may there be another update to the layer list?
            if self._input is not None and layer is not None:
                print(f"Engine.set_layer(): now updating activations ...")
                self._update_activation()
                print(f"Engine.set_layer(): ... updating activations finished")

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
            # FIXME[problem]: why does change does not work here?
            #self.change(unit_changed=True)
            print(f"activation.Engine.set_unit({unit}) - notifyObservers")
            self.print_observers()
            self.notifyObservers(unit_changed=True)

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
        if classification != self._classification:
            self._classification = classication
            self._update_activation()

    def _update_layer_list(self):
        """Update the (internal) list of layers for which activations
        should be computed.
        """
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
    #@change
    def _update_activation(self):
        """Set the :py:attr:`_activations` property by loading
        activations for :py:attr:`_layer` and :py:attr:`_data`.
        This is a noop if no layers are selected or no data is set.
        """
        if not self._network or self._network.busy:
            return  # FIXME[hack]
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

        else:
            self._activations = {}

        # FIXME[problem]: why does change does not work here?
        #self.change(activation_changed=True)
        self.notifyObservers(activation_changed=True)

    def get_activation(self, layer_id=None, batch_index: int=0) -> np.ndarray:
        if self._activations is None or not self._activations:
            return None
        if layer_id is None:
            layer_id = self._layer
            if layer_id is None:
                return None
        if not layer_id in self._activations:
            raise ValueError(f"Invalid layer ID '{layer_id}'. Valid IDs are: "
                             f"{', '.join(self._activations.keys())}")
        return self._activations[layer_id]


    ##########################################################################
    #                             UTILITIES                                  #
    ##########################################################################

    # FIXME[design]: some redundancy with Network.classify_top_n()
    def top_n_classifications(self, n=5) -> (np.ndarray, np.ndarray):
        """Get the network's top n classification results for the
        current input. The results will be sorted with highest ranked
        class first.

        Parameters
        ----------
        n: int
            The number of results to report.

        Returns
        -------
        classes:
            The class indices.
        scores:
            The corresponding class scores, i.e., the output value
            of the network for that class.
        """

        #
        # some sanity checks
        #
        no_result = (None, None)
        if not self._network:
            return no_result  # no network available

        if not self._network.is_classifier():
            return no_result  # no network available
            # FIXME[toddo]: raise RuntimeError("Network is not a classifier.")

        classification_layer_id = self._network.output_layer_id()
        if classification_layer_id not in self._activations:
            return no_result  # no classifiction values available
            # FIXME[toddo]: raise RuntimeError("No activation values have been computed.")

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

        return top_n_indices, class_scores[top_n_indices]

    def classification_rank(self, label: int) -> (int, float):
        #
        # some sanity checks
        #
        no_result = (None, None)
        if not self._network:
            return no_result  # no network available

        if not self._network.is_classifier():
            return no_result  # no network available
            # FIXME[toddo]: raise RuntimeError("Network is not a classifier.")

        classification_layer_id = self._network.output_layer_id()
        if classification_layer_id not in self._activations:
            return no_result  # no classifiction values available
            # FIXME[toddo]: raise RuntimeError("No activation values have been computed.")
        class_scores = self._activations[classification_layer_id]
        score = class_scores[label]
        rank = (class_scores > score).sum()

        return rank, score



from network.layers import StridingLayer

class MaximalActivations:
    """Collect input images that maximally activate given units.
    """

    def __init__(self, network):
        self._shape_adaptor = ShapeAdaptor(ResizePolicy.Bilinear())
        self._channel_adaptor = ShapeAdaptor(ResizePolicy.Channels())

        self._network = network

        self._layer_ids = []
        for layer_id, layer in self._network.layer_dict.items():
            print(layer_id, layer.get_id())
            self._layer_ids.append(layer_id)      

            self._network.get_receptive_field(layer_id, (2,2))

        self._shape_adaptor.setNetwork(network)
        self._channel_adaptor.setNetwork(network)

        self._initialize(self._layer_ids)

    def _initialize(self, layers, top_n=3):
        self._top_n = top_n
        self._max_activations = {}
        self._min_index = {}
        self._max_coordinates = {}
        
        for layer_id in layers:
            layer = self._network.layer_dict[layer_id]
            if isinstance(layer, StridingLayer):
                dims = 3
                #features = layer.filters
                features = layer.output_shape[-1]
            else:
                dims = 1
                features = layer.output_shape[-1]

            print(f"Layer {layer} ({layer.__class__.__name__}) has {dims} dimensions and {features} features.")

            self._max_activations[layer_id] = np.zeros((features, top_n))
            self._min_index[layer_id] = \
                self._max_activations[layer_id].argmin(axis=1)
            self._max_coordinates[layer_id] = np.zeros((features, top_n, dims),
                                                       dtype=np.uint32)

    def compute_activations(self, input, input_id):
        if not self._network or self._network.busy:
            return  # FIXME[hack]

        if input is None:
            return  # FIXME[hack]

        old_shape = input.shape
        input = self._shape_adaptor(input)
        input = self._channel_adaptor(input)
        print(f"Input {input_id}: shape {old_shape} -> {input.shape}")


        # compute the activations for the layers of interest
        activations = self._network.get_activations(self._layer_ids, input)

        # FIXME[hack]: should be done in network!
        for i in range(len(activations)):
            if activations[i].ndim in {2, 4}:
                if activations[i].shape[0] != 1:
                    raise RuntimeError('Attempting to visualise batch.')
                activations[i] = np.squeeze(activations[i], axis=0)

        self._activations = {id: activations[i]
                             for i, id in enumerate(self._layer_ids)}

        for layer_id in self._layer_ids:
            self.process_activations(self._activations[layer_id], layer_id, input_id)

    def process_activations(self, activations, layer_id, input_id):
        if isinstance(self._network.layer_dict[layer_id], StridingLayer):
            # FIXME[hack]: just consider the highest activation
            # (there may be more high activations in the activation
            # map, but they will be ignored for the sake of simplicity)
            channels = activations.shape[-1]
            a = activations.reshape(-1, channels)
            max_idx = a.argmax(0)
            coords = np.column_stack(np.unravel_index(max_idx,
                                                      activations.shape[:-1]))
            activations = a[max_idx, np.arange(a.shape[1])]

        features = activations.shape[0]
        min_values = self._max_activations[layer_id][np.arange(features),
                                                     self._min_index[layer_id]]
        units, = np.where(activations > min_values)
        print(f"{input_id}: {layer_id}: {len(units)}/{features}")
        for unit in units:
            self.update_max_activations(activations[unit], layer_id,
                                        unit, input_id)


    def update_max_activations(self, activation, layer_id, unit,
                               input_id, coordinate=None) -> None:
        min_index = self._min_index[layer_id][unit]
        self._max_activations[layer_id][unit, min_index] = activation
        self._min_index[layer_id][unit] = \
            self._max_activations[layer_id][unit].argmin()

