from toolbox import Toolbox
from datasources import Datasource

from typing import Dict, Iterable, Tuple

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
        Whether the selected unit changed. This also includes a change
        of the position in the activation map (if applicable).
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
    _position: Tuple[int, ...]
        Currently selected position in the activation map of the
        selected unit if applicable, otherwise (0,0) or None if
        no position is selected.
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
    _processed_input: np.ndarray
        The last input that was processed. This will usually be the same
        as the _input, but it may be different in concurrent scenarios,
        where the input may change while activations are computed.
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

        self._processed_input = None

        #
        # network related
        #
        self._network = None
        self._layer = None
        self._unit = None
        self._position = None
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
        """Set the toolbox for this activation Engine. The Engine will observe
        the Toolbox for 'input_changed' message and automatically
        compute new activation values whenever the Toolbox' input data
        changes.

        Parameter
        ---------
        toolbox: Toolbox
            The toolbox to use. If None, no toolbox will be used.
        """
        if toolbox:
            interests = Toolbox.Change('input_changed')
            self.observe(toolbox, interests=interests)
            # FIXME[bug?]: why is the following necessary?
            # Shouldn't there be an automatic notification when
            # calling observe() ...?
            self.set_input_data(data=toolbox.input_data,
                                label=toolbox.input_label,
                                datasource=toolbox.input_datasource,
                                description=toolbox.input_description)

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
        """Provide one data vector as input for this Engine.  The new data
        will be fed to the network to compute activation values.
        
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
            The data array. None means that this Engine will unset all
            activation values.
        label: int
            The data label. None if no label is available.
        description: str
            A description of the input data.

        """
        logger.info(f"Activation: set_input_data({data is not None and data.shape},{label},{description})")
        #
        # do some sanity checks and corrections
        #
        if data is not None and data.ndim:
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
        # FIXME[problem]: why does change does not work here?
        # self.change(input_changed=True)
        self.notifyObservers(input_changed=True)
            
        #
        # recompute the network activations
        #
        self._update_activation()

    @property
    def input_data(self):
        """The preprocessed input data, that is the data in the form fed to
        the Network. None means that there are no input data.
        """
        return self._input

    @input_data.setter
    def input_data(self, data):
        """Set the current input data. This is roughly equivalent to calling
        :py:meth:set_input_data().

        The assigned data can be either some raw data array (in which
        case label and description are set to None), or a tuple
        containing the data array, followed by label and description.
        """
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
        """The raw (unprocessed) input data as assigned to this Engine by
        :py:meth:set_input_data(). None means that there are no input
        data.
        """
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
            self._data_reshaped = data

            if data.shape[-1] == 3:
                if issubclass(data.dtype.type, np.integer):
                    data = data.astype(np.float32)
                    data -= np.asarray([0.485, 0.456, 0.406])*256
                else:
                    # FIXME[hack]: all pretrained torch models use this
                    # mean = [0.485, 0.456, 0.406]
                    # std=[0.229, 0.224, 0.225]
                    # normalize = transforms.Normalize(mean=mean, std=std),
                    # this does: data |--> (data - mean) / std
                    mean = np.asarray([0.485, 0.456, 0.406])
                    std = np.asarray([0.229, 0.224, 0.225])
                    data = (data - mean) / std
            # FIXME[todo]: this may be integrated into the Network ...
            
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
            raise TypeError("Expecting a Network, "
                            f"not {type(network)} ({network})")

        if self._network != network:
            self._network = network
            self._unit = None
            self._position = None
            self._layer = None

            if self._shape_adaptor is not None:
                self._shape_adaptor.setNetwork(network)
                self._channel_adaptor.setNetwork(network)
                self._update_input()

            # FIXME[problem]: why does change does not work here?
            # self.change(network_changed=True,
            #             layer_changed=True, unit_changed=True)
            self.notifyObservers(network_changed=True,
                                 layer_changed=True, unit_changed=True)

            # Finally we will trigger the computation of the activations
            self._update_activation(force=True)

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
            self._position = None
            self._layer = layer

            # FIXME[problem]: why does change does not work here?
            #self.change(layer_changed=True, unit_changed=True)
            self.notifyObservers(layer_changed=True, unit_changed=True)

            # FIXME[concept]: reconsider the update logic!
            #  should updating the layer_list automatically update the
            #  activation? or may there be another update to the layer list?
            self._update_activation(force=True)

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
    def set_unit(self, unit: int, position: Tuple[int, ...]=None) -> None:
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
        if unit != self._unit or position != self._position:
            self._unit = unit
            self._position = position
            # FIXME[problem]: why does change does not work here?
            #self.change(unit_changed=True)
            #self.print_observers()
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

    @property
    def position(self) -> Tuple[int, ...]:
        """The position in the activation map of the currently selected unit.

        Result
        ------
        position: Tuple[int, int]
            Index of the unit in the layer (0-based).
        """
        return self._position

    @property
    def receptive_field(self) -> (Tuple[int, ...], Tuple[int, ...]):
        if self._layer is None or self._position is None:
            return None
        
        layer = self._network.layer_dict[self._layer]
        return layer.receptive_field(self._position)

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
    def _update_activation(self, force: bool=False):
        """Set the :py:attr:`_activations` property by loading
        activations for :py:attr:`_layer` and :py:attr:`_data`.
        This is a noop if no layers are selected or no data is set.
        """
        if force:
            self._processed_input = None
        
        if self._network is not None and self._network.busy:
            # Another Thread seems to be active.
            # FIXME[hack]: this will work, if the other Thread is also
            # running self._update_activation() and is in the loop below
            # but not if it runs anything else. The clean way would be
            # to wait until the network is not busy anymore ...
            return

        while self._processed_input is not self._input:
            self._processed_input = self._input

            self._update_layer_list()
            logger.info("Activation: _update_activation: "
                        f"LAYERS={self._layers}")
            if self._layers and self._input is not None:
                layers = self._layers

                # compute the activations for the layers of interest
                activations = \
                    self._network.get_activations(layers,
                                                  self._processed_input)

                # FIXME[hack]: should be done in network!
                for i in range(len(activations)):
                    if activations[i].ndim in {2, 4}:
                        if activations[i].shape[0] != 1:
                            raise RuntimeError('Attempting to compute '
                                               'activation for a batch '
                                               'of images.')
                        activations[i] = np.squeeze(activations[i], axis=0)

                self._activations = {id: activations[i]
                                     for i, id in enumerate(layers)}

                # FIXME[debug]: if we work with multiple Threads, we have to
                # care for synchroization!
                if layers != self._layers:
                    logger.info(f"Activation: update_activation(): "
                                "LAYERS CHANGED DURING UPDATE "
                                "{layers} vs. {self._layers}")
                    self._processed_input = None
            else:
                self._activations = {}

            # FIXME[problem]: why does change not work here?
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

