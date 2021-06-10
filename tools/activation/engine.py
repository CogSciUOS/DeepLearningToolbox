
# =============================================================================


# FIXME[old]
class OldEngine(Worker, Toolbox.Observer, method='activation_changed',
             changes={'network_changed', 'input_changed',
                      'activation_changed'}):
    # pylint: disable=too-many-ancestors, too-many-instance-attributes
    """.. :py:class:: Engine

    The :py:class:`Engine` class encompassing network, current
    activations, and the like.

    An :py:class:`Engine` is :py:class:`Observable`. Changes in the
    :py:class:`Engine` are passed to observers calling the
    :py:meth:`Observer.activation_changed` method in order to inform
    them as to the exact nature of the model's change.

    **Changes**

    network_changed : bool
        The underlying :py:class:`network.Network` has changed,
        or its preparation state was altered. The new network
        can be accessed vie the :py:attr:`network` property.
    input_changed : bool
        The input data changed. The new data can be accessed
        as :py:attr:`data`.
    activation_changed: bool
        The network activation changed. This usually coincides
        with a change in input data, but may occur delayed in case
        of complex computation and multi-threading.

    Attributes
    ----------
    _data: np.ndarray
        The currently processed data. The data can be set by
        :py:meth:`set_data`. This will notify observers on
        `input_changed` and start the actual computation of
        the activation values.  Data will be annotated with
        `<network>_resized`, `<network>_preprocessed` and
        `<network>_layer` attributes.

    _layers: List[layer_ids]
        the layers of interest

    _classification: bool
        If True, the model will consider the current model
        as a classifier and record the output of the output layer
        in addition to the current (hidden) layer.

    _network: Network
        Currently active network

    _toolbox: Toolbox
        FIXME[question]: is this really needed?

    _activations: Dict[layer_ids, np.ndarray]
        Mapping layer_ids to the current activation values.
    """
    
    def __init__(self, network: Network = None, toolbox: Toolbox = None):
        """Create a new ``Engine`` instance.

        Parameters
        ----------
        network :   Network
                    Network instance backing the model
        """
        super().__init__()

        # data related
        self._data = None
        self._shape_adaptor = ShapeAdaptor(ResizePolicy.Bilinear())
        self._channel_adaptor = ShapeAdaptor(ResizePolicy.Channels())

        # network related
        self._network = None
        self._layers = None
        self._classification = False
        self.set_network(network)

        # toolbox
        self._toolbox = None
        self.set_toolbox(toolbox)

    @classmethod
    def observable_name(cls) -> str:
        """The :py:class:`activation.Engine` will be known as
        `'ActivationEngine'`.
        """
        return 'ActivationEngine'

    ##########################################################################
    #                              TOOLBOX                                   #
    ##########################################################################

    def set_toolbox(self, toolbox: Toolbox) -> None:
        """Set the toolbox for this activation Engine. The Engine will observe
        the Toolbox for 'input_changed' message and automatically
        compute new activation values whenever the Toolbox' input data
        changes.

        Parameters
        ----------
        toolbox: Toolbox
            The toolbox to use. If None, no toolbox will be used.
        """
        if self._toolbox is toolbox:
            return  # nothing changed

        if self._toolbox is not None:
            self.unobserve(self._toolbox)

        self._toolbox = toolbox
        self.data = None if toolbox is None else toolbox.input_data

        if toolbox is not None:
            interests = Toolbox.Change('input_changed')
            self.observe(toolbox, interests=interests)

    def toolbox_changed(self, toolbox: Toolbox, info: Toolbox.Change) -> None:
        """React to toolbox changes.
        The :py:class:`Engine` is only interested in new input data.
        """
        LOG.debug("tools.activation.Engine.toolbox_changed(%s)", info)
        if info.input_changed:
            self.work(toolbox.input_data)

    ##########################################################################
    #                          SETTING DATA                                  #
    ##########################################################################

    @property
    def data(self) -> Data:
        """The data currently used by this activation :py:class:`Engine`.
        """
        return self._data

    @data.setter
    def data(self, data: Data) -> None:
        """Set the data to be used by this activation :py:class:`Engine`.
        This will trigger processing that data. Observers will by a
        `activation_changed` notification, once processsing has
        finished and activation data have been added to that
        :py:class:`Data` object.
        """
        LOG.debug("tools.activation.Engine.data = %s", data)
        self.work(data)

    ##########################################################################
    #                     SETTING THE NETWORK                                #
    ##########################################################################

    def network_changed(self, network: Network, info: Network.Change) -> None:
        """React to changes of the :py:class:`Network`.
        The activation :py:class:`Engine` is interested when the
        network becomes prepared (or unprepared).
        """
        LOG.debug("tools.activation.Engine.network_changed(%s)", info)
        if info.state_changed:
            if network.prepared == (self._layers is None):
                self._prepare_network()

    def _prepare_network(self, layers: List[Layer] = None) -> None:
        LOG.info("Engine._prepare_network: %s (%s vs %s)",
                 None if self._network is None else self._network.key,
                 None if self._network is None else self._network.prepared,
                 self._layers is None)
        if self._network is None or not self._network.prepared:
            self._layers = None
            return

        self._layers = layers or []

        if self._shape_adaptor is not None:
            self._shape_adaptor.setNetwork(self._network)
            self._channel_adaptor.setNetwork(self._network)

        self._update_activations()

    def set_network(self, network: Network, layers: List[Layer] = None) -> None:
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

        if network is self._network:
            return  # nothing changed

        if self._network is not None:
            self.unobserve(self._network)
        if network is not None:
            interests = Network.Change('state_changed')
            self.observe(network, interests=interests)

        self._network = network

        # Prepare the network (this will also trigger the computation
        # of the activations)
        self._prepare_network(layers)
        self.change(network_changed=True)

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

    def add_layer(self, layer: Union[str, Layer]) -> None:
        """Add a layer to the list of activation layers.
        """
        if isinstance(layer, str):
            self._layers.append(self.network[layer])
        elif isinstance(layer, Layer):
            self._layers.append(layer)
        else:
            raise TypeError("Invalid type for argument layer: {type(layer)}")
        self._update_activations()

    def remove_layer(self, layer: Layer) -> None:
        """Remove a layer from the list of activation layers.
        """
        self._layers.remove(layer)

    def set_classification(self, classification: bool = True) -> None:
        """Record the classification results.  This assumes that the network
        is a classifier and the results are provided in the last
        layer.
        """
        if classification != self._classification:
            self._classification = classification
            self._update_activation()

    # --------------------------------------------------------------------------

    #
    # Activation and processor
    #

    def get_activation(self, layer: Union[Layer, str],
                       unit: int = None, batch_index: int = 0) -> np.ndarray:
        """Get the activation map for the specified layer.
        """
        layer_id = layer.key if isinstance(layer, Layer) else layer
        LOG.debug("Engine.get_activation(layer_id=%s, batch_index=%s)",
                  layer_id, batch_index)

        # FIXME[todo/concept]: we need a good concept to specify what
        # are valid layers and which layer activations shall be computed.
        # We have:
        #  - the list of layers from the Network
        #  - the current list of self._layers (plus self._classification)
        #     for which activations are to be computed
        #  - the list of already computed activations in the self._data object
        #
        # if not layer_id in ...:
        #    raise ValueError(f"Invalid layer ID '{layer_id}'. Valid IDs are: "
        #                     f"{', '.join(self.layer_ids)}")

        data = self.data
        if data is None:
            LOG.debug("Engine.get_activation: no data")
            return None

        network = self._network
        if network is None:
            LOG.debug("Engine.get_activation: no network")
            return None

        attribute = network.key + '_' + layer_id
        if not data.has_attribute(attribute):
            LOG.debug("Engine.get_activation: no activation values")
            return None

        activations = getattr(data, attribute)
        LOG.debug("Engine.get_activation: %s", activations.shape)

        if unit is not None:
            # FIXME[todo]: we have to check for channel first/last ...
            activations = activations[..., unit]
            #activations = activations[unit]
        return activations

    def _update_activations(self):
        if self._data is not None:
            self.work(self._data)

    def _process_data(self, data: Data) -> None:
        if self._layers is None:
            return  # nothing to do

        network = self._network
        if network is None:
            return  # cannot work without network

        # Preprocessing
        prefix = self.network.key + '_'
        self._preprocess_data(data, prefix)
        preprocessed = getattr(data, prefix + 'preprocessed')
        LOG.info("preprocessing data with prefix '%s': %s -> %s", prefix,
                 data.array.shape, preprocessed.shape)
        self._data = data
        self.change(input_changed=True)

        # Determining layers
        layer_ids = []
        for layer in self._layers:
            layer_id = layer.key
            attribute = prefix + layer_id
            if not data.has_attribute(attribute):
                layer_ids.append(layer_id)
        if self._classification and isinstance(network, Classifier):
            class_scores_id = network.scores.id
            if class_scores_id not in layer_ids:
                layer_ids.append(class_scores_id)

        LOG.info("computing activations for data <%s>, layers=%s",
                 data, layer_ids)
        activations = self._network.get_activations(preprocessed, layer_ids)
        # activations will be a List[np.ndarray]

        # FIXME[hack]: should be done in network!
        for i, activation in enumerate(activations):
            if activation.ndim in {2, 4}:
                if activation.shape[0] != 1:
                    raise RuntimeError('Attempting to compute '
                                       'activation for a batch '
                                       'of images.')
                activations[i] = np.squeeze(activation, axis=0)

        for layer_id, activation in zip(layer_ids, activations):
            attribute = prefix + layer_id
            data.add_attribute(attribute, activation, batch=True)

        # Notify observers that activation values are available
        # FIXME[problem]: why does change not work here?
        # self.change(activation_changed=True)
        self.notify_observers(activation_changed=True)

    def _preprocess_data(self, data: Data, prefix: str = '') -> None:
        """Preprocess the given data.
        This will add two attributes to the data object:
        `reshaped` will be a version of the original data in
        a shape suitable to be fed to the network and
        `preprocessed` will apply additional preprocessing steps
        (like normalization).
        """
        if data is None:
            return  # nothing to do

        values = data.array
        values = self._shape_adaptor(values)
        values = self._channel_adaptor(values)
        data.add_attribute(prefix + 'reshaped', values)

        if values.shape[-1] == 3:  # RGB image
            if issubclass(values.dtype.type, np.integer):
                values = values.astype(np.float32) / 256.0

            # FIXME[hack]: all pretrained torch models use this
            # mean = [0.485, 0.456, 0.406]
            # std=[0.229, 0.224, 0.225]
            # normalize = transforms.Normalize(mean=mean, std=std),
            # this does: data |--> (data - mean) / std
            mean = np.asarray([0.485, 0.456, 0.406])
            std = np.asarray([0.229, 0.224, 0.225])
            values = (values - mean) / std
            # FIXME[todo]: this may be integrated into the Network ...

        data.add_attribute(prefix + 'preprocessed', values)
