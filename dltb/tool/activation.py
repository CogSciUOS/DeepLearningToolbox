"""An engine accessing network activations.
"""

# standard imports
from typing import Union, List
import math
import logging

# third party imports
import numpy as np

# toolbox imports
from network import Network, Classifier, ShapeAdaptor, ResizePolicy
from network.layers import Layer
from ..datasource import Datasource, Datafetcher
from ..base.data import Data
from ..util.array import adapt_data_format, DATA_FORMAT_CHANNELS_FIRST
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

    Data processing
    ---------------

    The :py:class`ActivationTool` can be applied to `Data` objects. It
    will pass the `Data` object as argument to the underlying
    :py:class:`Network` and will store results as attributes in
    the `Data` object. It will use the following attributes:

    [tool_name]_activations:
        A dictionary mapping layer names to (numpy) arrays of activation
        values.
    """

    def __init__(self, network: Network = None, data_format: str = None,
                 **kwargs) -> None:
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
        self._data_format = data_format

        # network related
        self._network = None
        self.network = network

    @property
    def data_format(self) -> str:
        """
        """
        if self._data_format is not None:
            return self._data_format
        if self._network is not None:
            return self._network.data_format
        return None

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
        # FIXME[todo]: inputs should probably be Datalike
        """Preprocess the arguments and construct a Data object.
        """
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
        """Perform the actual operation, that is the computation of
        activation values for given input values.

        Arguments
        ---------
        inputs:
            Input data.
        layers:
            A list of layers for which to compute activations.
        """

        print(f"ActivationTool._process({type(inputs)}, {type(layers)})")
        # LOG.info("ActivationTool: computing activations for data <%s>, "
        #          "layers=%s, activation format=%s",
        #          inputs.shape, layers, self.data_format)

        if self._network is None:
            return None

        if not layers:
            return layers

        return self._network.get_activations(inputs, layers,
                                             data_format=self.data_format)

    def _postprocess(self, data: Data, what: str) -> None:
        if what == 'activations':
            activations_dict = dict(zip(data.layer_ids, data.activations_list))
            data.add_attribute(what, activations_dict)
            data.add_attribute('activations_dict', activations_dict)
        else:
            super()._postprocess(data, what)

    def data_activations(self, data: Data, layer: Layer = None,
                         unit: int = None,
                         data_format: str = None) -> np.ndarray:
        """Get the precomputed activation values for the current
        :py:class:`Data`.

        Arguments
        ---------
        data:
            The :py:class:`Data` object in which precomputed activations
            are stored.
        layer:
            The layer for which activation values should be obtained.
        unit:
            The unit for which activation values should be obtained.
        data_format:
            The data format (channel first or channel last) in which
            activation values should be returned.  If `None` (default),
            the default format of this :py:class:`ActivationTool`
            (according to :py:prop:`data_format`) is used.

        Result
        ------
        activations:
            The requested activation valeus. The type depends on the
            arguments:
            If no `layer` is specified, the result will be a
            dictionary mapping layer names (`str`) activation values
            (`numpy.ndarray`).
            If a `layer` is specified, the activation values (np.ndarray)
            of that layer are returned.
            If in addtion to `layer` also a `unit` is specified, only
            the activation value(s) for that unit are returned.
        """
        # FIXME[todo]: batch processing - add an 'index' argument ...

        # activations: dict[layer: str, activation_values: np.ndarray]
        activations = self.get_data_attribute(data, 'activations')

        if data_format is None:
            data_format = self.data_format
        elif unit is not None:
            LOG.warning("Providing a data_format (%s) has no effect "
                        "when querying unit activation", data_format)

        if layer is None:  # return the full dictionary
            if data_format is self.data_format:
                return activations  # no tranformation required

            # transform the data format of the activation values
            return list(map(lambda activation:
                            adapt_data_format(activation, input_format=
                                              self._data_format,
                                              output_format=data_format),
                            activations))
        if isinstance(layer, Layer):
            layer = layer.id
        activations = activations[layer]

        if data_format is not self.data_format:
            activations = adapt_data_format(activations,
                                            input_format=self.data_format,
                                            output_format=data_format)

        if unit is None:
            return activations

        return (activations[unit] if data_format == DATA_FORMAT_CHANNELS_FIRST
                else activations[..., unit])


class ActivationWorker(Worker):
    """A :py:class:`Worker` specialized to work with the
    :py:class:`ActivationTool`.

    layers:
        The layers for which activations shall be computed.

    data: (inherited from Worker)
        The current input data
    activations: dict
        The activations for the current data

    top_indices: Mapping[Layer, np.ndarray]
        A dictionary mapping layers to an array holding the indices
        of data points for the top activations.
    top_activations : Mapping[Layer, np.ndarray]
        A dictionary mapping layers to an array holding the top activation
        values.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._layer_ids = []
        self._fixed_layers = []
        self._classification = False

        self._activations = None
        self.top_indices = {}
        self.top_activations = {}

    #
    # Tool core functions
    #

    def _apply_tool(self, data: Data, **kwargs) -> None:
        """Apply the :py:class:`ActivationTool` on the given data.
        """
        self.tool.apply(self, data, layers=self._layer_ids, **kwargs)

    def activations(self, layer: Layer = None, unit: int = None,
                    data_format: str = None) -> np.ndarray:
        """Get the precomputed activation values for the current
        :py:class:`Data`.
        """
        activations = \
            self._tool.data_activations(self._data, layer=layer, unit=unit,
                                        data_format=data_format)
        LOG.debug("ActivationWorker.activations(%s,unit=%s,data_format=%s):"
                  " %s", layer, unit, data_format,
                  len(activations) if layer is None else activations.shape)
        return activations

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

    #
    # Layer configuration
    #

    def set_layers(self, layers: List[Layer]) -> None:
        """Set the layers for which activations shall be computed.

        """
        self._fixed_layers = \
            layers if isinstance(layers, list) else list(layers)
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

    #
    # work on Datasource
    #

    def extract_activations(self, datasource: Datasource,
                            batch_size: int = 128) -> None:
        samples = len(datasource)
        # Here we could:
        #  np.memmap(filename, dtype='float32', mode='w+',
        #            shape=(samples,) + network[layer].output_shape[1:])
        results = {
            layer: np.ndarray((samples,) +
                              self.tool.network[layer].output_shape[1:])
            for layer in self._layer_ids
        }

        fetcher = Datafetcher(datasource, batch_size=batch_size)

        try:
            index = 0
            for batch in fetcher:
                print("dl-activation: "
                      f"processing batch of length {len(batch)} "
                      f"with elements given as {type(batch.array)}, "
                      f"first element having index {batch[0].index} and "
                      f"shape {batch[0].array.shape} [{batch[0].array.dtype}]")
                # self.work() will make `batch` the current data object
                # of this Worker (self._data) and store activation values
                # as attributes of that data object:
                self.work(batch, busy_async=False)

                # obtain the activation values from the current data object
                activations = self.activations()

                # print(type(activations), len(activations))
                print("dl-activation: activations are of type "
                      f"{type(activations)} of length {len(activations)}")
                if isinstance(activations, dict):
                    for index, (layer, values) in \
                            enumerate(activations.items()):
                        print(f"dl-activation:  [{index}]: {values.shape}")
                        results[layer][index:index+len(batch)] = values
                elif isinstance(activations, list):
                    print("dl-activation: "
                          f"first element is {type(activations[0])} "
                          f"with shape {activations[0].shape} "
                          f"[{activations[0].dtype}]")
                    for index, values in enumerate(activations):
                        print(f"dl-activation:  [{index}]: {values.shape}")
                        layer = self._layer_ids[index]
                    results[layer][index:index+len(batch)] = values
                print("dl-activation: batch finished in "
                      f"{self.tool.duration(self._data)*1000:.0f} ms.")
        except KeyboardInterrupt:
            # print(f"error procesing {data.filename} {data.shape}")
            print("Keyboard interrupt")
            # self.output_status(top, end='\n')
        except InterruptedError:
            print("Interrupted.")
        finally:
            print("dl-activation: finished processing")
            # signal.signal(signal.SIGINT, original_sigint_handler)
            # signal.signal(signal.SIGQUIT, original_sigquit_handler)

    def iterate_activations(self, datasource: Datasource,
                            batch_size: int = 128):  # -> Iterator

        fetcher = Datafetcher(datasource, batch_size=batch_size)

        index = 0
        for data in fetcher:
            print("iterate_activations: "
                  f"processing {'batch' if data.is_batch else 'data'}")
            self.work(data, busy_async=False)
            activations = self.activations()
            if data.is_batch:
                for index, view in enumerate(data):
                    yield {layer: activations[layer][index]
                           for layer in activations}
            else:
                yield activations

    #
    # top activations
    #

    def merge_top_activations(self, top: int = None):
        for layer in self.layers:
            self._merge_layer_top_activations(layer, top)

    def _merge_layer_top_activations(self, layer: Layer, top: int = None):
        # channel last (batch, height, width, channel)
        new_activations = \
            self.activations(layer).reshape(-1, self.actviations.shape[-1])

        batch_len = len(new_activations)
        data_len = batch_len // self.actviations.shape[0]
        start_index = self.index_batch_start * data_len

        # activations has shape (batch, classes)
        batch = np.arange(batch_len)
        if top is None:
            top_indices = np.argmax(new_activations, axis=-1)
        else:
            # Remark: here we could use np.argsort(-class_scores)[:n]
            # but that may be slow for a large number classes,
            # as it does a full sort. The numpy.partition provides a faster,
            # though somewhat more complicated method.
            top_indices_unsorted = \
                np.argpartition(-new_activations, top)[batch, :top]
            order = \
                np.argsort((-new_activations)[batch, top_indices_unsorted.T].T)
            new_top_indices = top_indices_unsorted[batch, order.T].T

        if not start_index:
            self.top_indices[layer] = new_top_indices
            self.top_activations[layer] = new_activations[top_indices]
        else:
            merged_indices = np.append(self.top_indices[layer],
                                       new_top_indices + start_index)
            merged_activations = np.append(self.top_activations[layer],
                                           new_activations[top_indices])

            sort = np.argsort(merged_activations)
            self.top_indices[layer] = merged_indices[:sort]
            self.top_activations[layer] = merged_activations[:sort]

    def top_activations(self, activations: np.ndarray, top: int = 9,
                        datasource_index: int = None) -> None:
        """

        Arguments
        ---------
        activations

        Result
        ------
        top_activations:
            This is an array of shape (top, channels)
        top_indices:
            This is an array of shape (top, 2, channels).
            [n,0,channel] is the index of the datapoint in the datasource,
            while [n,1,channel] is the (1-dimensional) index in the
            activation map. This second index may have to be unraveled
            to obtain real activation map coordinates.
        """
        # remember the original shape
        shape = activations.shape

        # flatten activations per channel
        # ([batch,] position, channel) -> (indices, channel)
        activations = np.reshape(activations, (-1, shape[-1]))

        # get indices for top activations per channel, shape: (top, channels)
        # Remark: here we could use np.argsort(-class_scores)[:n]
        # but that may be slow for a large number classes,
        # as it does a full sort. The numpy.partition provides a faster,
        # though somewhat more complicated method.
        top_indices_unsorted = \
            np.argpartition(-activations, top, axis=0)[:top]

        # got correspondig (unsorted) top activations: shape (top, channels)
        top_activations = \
            activations[np.arange(top), top_indices_unsorted.T].T

        if isinstance(datasource_index, np.ndarray):
            # only for batches:
            batch_shape = (shape[0], math.prod(shape[1:-1]))
            batch_indices, position_indices = \
                np.unravel_index(top_indices_unsorted, batch_shape)
            datasource_indices = datasource_index[batch_indices]
            top_indices = np.append(datasource_indices[:, np.newaxis],
                                    position_indices[:, np.newaxis], axis=1)
        else:
            position_indices = top_indices_unsorted[:, np.newaxis]
            datasource_indices = \
                np.full(position_indices.shape, datasource_index, np.int)
            # shape: (top, 2, channels)
            top_indices = \
                np.append(datasource_indices, position_indices, axis=1)

        return top_activations, top_indices

    def _merge_top_activations(self, top_activations: np.ndarray,
                               top_indices: np.ndarray,
                               new_activations: np.ndarray,
                               new_indices: np.ndarray) -> None:
        """Merge activation values into top-n highscore. Both activation data
        consists of two arrays, the first (top_activations) the
        holding the actual activation values and the second
        (top_indices) holding the corresponding indices of the top
        scores.

        Arguments
        ---------
        top_activations:
            activation values of shape (top, channels)

        top_indices:
            corresponding indices in dataset / position of shape
            (top, 2, channels)

        new_activations:
            activation values of shape (top, channels)
        new_indices:
            corresponding indices in dataset / position of shape
            (top, 2, channels)

        """
        top = len(top_activations)
        merged_indices = np.append(top_indices, new_indices)
        merged_activations = np.append(top_activations, new_activations)
        sort = np.argsort(-merged_activations, axis=0)
        top_indices[:] = merged_indices[sort[:top]]
        top_activations[:] = merged_activations[sort[:top]]

    def _init_layer_top_activations(self, layers = None, top: int = 9) -> None:
        if layers is None:
            layers = self._fixed_layers
        for layer in layers:
            self._top_activations[layer] = \
                np.full((layer.filters, layer.filters), -np.inf)
            # index: (datasource index, fiter index)
            self._top_indices[layer] = \
                np.full((layer.filters, 2, layer.filters),
                        np.nan, dtype=np.int)

    def _update_layer_top_activations(self, layers = None,
                                      top: int = 9) -> None:
        if layers is None:
            layers = self._fixed_layers
        for layer in layers:
            top_activations, top_indices = \
                self.top_activations(self.activations(layer),
                                     datasource_index=self._data.index)
            self._merge_top_activations(self._top_activations[layer],
                                        self._top_indices[layer],
                                        top_activations, top_indices)

