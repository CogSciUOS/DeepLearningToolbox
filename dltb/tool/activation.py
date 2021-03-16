"""An engine accessing network activations.
"""

# standard imports
from typing import Union, List, Tuple, Iterable
from pathlib import Path
import os
import json
import logging
# import math
# import functools
# import operator

# third party imports
import numpy as np

# toolbox imports
from network import Network, Classifier, ShapeAdaptor, ResizePolicy
from network.layers import Layer
from ..datasource import Datasource, Datafetcher
from ..base.prepare import Preparable
from ..base.data import Data
from ..util.array import adapt_data_format, DATA_FORMAT_CHANNELS_FIRST
from ..util import nphelper
from ..config import config
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


class PersistentTool(Preparable):
    """
    Arguments
    ---------
    network:

    datasource:

    mode:
        The mode of operation: `r` opens the archive for reading and
        `w` for writing. In case of `w`, the archive will use existing
        files if they exist or create new ones otherwise.

    """
    # FIXME[hack]: put this in the config file ...
    config.activations_directory = Path('/space/home/ulf/activations')

    # _datasource:
    #    The Datasource for which activation values are computed
    _datasource: Union[str, Datasource] = None

    # _network:
    #    The Network by which activation values are obtained.
    _network: Union[str, Network] = None

    # _layers:
    #    The keys of the network layers for which activation values are
    #    stored in the ActivationsArchive
    _layers: List[str] = None  # sequence of layers

    def __init__(self, network: Union[str, Network],
                 datasource: Union[str, Datasource],
                 layers = None, mode: str = 'r', **kwargs) -> None:
        super().__init__(**kwargs)
        self._network = network
        self._datasource = datasource
        self._layers = layers and [layer.key if isinstance(layer, Layer)
                                   else layer for layer in self._layers]
        self._mode = mode
        self._directory = Path(config.activations_directory) /\
            ((network.key if isinstance(network, Network)
              else network) + '-' +
             (datasource.key if isinstance(datasource, Datasource)
              else datasource))
        LOG.info("ActivationsArchiveNumpy at '%s' initalized.",
                 self._directory)

    @property
    def directory(self) -> Path:
        """The name of the directory into which this
        :py:class:`ActivationsArchiveNumpy` is stored on disk.
        """
        return self._directory

    def __len__(self) -> int:
        """The length of the archive. This is the :py:prop:`valid` size if the
        archive is opened in read mode and the :py:prop:`total` size
        if opened in write mode.

        """
        return self.valid if self._mode == 'r' else self.total

    @property
    def valid(self) -> int:
        """The valid size of this archive. May be less than
        :py:prop:`total` if the archive has not been fully filled yet.
        """
        return None if self._meta is None else self._meta['valid']

    @property
    def total(self) -> int:
        """The total size of this archive. May be more than the
        :py:prop:`valid` size if the archive has not been fully filled yet.
        """
        return None if self._meta is None else self._meta['total']

    def _preparable(self) -> bool:
        if self._mode == 'r' and not self._directory.is_dir():
            return False
        if self._mode == 'w' and not isinstance(self._network, Network):
            return False
        return super()._preparable()

    def _prepared(self) -> bool:
        return self._meta is not None and super()._prepared()

    def flush(self) -> None:
        if self._mode != 'w':
            raise ValueError(f"TopActivations in mode '{self._mode}' "
                             "is not writable")
        if self._meta is not None:
            with open(self.filename_meta, 'w') as outfile:
                json.dump(self._meta, outfile)


class TopActivations(PersistentTool):
    """The :py:class:`TopActivations` stores the top activation values
    for the layers of a :py:class:`Network`.

    top: int
        The number of activation layers to store
    layers: Sequence[Layer]
        The layers for which top activation values are stored.
    top_indices: Mapping[Layer, np.ndarray]
        A dictionary mapping layers to an array holding the indices
        of data points for the top activations.
        The arrays have a shape of (channels, top, indices), with indices
        being the number of indices necessary to index an activation
        map in the layer: for a dense layer, this is 1 (batch, )
        while for 2D layers this is 3 (batch, row, column).
    top_activations : Mapping[Layer, np.ndarray]
        A dictionary mapping layers to an array holding the top activation
        values. The arrays have the shape (channels, top),
        with channels being the number of channels in the layer
        and top the number of top activation values to be stored.
    """

    @staticmethod
    def _merge_top(target_indices, target_values,
                   new_indices, new_values) -> None:
        # indices: shape = (channels, top, indices)
        # values: shape = (channels, top)
        top = target_values.shape[1]
        indices = np.append(target_indices, new_indices, axis=1)
        values = np.append(target_values, new_values, axis=1)

        # top_indices: shape = (channels, top)
        top_indices = nphelper.argtop(values, top, axis=1)
        target_values[:] = np.take_along_axis(values, top_indices, axis=1)
        # FIXME[bug]: ValueError: `indices` and `arr` must have the
        # same number of dimensions
        # target_indices[:] = np.take_along_axis(indices, top_indices, axis=1)
        for coordinate in range(target_indices.shape[-1]):
            target_indices[:, :, coordinate] = \
                np.take_along_axis(indices[:, :, coordinate],
                                   top_indices, axis=1)

    def __init__(self, top: int = 9, **kwargs) -> None:
        super().__init__(**kwargs)
        self._top = top
        self._meta = None
        self._top_indices = None
        self._top_activations = None

    @property
    def filename_meta(self) -> Path:
        """The name of the file holding the meta data for this
        :py:class:`ActivationsArchiveNumpy`.
        """
        return self._directory / f'top-{self._top}.json'

    def filename_top(self, layer: str) -> Path:
        return self._directory / f'top-{self._top}-{layer}.npy'

    @property
    def top_(self) -> int:
        """The number of top values to record per layer/channel
        """
        return self._top

    def layers(self, *what) -> Iterable[Tuple[str, type, Tuple[int]]]:
        """Iterate layer of layer information.

        Arguments
        ---------
        what:
            Specifies the what information should be provided. Valid
            values are: `'name'` the layer name,
            `'layer'` the actual layer object (only available if the
            :py:class:`Network`, not just the network key, has been provided
             upon initialization of this :py:class:`ActinvationsArchive`).
        """
        if not what:
            what = ('name', )
        elif 'layer' in what and not isinstance(self._network, Network):
            raise ValueError("Iterating over Layers is only possible with "
                             "an initialized Network.")
        for layer in self._layers:
            name = layer.key if isinstance(layer, Layer) else layer
            values = tuple((name if info == 'name' else
                            self._network[name] if info == 'layer' else '?')
                           for info in what)
            yield values[0] if len(what) == 1 else values

    def _prepare(self) -> None:
        super()._prepare()
        filename_meta = self.filename_meta
        self._top_indices = {}
        self._top_activations = {}
        if filename_meta.exists():
            with open(filename_meta, 'r') as file:
                meta = json.load(file)
            for name in self.layers('name'):
                with self.filename_top(name).open('rb') as file:
                    self._top_indices[name] = np.load(file)
                    self._top_activations[name] = np.load(file)
        else:  # mode == 'w' and isinstance(network, Network):
            length = len(self._datasource)
            meta = {
                'total': length,
                'valid': 0,
                'shape': {}
            }
            if self._layers is None:
                self._layers = list(self._network.layer_names())
            for name, layer in self.layers('name', 'layer'):
                channels = layer.output_shape[-1]
                indices = len(layer.output_shape) - 1  # -1 for the channel
                # indices: (channels, top, indices)
                self._top_indices[name] = \
                    np.full((channels, self._top, indices), -1, np.int)
                # activations: (channels, top)
                self._top_activations[name] = \
                    np.full((channels, self._top), np.NINF, np.float32)
            meta['layers'] = self._layers
        self._meta = meta

    def _unprepare(self) -> None:
        # make sure all information is stored
        if self._mode == 'w':
            self.flush()
        self._meta = None
        self._top_indices = None
        self._top_activations = None
        LOG.info("TopActivations at '%s' unprepared.",
                 self._directory)
        super()._unprepare()

    def flush(self) -> None:
        super().flush()
        if self._top_indices is not None:
            for name in self.layers('name'):
                with self.filename_top(name).open('wb') as file:
                    np.save(file, self._top_indices[name])
                    np.save(file, self._top_activations[name])

    def __iadd__(self, values) -> object:
        index = np.asarray([self._meta['valid']], dtype=np.int)

        if isinstance(values, dict):
            for layer, layer_values in values.items():
                self._update_values(layer, layer_values[np.newaxis], index)
        elif isinstance(values, list):
            if len(values) != len(self._layers):
                raise ValueError("Values should be a list of length"
                                 f"{len(self._layers)} not {len(values)}!")
            for layer, layer_values in zip(self._layers, values):
                self._update_values(layer, layer_values[np.newaxis], index)
        else:
            raise ValueError("Values should be a list (of "
                             f"length {len(self._layers)}) "
                             f"or a dictionary, not {type(values)}")
        self._meta['valid'] += 1
        return self

    def _update_values(self, layer: str, value: np.ndarray,
                       index: np.ndarray = None) -> None:
        """Update the top activation lists with new values.

        Arguments
        ---------
        layer:
            The layer for which top activation values should be
            updated.
        value:
            The activation map. This is expected to be of shape
            (batch, position..., channel).
        index:
            The index of the activation value in the
            :py:class:`Datasource`-
        """
        layer, name = self._network[layer], layer

        # slim_values have shape (batch*position..., channel)
        slim_values = value.reshape((-1, value.shape[-1]))

        # top_slim: array of shape (top, channel),
        #   containing the indices in the value array for the top elements
        #   for each channel (i.e. values from 0 to len(slim_values))
        top = min(self._top, len(slim_values))
        top_slim = nphelper.argtop(slim_values, top=top, axis=0)

        # top_activations: (channel, top)
        top_activations = np.take_along_axis(slim_values, top_slim, axis=0).T

        # the index shape is (batch, positions...), without channel
        shape = (len(value), ) + layer.output_shape[1:-1]

        # top_indices have index as (batch, position..., channel)
        #  indices * (top, channel)
        #   -> (indices, top, channel)
        #   -> (channel, top, indices)
        top_indices = np.stack(np.unravel_index(top_slim, shape)).T

        # adapt the batch index
        if index is not None:
            top_indices[:, :, 0] = index[top_indices[:, :, 0]]

        self._merge_top(self._top_indices[name], self._top_activations[name],
                        top_indices, top_activations)

    def activations(self, layer, channel=...) -> np.ndarray:
        return self._top_activations[layer][channel]

    def indices(self, layer, channel=...) -> np.ndarray:
        return self._top_indices[layer][channel]

    def receptive_field(self, layer, channel, top=0) -> np.ndarray:
        indices = self.indices(layer, channel)[top]
        image = self._datasource[indices[0]]
        return self._network.\
            extract_receptive_field(layer, indices[1:-1], image)

    def info(self) -> None:
        """Output a summary of this :py:class:`ActivationsArchiveNumpy`.
        """
        print(f"Archive at {self.directory}: {self.valid}/{self.total}")

    def old_merge_layer_top_activations(self, layer: Layer, top: int = None):
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
            self._top_indices[layer] = new_top_indices
            self._top_activations[layer] = new_activations[top_indices]
        else:
            merged_indices = np.append(self._top_indices[layer],
                                       new_top_indices + start_index)
            merged_activations = np.append(self._top_activations[layer],
                                           new_activations[top_indices])

            sort = np.argsort(merged_activations)
            self._top_indices[layer] = merged_indices[:sort]
            self._top_activations[layer] = merged_activations[:sort]


    def old_top_activations(self, activations: np.ndarray, top: int = 9,
                        datasource_index: int = None) -> None:
        """Get the top activattion values and their indices in a
        batch of activation maps.

        Arguments
        ---------
        activations:
            A batch of activation maps of shape
            (batch, position..., channels).
        top:
            The number of top values to extract.
        datasource_index:

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
        # ([batch,] position..., channel) -> (indices, channel)
        activations = np.reshape(activations, (-1, shape[-1]))

        # get indices for top activations per channel, shape: (top, channels)
        # Remark: here we could use np.argsort(-class_scores)[:n]
        # but that may be slow for a large number classes,
        # as it does a full sort. The numpy.partition provides a faster,
        # though somewhat more complicated method.
        top_indices_unsorted = \
            np.argpartition(-activations, top, axis=0)[:top]

        # get correspondig (unsorted) top activations: shape (top, channels)
        top_activations = \
            activations[np.arange(top), top_indices_unsorted.T].T

        if isinstance(datasource_index, np.ndarray):
            # working on a batch:
            # math.prod ist only available from 3.8 onward ...
            # batch_shape = (shape[0], math.prod(shape[1:-1]))
            batch_shape = (shape[0], np.prod(shape[1:-1]))
            # batch_shape = \
            #     (shape[0], functools.reduce(operator.mul, shape[1:-1]))
            batch_indices, position_indices = \
                np.unravel_index(top_indices_unsorted, batch_shape)
            datasource_indices = datasource_index[batch_indices]
            top_indices = np.append(datasource_indices[:, np.newaxis],
                                    position_indices[:, np.newaxis], axis=1)
        else:
            # working on activations for a single input:
            position_indices = top_indices_unsorted[:, np.newaxis]
            datasource_indices = \
                np.full(position_indices.shape, datasource_index, np.int)
            # shape: (top, 2, channels)
            top_indices = \
                np.append(datasource_indices, position_indices, axis=1)

        return top_activations, top_indices

    def old_merge_top_activations(self, top_activations: np.ndarray,
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

    def old_init_layer_top_activations(self, layers = None, top: int = 9) -> None:
        if layers is None:
            layers = self._fixed_layers
        for layer in layers:
            self._top_activations[layer] = \
                np.full((layer.filters, layer.filters), -np.inf)
            # index: (datasource index, fiter index)
            self._top_indices[layer] = \
                np.full((layer.filters, 2, layer.filters),
                        np.nan, dtype=np.int)

    def old_update_layer_top_activations(self, layers = None,
                                         top: int = 9) -> None:
        if layers is None:
            layers = self._fixed_layers
        for layer in layers:
            top_activations, top_indices = \
                self._top_activations(self.activations(layer),
                                      datasource_index=self._data.index)
            self._merge_top_activations(self._top_activations[layer],
                                        self._top_indices[layer],
                                        top_activations, top_indices)


class ActivationWorker(Worker):
    """A :py:class:`Worker` specialized to work with the
    :py:class:`ActivationTool`.

    layers:
        The layers for which activations shall be computed.

    data: (inherited from Worker)
        The current input data
    activations: dict
        The activations for the current data

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._layer_ids = []
        self._fixed_layers = []
        self._classification = False

        self._activations = None

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



class ActivationsArchive(Preparable):
    """An :py:class:`ActinvationsArchive` represents an archive of
    activation values obtained by applying a :py:class:`Network` to
    a :py:class:`Datsource`.


    Intended use:

    >>> with ActivationsArchiveNumpy(network, datasource, mode='w') as archive:
    >>>     for index in range(archive.valid, archive.total):
    >>>         archive += network.get_activations(datasource[index])

    >>> with ActivationsArchiveNumpy(network, datasource, mode='w') as archive:
    >>>     for batch in datasource.batches(batch_size=128):
    >>>         activation_tool.process(batch)
    >>>         archive += batch

    >>> with ActivationsArchiveNumpy(network, datasource, mode='w') as archive:
    >>>     archive.fill()
    """
    # FIXME[todo]:
    # - allow ActivationTool instead of Network
    # - make this class an ActivationTool
    # - documentation


class ActivationsArchiveNumpy(ActivationsArchive, PersistentTool):
    """The :py:class:`ActivationsArchiveNumpy` realizes an
    :py:class:`ActivationsArchive` based on the Numpy `memmap`
    mechanism.

    All files of the :py:class:`ActivationsArchiveNumpy` are stored in
    the directory :py:prop:`directory`. Each layer gets a separate
    file, called `[LAYER_NAME].dat`. Metadata for the archive are
    stored in JSON format into the file `meta.json`.

    The total size of an :py:class:`ActivationsArchiveNumpy`, that is
    the number of data points for which activations are stored in the
    archive, has to be provided uppon initialization and cannot be
    changed afterwards. This number be accessed via the property
    :py:prop:`total`.  The archive supports incremental updates,
    allowing to fill the archive in multiple steps and to continue
    fill operations that were interrupted. The number of valid data
    points filled so far is stored in the metadata as property
    `valid` and can be accessed through the property :py:prop:`valid`.

    Note: The numpy `memmap` mechanism does not provide means for
    compression.  Files are stored uncompressed and may have extreme
    sizes for larger activation maps of datasources.

    Note: Depending on the file system, memmap may create files of
    desired size but only allocate disk space while filling the files.
    This may result in an (uncatchable) bus error if the device runs
    out of space.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._layers_memmap = None
        self._meta = None
        LOG.info("ActivationsArchiveNumpy at '%s' initalized.",
                 self._directory)

    @property
    def filename_meta(self) -> Path:
        """The name of the file holding the meta data for this
        :py:class:`ActivationsArchiveNumpy`.
        """
        return self._directory / 'meta.json'

    def layers(self, *what) -> Iterable[Tuple[str, type, Tuple[int]]]:
        """Iterate layer of layer information.

        Arguments
        ---------
        what:
            Specifies the what information should be provided. Valid
            values are: `'name'` the layer name,
            `'dtype'` the dtype of the layer,
            `'shape'` the layer layer,
            `'layer'` the actual layer object (only available if the
            :py:class:`Network`, not just the network key, has been provided
             upon initialization of this :py:class:`ActinvationsArchive`).
        """
        if not what:
            what = ('name', )
        elif 'layer' in what and not isinstance(self._network, Network):
            raise ValueError("Iterating over Layers is only possible with "
                             "an initialized Network.")
        for layer in self._layers:
            name = layer.key if isinstance(layer, Layer) else layer
            memmap = self._layers_memmap[name]
            yield ((name if info == 'name' else
                    memmap.dtype if info == 'dtype' else
                    memmap.shape[1:] if info == 'shape' else
                    memmap.nbytes if info == 'bytes' else
                    self._network[name] if info == 'layer' else '?')
                   for info in what)

    def _prepare(self) -> None:
        super()._prepare()

        # prepare the meta data
        filename_meta = self.filename_meta
        if filename_meta.exists():
            with open(filename_meta, 'r') as file:
                meta = json.load(file)
            if self._layers is None:
                self._layers = list(meta['shape'].keys())
        else:  # mode == 'w' and isinstance(network, Network):
            length = len(self._datasource)
            meta = {
                'total': length,
                'valid': 0,
                'dtype': 'float32',  # FIXME[hack]: we should determine dtype
                'shape': {}
            }
            if self._layers is None:
                self._layers = list(self._network.layer_names())
            for layer in self._layers:
                meta['shape'][layer] = \
                    (length,) + self._network[layer].output_shape[1:]
            os.makedirs(self.directory, exist_ok=True)

        # check if all requested layers are availabe
        available_layers = set(meta['shape'].keys())
        requested_layers = set(self._layers)
        if not requested_layers.issubset(available_layers):
            raise ValueError(f"Some requested layers {requested_layers} "
                             f"are not available {available_layers}")
        if self._mode == 'w' and available_layers != requested_layers:
            # make sure that all available layers are written to avoid
            # inconsistent data
            raise ValueError(f"Some available layers {available_layers} "
                             "are not mentioned as write layers "
                             f"{requested_layers}")
        self._meta = meta

        # prepare the layer memmaps
        layers = meta.shape.keys() if self._layers is None else self._layers
        dtype = np.dtype(meta['dtype'])
        self._layers_memmap = {}
        for layer in layers:
            layer_name = layer.key if isinstance(layer, Layer) else layer
            layer_filename = self.directory / (layer_name + '.dat')
            shape = tuple(meta['shape'][layer_name])
            mode = 'r' if self._mode == 'r' else \
                ('r+' if layer_filename.exists() else 'w+')
            self._layers_memmap[layer_name] = \
                np.memmap(layer_filename, dtype=dtype, mode=mode, shape=shape)

        LOG.info("ActivationsArchiveNumpy at '%s' with %d layers and "
                 "%d/%d entries prepared for mode '%s'.", self._directory,
                 len(self._layers), self.valid, self.total, self._mode)

    def _unprepare(self) -> None:
        # make sure all information is stored
        if self._mode == 'w':
            self.flush()

        # close the memmap objects
        if self._layers_memmap is not None:
            for layer, memmap in self._layers_memmap.items():
                del memmap
        self._layers_memmap = None
        self._meta = None
        LOG.info("ActivationsArchiveNumpy at '%s' unprepared.",
                 self._directory)
        super()._unprepare()

    def flush(self) -> None:
        """Flush unwritten data to the disk.  This will also update
        the metadata file (:py:prop:`filename_meta`) to reflect the
        current state of the archive.  :py:meth:`flush` is automatically
        called when upreparing or deleting this
        :py:class:`ActivationsArchiveNumpy` object.

        Note: flusing the data is only allowed (and only makes sense),
        if this :py:class:`ActivationsArchiveNumpy` is in write mode.
        """
        super().flush()
        if self._layers_memmap is not None:
            for layer, memmap in self._layers_memmap.items():
                memmap.flush()

    def __getitem__(self, key) -> None:
        if isinstance(key, tuple):
            layer = key[0]
            index = key[1]
        else:
            layer = None
            index = key

        if layer is None:
            return {layer: memmap[index]
                    for layer, memmap in self._layers_memmap.items()}

        return self._layers_memmap[layer][index]

    def __setitem__(self, key, values) -> None:
        if self._mode != 'w':
            raise ValueError(f"Archive in mode '{self._mode}' is not writable")

        if isinstance(key, tuple):
            layer = key[0]
            index = key[1]
        else:
            layer = None
            index = key

        if layer is None:
            if isinstance(values, dict):
                for layer, layer_values in values.items():
                    self._update_values(layer, index, layer_values)
            elif isinstance(values, list):
                if len(values) != len(self._layers):
                    raise ValueError("Values should be a list of length"
                                     f"{len(self._layers)} not {len(values)}!")
                for layer, layer_values in zip(self._layers, values):
                    self._update_values(layer, index, layer_values)
            else:
                raise ValueError("Values should be a list (of "
                                 f"length {len(self._layers)}) "
                                 f"or a dictionary, not {type(values)}")
        else:
            self._update_values(layer, index, values)

    def _update_values(self, layer, index, value) -> None:
        if isinstance(layer, Layer):
            layer = layer.key
        try:
            self._layers_memmap[layer][index] = value
        except KeyError:
            raise KeyError(f"Invalid layer '{layer}', valid layers "
                           f"are {list(self._layers_memmap.keys())}")

    def __iadd__(self, values) -> object:
        """Add activation values to this
        :py:class:`ActivationsArchiveNumpy`.

        Arguments
        ---------
        values:
            The activation values to add.  Currently only a list or
            dictionary of activation values are supported.
        """
        # FIXME[todo]: allow to add a batch of values
        index = self._meta['valid']
        self[index] = values
        self._meta['valid'] += 1
        return self

    def fill(self, overwrite: bool = False) -> None:
        """Fill this :py:class:`ActinvationsArchive` by computing activation
        values for data from the underlying :py:class:`Datasource`.

        Arguments
        ---------
        overwrite:
            If `True`, the fill process will start with the first
            data item, overwriting results from previous runs.
            If `False`, the fill process will start from where the
            last process stopped (if the archive is already filled
             completly, no further computation is started).
        """
        if not isinstance(self._network, Network):
            self._network = Network[self._network]
            self._network.prepare()
        if not isinstance(self._datasource, Datasource):
            self._datasource = Datasource[self._datasource]
            self._datasource.prepare()
        if overwrite:
            self._meta['valid'] = 0
        with self:
            for index in range(self.valid, self.total):
                self += self._network.get_activations(self._datasource[index])

    def info(self) -> None:
        """Output a summary of this :py:class:`ActivationsArchiveNumpy`.
        """
        print(f"Archive at {self.directory}: {self.valid}/{self.total}")
        for name, dtype, shape, size in \
                self.layers('name', 'dtype', 'shape', 'bytes'):
            print(f" - {name+':':20s} {str(shape):20s} "
                  f"of type {str(dtype):10s} [{format_size(size)}]")

# FIXME[todo]: move to util
def format_size(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi' 'Ei', 'Zi']:
        if num < 1024:
            return f"{num}{unit}{suffix}"
        num //= 1024
    return f"{num}Yi{suffix}"
