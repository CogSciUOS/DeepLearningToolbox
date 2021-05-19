"""A collection of tools for dealing with network activations. This
comprises:

* The :py:class:`ActivationTool` that allows to obtain activation values
  from a :py:class:`Network`.

* The :py:class:`ActivationWorker` is a controller for an
  :py:class:`ActivationTool`, allowing to run it asynchronously.

* The :py:class:`ActivationArchive` allows to store activation values
  for later processing.

"""

# standard imports
from abc import ABC, abstractmethod
from typing import Union, Sequence, List, Tuple, Iterable, Iterator, Set
from pathlib import Path
import os
import logging

# third party imports
import numpy as np

# toolbox imports
from network import Network, Classifier, ShapeAdaptor, ResizePolicy
from network.layers import Layer
from ..datasource import Datasource, Datafetcher
from ..base.observer import BaseObserver
from ..base.prepare import Preparable
from ..base.store import Storable, FileStorage
from ..base.data import Data
from ..util.array import adapt_data_format, DATA_FORMAT_CHANNELS_FIRST
from ..util import nphelper, formating
from ..config import config
from .highscore import Highscore, HighscoreGroup, HighscoreCollection
from .highscore import HighscoreGroupNumpy
from . import Tool, Worker

# logging
LOG = logging.getLogger(__name__)


class Fillable(Storable, ABC, storables=['_valid', '_total']):
    """A :py:class:`Fillable` object can be incrementally filled.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._total = 0
        self._valid = 0

    def __len__(self) -> int:
        """The length of this :py:class:`Fillable` is its
        :py:prop:`valid` size.
        """
        return self.valid

    def _post_prepare(self) -> None:
        super()._post_prepare()

        if not self.full():
            LOG.warning("Fillable is only partly filled (%d/%d)",
                        self._valid, self._total)

    @property
    def total(self) -> int:
        """The total size of this :py:class:`Fillable`. May be more than the
        :py:prop:`valid`.

        """
        return self._total

    @property
    def valid(self) -> int:
        """The valid size of the :py:class:`Fillable` to operate on.  May be
        less than :py:prop:`total` if just a part has been filled yet.

        """
        return self._valid

    def full(self) -> bool:
        """A flag indicating if this :py:class:`ActivationsArchive` archive
        is completely filled, meaning all activation values for the
        :py:class:`Datasource` have been added to the archive.
        """
        return self.valid == self.total

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
        if overwrite:
            self._valid = 0
        with self:
            for index in range(self.valid, self.total):
                self.fill_item(index)
                self.valid = index

    @abstractmethod
    def fill_item(self, index: int) -> None:
        """Fill the given item.
        """


class DatasourceTool(Preparable):
    """A tool that makes use of a :py:class:`Datasource`.

    Properties
    ---------
    datasource: Datasource
        The datasource to process
    """

    # _datasource:
    #    The Datasource for which activation values are computed
    _datasource: Union[str, Datasource] = None
    _datasource_required: bool = True

    def __init__(self, datasource: Union[Datasource, str] = None,
                 layers=None, **kwargs) -> None:
        super().__init__(**kwargs)

        if isinstance(datasource, str):
            self._datasource_key = datasource
            self._datasource = None
        elif isinstance(datasource, Datasource):
            self._datasource_key = datasource.key
            self._datasource = datasource
        else:
            raise ValueError(f"Invalid type {type(datasource)} "
                             "for datasource argument.")

        self._layers = layers and [layer.key if isinstance(layer, Layer)
                                   else layer for layer in self._layers]

    def _prepare(self) -> None:
        super()._prepare()
        if self._datasource_required:
            self._prepare_datasource()

    def _prepare_datasource(self) -> None:
        if self._datasource is None:
            self._datasource = Datasource[self._datasource_key]
        self._datasource.prepare()

    @property
    def datasource_key(self) -> str:
        """Datasource key.
        """
        return self._datasource_key


class NetworkTool(Preparable):
    """A :py:class:`NetworkTool` makes use of a :py:class:`Network`.
    """
    # _network:
    #    The Network by which activation values are obtained.
    _network: Union[str, Network] = None
    _network_required: bool = True

    # _layers:
    #    The keys of the network layers that are used by this NetworkTool
    _layers: List[str] = None  # sequence of layers

    def __init__(self, network: Union[Network, str] = None,
                 layers: Union[Layer, str] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(network, str):  # we got a network key
            self._network_key = network
            self._network = None
        elif isinstance(network, Network):
            self._network_key = network.key
            self._network = network
        else:
            raise ValueError(f"Invalid type {type(network)} "
                             "for network argument.")

        if layers is not None:
            self._layers = [layer.key if isinstance(layer, Layer) else layer
                            for layer in layers]
        else:
            self._layers = None

    @property
    def network_key(self) -> str:
        """Network key.
        """
        return self._network_key

    def _prepare(self) -> None:
        super()._prepare()

        if self._network_required:
            self._prepare_network()

        if self._layers is not None and self._network is not None:
            self.check_layers(self._network.layer_names())

    def _prepare_network(self) -> None:
        if self._network is None:
            self._network = Network[self._network_key]
        self._network.prepare()

    def layers(self, *what) -> Iterable[Tuple]:
        """Iterate layer of layer information.

        Arguments
        ---------
        what:
            Specifies what information should be provided. Valid
            values are: `'name'` the layer name,
            `'layer'` the actual layer object,
            `'shape'` the output shape (without batch axis).
            The values `'layer'` and `'shape'` are only available if the
            :py:class:`Network`, not just the network key, has been provided
             upon initialization of this :py:class:`NetworkTool`).
        """
        if self._layers is None:
            return  # nothing to do
        if not what:
            what = ('name', )
        elif (('layer' in what or 'shape' in what) and
              not isinstance(self._network, Network)):
            raise ValueError(f"Iterating over {what} is only possible with "
                             "an initialized Network.")
        for layer in self._layers:
            name = layer.key if isinstance(layer, Layer) else layer
            layer = layer if isinstance(layer, Layer) else self._network[name]
            values = tuple((name if info == 'name' else
                            layer if info == 'layer' else
                            layer.output_shape[1:] if info == 'shape' else
                            '?')
                           for info in what)
            yield values[0] if len(what) == 1 else values

    def check_layers(self, layers: Sequence[Union[Layer, str]],
                     exact: bool = False) -> None:
        """Check if the layers requested by the tool (stored in
        :py:prop:`_layers`) are contained in the available layers.

        Arguments
        ---------
        layers:
            The available layers.
        exact:
            If `True`, then an exact match is required, if `False`
            it is also ok if :py:prop:`_layers` are a subset of
            `layers`.
        """
        # check if all requested layers are availabe
        available_layers = set(layer.key if isinstance(layer, Layer)
                               else layer for layer in layers)
        requested_layers = set(self._layers)

        if exact and (available_layers != requested_layers):
            diff = requested_layers.symmetric_difference(available_layers)
            raise ValueError(f"Requested layers {requested_layers} and "
                             f"available layer {available_layers} differ:"
                             f"{diff}")
        if not requested_layers.issubset(available_layers):
            raise ValueError(f"Some requested layers {requested_layers} "
                             f"are not available {available_layers}")


class IteratorTool(Storable, storables=['_current_index']):
    """The abstract :py:class:`IteratorTool` interface is intended to
    support tools that do iterative processing.  It has an internal
    index which will be stored (and restored) if the tool is
    intialized with the `store=True` parameter.

    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._current_index = 0

    @property
    def current_index(self) -> int:
        """The current index of this :py:class.`IteratorTool`.
        All indices smaller the the `current_index` already have
        been processed, all indices starting with `current_index`
        still have to be processed.
        """

    def current_range(self, end: int = None) -> Iterator[int]:
        """Iterate over the valid indices for this tool, starting from the
        current index up to the end (the length of this tool).
        """
        if end is None:
            end = len(self)
        for index in range(self._current_index, end):
            yield index
            self._current_index = index


class DatasourceActivations(DatasourceTool, NetworkTool, IteratorTool, ABC):
    """An interface for classes that provide activation values for the of
    an (indexed) :py:class:`Datasource`. The interface allows to
    either access activation values for individual data items,
    identified by their (numerical) index, or to iterate over the
    activation values.  Activation values can be obtained for
    individual layers, or for all simultanously.
    """

    @abstractmethod
    def __getitem__(self, key) -> Union[np.ndarray, dict]:
        """
        Arguments
        ---------
        key:
            Either `index`, or a tuple `(layer, index)`.

        Result
        ------
        activations:
            Either a single activation map of `(layer, index)`, or
            a dictionary of activation maps for `index`.
        """
        
    def __iter__(self) -> Iterator:
        return self.activations()

    def activations(self, layer: str = None) -> Iterator:
        """Iterate over the activation values.
        """
        for index in self.current_range():
            yield self[layer, index]


class DatasourceNetworkActivations(DatasourceActivations):
    """An implementation of the :py:class:`DatasourceActivations` that
    obtains activation values by computing them using a
    :py:class:`Network`.
    """
    _network_required: bool = True
    _datasource_required: bool = True

    def __getitem__(self, key) -> Union[np.ndarray, dict]:
        """
        Arguments
        ---------
        key:
            Either `index`, or a tuple `(layer, index)`.

        Result
        ------
        activations:
            Either a single activation map of `(layer, index)`, or
            a dictionary of activation maps for `index`.
        """
        layer, index = key if isinstance(key, tuple) else (None, key)
        return self._network.get_activations(self._datasource[index], layer)


class ActivationsArchive(DatasourceActivations, Fillable, Storable, ABC):
    """An :py:class:`ActinvationsArchive` represents an archive of
    activation values obtained by applying a
    :py:class:`ActivationTool` to a :py:class:`Datsource`.

    The total size of an :py:class:`ActivationsArchiveNumpy`, that is
    the number of data points for which activations are stored in the
    archive, has to be provided uppon initialization and cannot be
    changed afterwards. This number be accessed via the property
    :py:prop:`total`.  The archive supports incremental updates,
    allowing to fill the archive in multiple steps and to continue
    fill operations that were interrupted. The number of valid data
    points filled so far is stored in the metadata as property
    `valid` and can be accessed through the property :py:prop:`valid`.

    If no explicit argument is provided, the
    :py:class:`ActivationsArchiveNumpy` uses the value
    `config.activations_directory` as well as the `ActivationTool` and
    `Datasource` identifiers (as provided by the `key` property),
    to construct a directory name.
    
    Use cases
    ---------

    Fill the archive by iterating over a :py:class:`Datasource`:

    >>> with ActivationsArchive(network, datasource, store=True) as archive:
    >>>     archive.fill()

    This can also be achieved explicitly:

    >>> with ActivationsArchive(network, datasource, store=True) as archive:
    >>>     for index in range(archive.valid, archive.total):
    >>>         archive += network.get_activations(datasource[index])

    Batchwise filling is also supported:

    >>> with ActivationsArchive(network, datasource, store=True) as archive:
    >>>     for batch in datasource.batches(batch_size=64,start=archive.valid):
    >>>         activation_tool.process(batch)
    >>>         archive += batch

    Once the archive is (partly) filled, it can be used in read only mode:

    >>> with ActivationsArchive(network, datasource, mode='r') as archive:
    >>>     activations = archive[index, 'layer1']
    >>>     batch_activations = archive[index1:index2, 'layer1']

    Activations can also be obtained encapsuled in a :py:class:`Data`
    object:

    >>> with ActivationsArchive(network, datasource, mode='r') as archive:
    >>>     data = archive.data(index, 'layer1')


    Properties
    ----------

    Storable properties
    -------------------
    total: int
        The total number of entries in the underlying :py:class:`Datasource`.
    valid: int
        The number of data entries alread processed.
    _layers: List[str]

    Arguments
    ---------
    layers: List[str]
        The layers to be covered by this :py:class:`ActivationsArchive`.
        If opened for reading (restore), this has to be a subset of
        the layers present in the stored archive.  If creating a new
        archive, this may be any subset of the layers present in the
        :py:class:`Network` (if `None`, all layers of that network
        will be covered). If updating an existing archive, if not `None`,
        the layer list has to exactly match the layers of that archive.
    store: bool
        Open the archive for writing.
    restore: bool
        Open the archive for reading.
    """

    def __new__(cls, **_kwargs) -> 'ActivationsArchive':
        if cls is ActivationsArchive:
            new_cls = ActivationsArchiveNumpy
        else:
            new_cls = cls
        return super(ActivationsArchive, new_cls).__new__(new_cls)

    def __init__(self, **kwargs) -> None:
        # Change the default Storable behaviour (store/restore), to
        # just restore, if no explicit 'store' flag is given:
        if 'store' not in kwargs:
            kwargs['store'] = False
            kwargs['restore'] = True
        super().__init__(**kwargs)

        if not isinstance(self._storage, FileStorage):
            directory = Path(config.activations_directory) /\
                (self._network_key + '-' + self._datasource_key)
            self._storage = FileStorage(directory=directory)

        if self._store_flag:
            self._datasource_required = True
            self._network_required = True
        LOG.info("ActivationsArchiveNumpy with storage '%s' initalized.",
                 self._storage)

    @property
    def directory(self) -> str:
        """The directory in which the activation maps are stored.
        """
        return self._storage.directory

    def _prepare(self) -> None:
        super()._prepare()
        if self._total is None:  # new archive
            self._total = len(self._datasource)
            self._valid = 0

    def fill_item(self, index: int) -> None:
        """Fill the given item.
        """
        self[index] = self._network.get_activations(self._datasource[index])

    def __iadd__(self, values) -> object:
        """Add activation values to this
        :py:class:`ActivationsArchive`.

        Arguments
        ---------
        values:
            The activation values to add.  Currently only a list or
            dictionary of activation values are supported.
        """
        # FIXME[todo]: allow to add a batch of values
        self[self.valid] = values
        self._valid += 1
        return self


class ActivationsArchiveNumpy(ActivationsArchive, DatasourceTool, storables=[
        'shape', 'dtype']):
    """The :py:class:`ActivationsArchiveNumpy` realizes an
    :py:class:`ActivationsArchive` based on the Numpy `memmap`
    mechanism.

    All files of the :py:class:`ActivationsArchiveNumpy` are stored in
    the directory :py:prop:`directory`. Each layer gets a separate
    file, called `[LAYER_NAME].dat`. Metadata for the archive are
    stored in JSON format into the file `meta.json`.


    Notes
    -----

    Note: The numpy `memmap` mechanism does not provide means for
    compression.  Files are stored uncompressed and may have extreme
    sizes for larger activation maps of datasources.

    Note: Depending on the file system, memmap may create files of
    desired size but only allocate disk space while filling the files.
    This may result in an (uncatchable) bus error if the device runs
    out of space.

    """

    def __init__(self, dtype: str = 'float32', **kwargs) -> None:
        self._layers_memmap = None
        super().__init__(**kwargs)
        self.dtype = dtype
        self.shape = None
        LOG.info("%s initalized (%s/%s).", type(self).__name__,
                 self._network_key, self._datasource_key)

    def layers(self, *what) -> Iterator[Tuple]:
        """Iterate over the layer information for the layers covered by this
        :py:class:`ActivationsArchiveNumpy`.

        Arguments
        ---------
        what: str
            Specifies the what information should be provided. Valid
            values are: `'name'` the layer name,
            `'dtype'` the dtype of the layer,
            `'shape'` the layer layer,
            `'layer'` the actual layer object (only available if the
            :py:class:`Network`, not just the network key, has been provided
             upon initialization of this :py:class:`ActinvationsArchive`).
        """
        if self._layers_memmap is None:
            for value in super().layers(*what):
                yield value
            return

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

    def _prepared(self) -> bool:
        return self._layers_memmap is not None and super()._prepared()

    def _prepare(self) -> None:
        super()._prepare()

        # make sure that all requested layers are available. In write
        # mode, all available layers should be updated to avoid
        # inconsistent data
        if self._layers is None:
            self._layers = list(self.shape.keys())
        else:
            self.check_layers(self.shape.keys(), exact=self._store_flag)

        # prepare the layer memmaps
        memmaps = {}
        dtype = np.dtype(self.dtype)
        for layer in self._layers:
            filename = self._storage.filename(layer + '.dat')
            shape = tuple(self.shape[layer])
            mode = 'r' if not self._store_flag else \
                ('r+' if filename.exists() else 'w+')
            memmaps[layer] = \
                np.memmap(filename, dtype=dtype, mode=mode, shape=shape)
        self._layers_memmap = memmaps

        LOG.info("ActivationsArchiveNumpy with storage '%s' and %d layers and "
                 "%d/%d entries prepared for store_flag=%s.", self._storage,
                 len(self._layers), self.valid, self.total, self._store_flag)

    def _unprepare(self) -> None:
        # close the memmap objects
        if self._layers_memmap is not None:
            for memmap in self._layers_memmap.values():
                del memmap
        self._layers_memmap = None
        self._layers = None
        LOG.info("ActivationsArchiveNumpy with storage '%s' unprepared.",
                 self._storage)
        super()._unprepare()

    def _store(self) -> None:
        """Write unwritten data to the disk.  This will also update the
        metadata file to reflect the current state of the archive.
        :py:meth:`store` is automatically called when upreparing or
        deleting this :py:class:`ActivationsArchiveNumpy` object.
        """
        super()._store()
        if self._layers_memmap is not None:
            for memmap in self._layers_memmap.values():
                memmap.flush()

    def _fresh(self) -> None:
        """Creating a fresh archive.
        """
        super()._fresh()
        self._prepare_network()
        self._prepare_datasource()
        self._total = len(self._datasource)
        self.shape = {}

        if self._layers is None:
            self._layers = list(self._network.layer_names())
        total = (self.total,)
        for name, shape in self.layers('name', 'shape'):
            self.shape[name] = total + shape

        # We require the storage directory to exist in order
        # to place the memmeap files.
        os.makedirs(self._storage.directory, exist_ok=True)

    def __getitem__(self, key) -> None:
        layer, index = key if isinstance(key, tuple) else (None, key)

        if layer is None:
            return {layer: memmap[index]
                    for layer, memmap in self._layers_memmap.items()}

        return self._layers_memmap[layer][index]

    def __setitem__(self, key, values) -> None:
        if not self._store_flag:
            raise ValueError("Archive is not writable")

        layer, index = key if isinstance(key, tuple) else (None, key)

        if self._layers is None:
            raise ValueError(f"Cannot set item {layer} "
                             "as no Layers have been initialized "
                             "for this ActivationsArchiveNumpy.")

        if layer is None:
            if isinstance(values, dict):
                for layer, layer_values in values.items():
                    self._update_values(layer, index, layer_values)
            elif isinstance(values, list):
                if len(values) != len(self._layers):
                    raise ValueError("Values should be a list of length "
                                     f"{len(self._layers)} not {len(values)}!")
                for layer, layer_values in zip(self._layers, values):
                    self._update_values(layer, index, layer_values)
            else:
                raise ValueError("Values should be a list (of "
                                 f"length {len(self._layers)} "
                                 f"or a dictionary, not {type(values)}")
        else:
            self._update_values(layer, index, values)

    def _update_values(self, layer, index, value) -> None:
        if isinstance(layer, Layer):
            layer = layer.key
        try:
            self._layers_memmap[layer][index] = value
        except KeyError as error:
            raise KeyError(f"Invalid layer '{layer}', valid layers are "
                           f"{list(self._layers_memmap.keys())}") from error

    def info(self) -> None:
        """Output a summary of this :py:class:`ActivationsArchiveNumpy`.
        """
        print(f"Archive with storage {self._storage}: "
              f"{self.valid}/{self.total}")
        total_size = 0
        for name, dtype, shape, size in \
                self.layers('name', 'dtype', 'shape', 'bytes'):
            print(f" - {name+':':20s} {str(shape):20s} "
                  f"of type {str(dtype):10s} [{formating.format_size(size)}]")
            total_size += size
        print((f"No layers" if self._layers is None else
               f"Total {len(self._layers)} layers") + 
              f" and {formating.format_size(total_size)}")


class TopActivations(HighscoreCollection, DatasourceTool, NetworkTool,
                     IteratorTool, Storable, storables=['_top', 'shape']):
    """The :py:class:`TopActivations` stores the top activation values
    for the layers of a :py:class:`Network`.

    Iterative usage (the `+=` operator)
    -----------------------------------

    In iterative mode, the :py:class:`TopActivations` object will
    use an internal counter for indexing.  Activation values can be
    iteratively added using the `+=` operator.  New activation values
    will get the current internal counter as index value.

    >>> with TopActivations(top=5) as top_activations:
    >>>     for data in datasource[len(top_activations):]:
    >>>         top_activations += network.get_activations(data)


    Properties
    ----------

    top: int
        The number of activation layers to store

    layers: Sequence[Layer]
        The layers for which top activation values are stored.


    """

    def __init__(self, top: int = 9, **kwargs) -> None:
        super().__init__(**kwargs)
        self.shape = None
        self._top = top

    def disciplines(self) -> Iterator[Tuple[str, int]]:
        """The disciplines in this high score. A discipline is described
        by a layer name (key) and a channel index.
        """
        for layer in self.layers('name'):
            for channel in range(self.shape[layer][-1]):
                yield (layer.key, channel)

    def highscore(self, discipline: Tuple[str, int]) -> Highscore:
        """The highscore for a discipline (identified by layer name and
        channel number).
        """
        return self.highscore_group(discipline[0])[discipline[1]]

    def highscore_group(self, layer: Union[Layer, str]) -> HighscoreGroup:
        """The highscore group for a layer.
        """
        name = layer.key if isinstance(layer, Layer) else layer
        return self._highscores[name]

    def activations(self, layer: Union[Layer, str],
                    channel: slice = ...) -> np.ndarray:
        """Top activation values for a given layer and channel.

        Arguments
        ---------
        layer:
            The layer for which the top activation values should
            be returned.
        channel:
            The channel in the layer for which top actviations are
            to be returned. If no channel is provided, the top activation
            values for all channels of that layer are returned.

        Result
        ------
        activations:
            A numpy array providing the activation values. The
            shape will be (top, ) if a channel was specified and
            (channels, top) if no channel was specified.
        """
        return self.highscore_group[layer].scores[channel]

    def indices(self, layer: Union[Layer, str],
                channel: slice = ...) -> np.ndarray:
        """Return indices identifying the input stimuli that
        resulted in the top activation values.

        Arguments
        ---------
        layer:
            The layer for which the indices of top activation inputs
            should be returned.
        channel:
            The channel in the layer for which the indices are to be
            returned.  If no channel is provided, the indices
            of top activation values for all channels of that layer
            are returned.

        Result
        ------
        indices:
            A numpy array providing the activation values. The
            shape will be (top, coordinates, ) if a channel was specified
            and (channels, top, coordinates) if no channel was specified.
        """
        return self.highscore_group[layer].owners[channel]

    @property
    def filename_meta(self) -> Path:
        """The name of the file holding the meta data for this
        :py:class:`ActivationsArchiveNumpy`.
        """
        return self._storage.filename(f'top-{self._top}.json')

    def filename_top(self, layer: str) -> Path:
        """The name of a file for storing top activation values for
        a given :py:class:`Layer`.
        """
        return self._storage.filename(f'top-{self._top}-{layer}.npy')

    def _fresh(self) -> None:
        """Prepare new meta data
        """
        super()._fresh()
        self._prepare_datasource()
        self._prepare_network()

        self.shape = {}

        if self._layers is None:
            self._layers = list(self._network.layer_names())

        for name, layer in self.layers('name', 'layer'):
            shape = layer.output_shape
            channels = shape[-1]
            indices = len(shape) - 1  # -1 for the channel
            # indices: (channels, top, indices)
            self._highscores[name] = \
                HighscoreGroupNumpy(top=self._top, size=channels,
                                    owner_dimensions=indices)
            self.shape[name] = shape

    def _unprepare(self) -> None:
        super()._unprepare()
        LOG.info("TopActivations with storage '%s' unprepared.",
                 self._storage)

    def _store(self) -> None:
        for name in self.layers('name'):
            with self.filename_top(name).open('wb') as outfile:
                self._highscores[name].store(outfile)
        super()._store()

    def _restore(self) -> None:
        super()._restore()  # this should restore the meta data

        for name in self.layers('name'):
            # layer shape: (batch, position..., channel)
            shape = self.shape[name]
            channels = shape[-1]
            # indices: (batch, position...)
            indices = len(shape) - 1  # -1 for the channel
            highscore = HighscoreGroupNumpy(top=self._top, size=channels,
                                            owner_dimensions=indices)
            with self.filename_top(name).open('rb') as file:
                highscore.restore(file)
            self._highscores[name] = highscore

    def __iadd__(self, values) -> object:
        # the index array (containing only one index: the current position)
        index = np.asarray([self._current_index], dtype=np.int)

        if isinstance(values, dict):
            for layer, layer_values in values.items():
                self._highscores[layer].update(index, layer_values[np.newaxis])
        elif isinstance(values, list):
            if len(values) != len(self._layers):
                raise ValueError("Values should be a list of length"
                                 f"{len(self._layers)} not {len(values)}!")
            for layer, layer_values in zip(self._layers, values):
                self._highscores[layer].update(index, layer_values[np.newaxis])
        else:
            raise ValueError("Values should be a list (of "
                             f"length {len(self._layers)}) "
                             f"or a dictionary, not {type(values)}")
        self._current_index += 1
        return self

    def fill(self, activations: DatasourceActivations) -> None:
        """Fill this :py:class:`TopActivations` from a
        :py:class:`DatasourceActivations` object. The object have to
        be compatible, that is the :py:class:`Network` and the
        :py:class:`Datasource` have to agree.

        """
        if self._datasource_key != activations.datasource_key:
            raise ValueError("Incompatible datasoures:"
                             f"{self.datasource_key} for TopActivations vs."
                             f"{activations.datasource_key} for "
                             "DatasourceActivations")
        if self.network_key != activations.network_key:
            raise ValueError("Incompatible networks:"
                             f"{self.network_key} for TopActivations vs."
                             f"{activations.network_key} for "
                             "DatasourceActivations")
        while self._current_index < len(activations):
            self += activations[self._current_index]

    def receptive_field(self, layer: Union[Layer, str],
                        channel: int, top: int = 0) -> np.ndarray:
        """Optain the image patch that causes a top activation of
        the :py:class:`network`.

        This function is only available, if :py:class:`Network` and
        :py:class:`Datasource` are available and prepared.
        """
        indices = self.indices(layer, channel)[top]
        image = self._datasource[indices[0]]
        return self._network.\
            extract_receptive_field(layer, indices[1:-1], image)

    def info(self) -> None:
        """Output a summary of this :py:class:`TopActivations`.
        """
        print(f"TopActivations({self._top}): filled with "
              f"{self._current_index} entries from {self._datasource}")


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
    _network: Network = None

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
        # FIXME[old]:
        self._shape_adaptor = ShapeAdaptor(ResizePolicy.Bilinear())
        self._channel_adaptor = ShapeAdaptor(ResizePolicy.Channels())
        self._data_format = data_format

        # network related
        self.network = network

    @property
    def data_format(self) -> str:
        """The data format (channel first/channel last/...) to be used by this
        :py:class:`ActivationTool`.  If no data format has been set
        for this :py:class:`ActivationTool`, the data format of the
        underlying network will be used.
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
        context = super()._preprocess(**kwargs)
        array = inputs.array if isinstance(inputs, Data) else inputs
        context.add_attribute('inputs', array)
        unlist = False
        if layer_ids is None:
            layer_ids = list(self._network.layer_dict.keys())
        elif not isinstance(layer_ids, list):
            layer_ids, unlist = [layer_ids], True
        context.add_attribute('layer_ids', layer_ids)
        context.add_attribute('unlist', unlist)
        return context

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
        if activations is None:
            return None

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
                            adapt_data_format(activation,
                                              input_format=self._data_format,
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

    @staticmethod
    def top_indices(activations: np.ndarray, top: int = 1,
                    sort: bool = False) -> np.ndarray:
        """Get the indices of the top activations.
        """
        return nphelper.argmultimax(activations, num=top, sort=sort)

    @staticmethod
    def top_activations(activations: np.ndarray, top: int = 1,
                        sort: bool = False) -> np.ndarray:
        """Get the top activation values.
        """
        return nphelper.multimax(activations, num=top, sort=sort)


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

    class Observer(BaseObserver):
        """An :py:class:`Observer` of a :py:class:`ActivationWorker`
        should specify which layers should be computed.
        """

        def layers_of_interest(self, worker) -> Set[Layer]:
            # pylint: disable=no-self-use,unused-argument
            """The layers that this :py:class:`Observer` is interested in.
            """
            return set()

    def __init__(self, network: Network = None, tool: ActivationTool = None,
                 **kwargs) -> None:
        if network is not None:
            if tool is not None:
                raise ValueError("Cannot use both 'tool' and 'network' "
                                 "for initializing a ActivationWorker")
            tool = ActivationTool(network)
        super().__init__(tool=tool, **kwargs)
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
                  None if activations is None else
                  len(activations) if layer is None else activations.shape)
        return activations

    def _ready(self) -> bool:
        # FIXME[hack]
        return (super()._ready() and
                self._tool.network is not None and
                self._tool.network.prepared)

    @property
    def network(self) -> Network:
        """The network employed by this :py:class:`ActivationWorker`.
        """
        return self._tool.network

    # FIXME[todo]: should be renamed or become a setter
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
        self.change(tool_changed=True)

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
        network = self.network
        if network is None or not network.prepared:
            return  # nothing to do

        layers = set()
        # FIXME[problem]: does not work with QObserver:
        #   'QObserverHelper' object has no attribute 'layers_of_interest'
        # for observer in self._observers:
        #     layers |= observer.layers_of_interest(self)
        layer_ids = set(map(lambda layer: layer.id, layers))

        layer_ids |= set(map(lambda layer: layer.id, self._fixed_layers))
        if self._classification and isinstance(network, Classifier):
            layer_ids |= {network.score_layer.id}

        # from set to list
        layer_ids = [layer.id for layer in network.layers()
                     if layer.id in layer_ids]

        got_new_layers = layer_ids > self._layer_ids and self._data is not None
        self._layer_ids = layer_ids
        if got_new_layers:
            self.work(self._data)

    #
    # work on Datasource
    #

    def extract_activations(self, datasource: Datasource,
                            batch_size: int = 128) -> None:
        """Compute network activation values for data from a
        :py:class:`Datasource`.

        Activation values are stored in a variable called `result`.
        """
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
                            batch_size: int = 128) -> Iterator:
        """Iterate over a :py:class:`Datasource` and compute activation
        values for all :py:class:`Data` from that source.

        """

        fetcher = Datafetcher(datasource, batch_size=batch_size)

        index = 0
        for data in fetcher:
            print("iterate_activations: "
                  f"processing {'batch' if data.is_batch else 'data'}")
            self.work(data, busy_async=False)
            activations = self.activations()
            if data.is_batch:
                for index, _view in enumerate(data):
                    yield {layer: activations[layer][index]
                           for layer in activations}
            else:
                yield activations


#
#
# OLD
#
#


class OldTopActivations(TopActivations):
    """FIXME[old]: old methods, probably can be removed ...
    """
    activations = None
    index_batch_start = None
    _top_indices = None
    _top_activations = None
    _fixed_layers = None
    _data = None

    def _old_update_values(self, layer: str, value: np.ndarray,
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
        top_slim = nphelper.argmultimax(slim_values, num=top, axis=0)

        # top_activations: (top, channel)
        top_activations = np.take_along_axis(slim_values, top_slim, axis=0)

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
            # pylint: disable=unbalanced-tuple-unpacking
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

    def old_init_layer_top_activations(self, layers=None,
                                       top: int = 9) -> None:
        if layers is None:
            layers = self._fixed_layers
        for layer in layers:
            self._top_activations[layer] = \
                np.full((layer.filters, layer.filters), -np.inf)
            # index: (datasource index, fiter index)
            self._top_indices[layer] = \
                np.full((layer.filters, 2, layer.filters),
                        np.nan, dtype=np.int)

    def old_update_layer_top_activations(self, layers=None,
                                         top: int = 9) -> None:
        if layers is None:
            layers = self._fixed_layers
        for layer in layers:
            top_activations, top_indices = \
                self._top_activations(self.activations(layer),
                                      datasource_index=self._data.index)
            self._old_merge_top_activations(self._top_activations[layer],
                                            self._top_indices[layer],
                                            top_activations, top_indices)

    @staticmethod
    def _old_merge_top(target_owners: np.ndarray, target_scores: np.ndarray,
                       new_owners: np.ndarray, new_scores: np.ndarray) -> None:
        """me
        """
        # indices: shape = (size, top, indices)
        # values: shape = (size, top)
        top = target_scores.shape[1]
        indices = np.append(target_owners, new_owners, axis=1)
        values = np.append(target_scores, new_scores, axis=1)

        # top_indices: shape = (size, top)
        top_indices = nphelper.argmultimax(values, top, axis=1)
        target_scores[:] = np.take_along_axis(values, top_indices, axis=1)
        # FIXME[bug]: ValueError: `indices` and `arr` must have the
        # same number of dimensions
        # target_owners[:] = np.take_along_axis(indices, top_indices, axis=1)
        for coordinate in range(target_owners.shape[-1]):
            target_owners[:, :, coordinate] = \
                np.take_along_axis(indices[:, :, coordinate],
                                   top_indices, axis=1)
