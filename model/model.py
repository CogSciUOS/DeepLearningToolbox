import numpy as np
from scipy.misc import imresize

from network import Network
from network.layers import Layer
from util import ArgumentError
from qtgui.datasources import DataSource, DataArray, DataDirectory, DataFile, DataSet



class ModelChange(dict):
    '''
    .. :py:class:: ModelChange

    A class whose instances are passed to observers in :py:meth:`observer.Observer.modelChanged`
    in order to inform them as to the exact nature of the model's change.


    Attributes
    ----------
    network_changed :   bool
                        Whether the underlying :py:class:`network.Network` has changed
    layer_changed   :   bool
                        Whether the current :py:class:`network.layers.Layer` has changed
    unit_changed    :   bool
                        Whether the selected unit changed
    input_index_changed :   bool
                            Whether the index into the dataset changed
    dataset_changed :   bool
                        Whether the underlying :py:class:`qtgui.datasources.DataSource`
                        has changed
    '''

    def __init__(self, **kwargs):
        self['network_changed']     = False
        self['layer_changed']       = False
        self['unit_changed']        = False
        self['input_index_changed'] = False
        self['dataset_changed']     = False
        # set additional properties, if given.
        for k, v in kwargs.items():
            self[k] = v

    def __getattr__(self, attr):
        '''Override for making dict entries accessible by dot notation.

        Parameters
        ----------
        attr    :   str
                    Name of the attribute

        Returns
        -------
        object

        Raises
        ------
        ValueError  :   For unknown attributes.
        '''
        try:
            return self[attr]
        except KeyError:
            raise ValueError(f'{self.__class__.__name__} has no attribute \'{attr}\'.')

    def __setattr__(self, attr, value):
        '''Override for disallowing arbitrary keys in dict.

        Parameters
        ----------
        attr    :   str
                    Name of the attribute
        value   :   object

        Raises
        ------
        ValueError  :   For unknown attributes.
        '''
        if attr not in self:
            raise ValueError(f'{self.__class__.__name__} has no attribute \'{attr}\'.')
        else:
            self[attr] = value

    @staticmethod
    def all():
        '''Create a :py:class:`ModelChange` instance with all properties set to ``True``.'''
        return ModelChange(network_changed=True, layer_changed=True, unit_changed=True,
                           input_index_changed=True, dataset_changed=True)



class Model(object):
    '''.. :py:class:: Model

    Model class encompassing network, current activations, and the like.

    Attributes
    ----------
    _observers  :   set
                    Objects observing this class for changes
    _data   :   np.ndarray
                Current input data
    _current_index  :   int
                        Index of ``_data`` in the data set
    _layer  :   Layer
                Currently selected layer
    _unit   :   int
                Currently selected unit in the layer
    _network    :   Network
                    Currently active network
    _networks   :   dict
                    All available networks. TODO: Move this out of the model
    _current_source     :    DataSource
                            The current :py:class:`qtgui.datasources.DataSource`
    _current_activation :   np.ndarray
                            The last computed activations
    '''
    _observers:             set        = set()
    _data:                  np.ndarray = None
    _network:               Network    = None
    _networks:              dict       = {}
    _layer:                 Layer      = None
    _unit:                  Layer      = None
    _sources:               dict       = {}
    _current_index:         int        = None
    _current_activation:    np.ndarray = None
    _current_source:        DataSource = None

    def __init__(self, network: Network):
        '''Create a new ``Model`` instance.

        Parameters
        ----------
        network :   Network
                    Network instance backing the model
        '''
        self._network = network
        self._networks[str(network)] = network

    def __len__(self):
        '''Returns the number of elements in the currently selected dataset.

        Returns
        -------
        int
        '''
        source = self._current_source
        return 0 if source is None else len(source)

    ################################################################################################
    #                                      OBSERVER HANDLING                                       #
    ################################################################################################
    def addObserver(self, observer):
        '''Add an object to observe this model.

        Parameters
        ----------
        observer    :   object
                        Object which wants to be notified of changes. Must supply a
                        :py:meth:`observer.modelChanged` method.

        '''
        self._observers.add(observer)

    def notifyObservers(self, info: ModelChange):
        '''Notify all observers that the state of this model has changed.

        Parameters
        ----------
        info    :   ModelChange
                    Changes in the model since the last update. If ``None``, do not publish update.

        '''
        if info:
            for o in self._observers:
                o.modelChanged(self, info)

    ################################################################################################
    #                           SETTING CURRENTLY VISUALISED PROPERTIES                            #
    ################################################################################################
    def setNetwork(self, network=None, force_update: bool=False):
        '''Set the current network. Update will only be published if not already selected.

        Parameters
        ----------
        network :   str or int or network.network.Network
                    Key for the network
        force_update    :   bool
                            Force update to be published. This is useful if the current network was
                            passed to the constructor, but not loaded in the GUI yet. The GUI may
                            make sure updates are published by setting this argument.

        Returns
        -------
        ModelChange
            Change notification for the task runner to handle.
        '''
        if self._network != network:
            if isinstance(network, Network):
                if network not in self._networks.values():
                    self.addNetwork(network)
                self._network = network
            elif isinstance(network, str) or isinstance(network, int):
                self._network = self._networks[str(network)]
            else:
                raise ArgumentError(f'Unknown network type {network.__class__}')
            self.setLayer(None)
            return ModelChange(network_changed=True, layer_changed=True)
        elif force_update:
            # argh. code smell
            self.setLayer(None)
            return ModelChange(network_changed=True, layer_changed=True)
        return None


    def idForLayer(self, layer_str: str):
        '''Obtain the numeric id for a given layer identifier.
        .. note:: This operation is linear in the number of layers in the current network.

        Parameters
        ----------
        layer_str   :   str
                        Identifier of the layer

        Returns
        -------
        int
            layer index
        '''
        try:
            layer_keys = list(self._network.layer_dict.keys())
            return layer_keys.index(layer_str)
        except ValueError:
            raise ValueError(f'Layer for string {layer_str} not found.')

    def setLayer(self, layer: Layer=None):
        '''Set the current layer to choose units from.

        Parameters
        ----------
        layer   :   Layer
                    Layer instance to display

        Returns
        -------
        ModelChange
            Change notification for the task runner to handle.
        '''
        if self._layer != layer:
            self._layer = layer
            if self._data is not None and layer:
                self._update_activation()
                if self._unit:
                    self._unit = None
                    return ModelChange(layer_changed=True, unit_changed=True)
                else:
                    return ModelChange(layer_changed=True)
        return None

    def setUnit(self, unit: int=None):
        '''Change the currently visualised channel/unit. This should be called when the
        user clicks on a unit in the :py:class:`QActivationView`.

        Parameters
        ----------
        unit    :   int
                    Index of the unit in the layer (0-based)

        Returns
        -------
        ModelChange
            Change notification for the task runner to handle.
        '''
        n_units = self._current_activation.shape[-1]
        if unit >= 0 and unit < n_units:
            self._unit = unit
            return ModelChange(unit_changed=True)

        return None

    def editIndex(self, index):
        '''Set the current dataset index. Index is left unchanged if out of range.

        Parameters
        ----------
        index   :   int
        '''
        if index is not None:
            try:
                index = int(index)
                if index < 0:
                    raise ValueError('Index out of range')
            except ValueError:
                index = self._current_index

        if index != self._current_index:
            return self._setIndex(index)
        else:
            return None

    def _update_activation(self):
        '''Set the :py:attr:`_current_activation` property by loading activations for
        :py:attr:`_layer` and :py:attr:`_data`. This is a noop if no layer is selected or no data is
        set.'''
        if self._layer and self._data is not None:
            self._current_activation = self.activationsForLayers([self._layer], self._data)

    def _setIndex(self, index=None):
        '''Helper for setting dataset index. Will do nothing if ``index`` is ``None``. This method
        will update the appropriate fields of the model.

        Parameters
        ----------
        index   :   int

        Returns
        -------
        ModelChange
            Change notification for the task runner to handle.
        '''
        source = self._current_source
        if index is None or source is None or len(source) < 1:
            index = None
        elif index < 0:
            index = 0
        elif index >= len(source):
            index = len(source) - 1

        self._current_index = index
        if not source or index is None:
            self._data, _info = None, None
        else:
            self._setInputData(source[index].data)
            if self._layer:
                self._update_activation()

        return ModelChange(input_index_changed=True)

    ################################################################################################
    #                                         SETTING DATA                                         #
    ################################################################################################
    def setDataSource(self, source: DataSource, synchronous=True):
        '''Update the :py:class:`DataSource`.

        Returns
        -------
        ModelChange
            Change notification for the task runner to handle.
        '''
        self._current_source = source
        self._setIndex(0)
        self._update_activation()

        change = ModelChange(dataset_changed=True, input_index_changed=True)
        if not synchronous:
            return change
        else:
            self.notifyObservers(change)

    def setDataArray(self, data: np.ndarray=None):
        '''Set the data array to be used.

        Parameters
        ----------
        data:
            An array of data. The first axis is used to select the
            data record, the other axes belong to the actual data.
        '''
        self.setDataSource(DataArray(data))

    def setDataFile(self, filename: str):
        '''Set the data file to be used.'''
        self.setDataSource(DataFile(filename))

    def setDataDirectory(self, dirname: str=None):
        '''Set the directory to be used for loading data.'''
        self.setDataSource(DataDirectory(dirname))

    def setDataSet(self, name: str):
        '''Set a data set to be used.

        Parameters
        ----------
        name    :   str
                    The name of the dataset. The only dataset supported up to now
                    is 'mnist'.
        '''
        self.setDataSource(DataSet(name))

    def _setInputData(self, data: np.ndarray=None, description: str=None):
        '''Provide one data vector as input for the network.
        The input data must have 2, 3, or 4 dimensions.

        - 2 dimensions means a single gray value image
        - 3 dimensions means a single three-channel image. The channels will
          be repeated thrice to form a three-dimensional input
        - 4 dimensions are only supported insofar as the shape must be
          ``(1, A, B, C)``, meaning the fist dimension is singular and can be
          dropped. Actual batches are not supported.

        The input image shape will be adapted for the chosen network by resizing
        the images

        Parameters
        ----------
        data    :   np.ndarray
                    The data array
        description :   str
                        Data description
        '''

        if data is None or not data.ndim:
            raise ArgumentError('Data cannot be None.')

        if data is not None:
            network_shape = self._network.get_input_shape(include_batch=False)

            if data.ndim > 4 or data.ndim < 2:
                raise ArgumentError(f'Data must have between 2 '
                                    'and 4 dimensions (has {data.ndim}).')

            if data.ndim == 4:
                if data.shape[0] == 1:
                    # first dimension has size of 1 -> remove
                    data = data.squeeze(0)
                else:
                    raise ArgumentError('Cannot visualize batch of images')

            if data.ndim == 2:
                # Blow up to three dimensions by repeating the channel
                data = data[..., np.newaxis].repeat(3, axis=2)

            if data.shape[0:2] != network_shape[0:2]:
                # Image does not fit into network -> resize
                # TODO: Replace with ShapeAdaptor
                data = imresize(data, network_shape[0:2])

            if data.shape[2] != network_shape[2]:
                # different number of channels
                # FIXME[hack]: find better way to do RGB <-> grayscale
                # conversion

                if network_shape[2] == 1:
                    data = np.mean(data, axis=2)
                elif network_shape[2] == 3 and data.shape[2] == 1:
                    data = data.repeat(3, axis=2)
                else:
                    raise ArgumentError('Incompatible network input shape.')

        self._data = data

    def addNetwork(self, network):
        '''Add a model to visualise. This will add the network to the list of
        choices and make it the currently selcted one.

        Parameters
        ----------
        network     :   network.network.Network
                        A network (should be of the same class as currently
                        selected ones)

        Returns
        -------
        ModelChange
            Change notification for the task runner to handle.
        '''
        name = 'Network ' + str(self._network_selector.count())
        self._networks[name] = network
        self.setNetwork(network)
        return ModelChange(network_changed=True)

    ################################################################################################
    #                                          UTILITIES                                           #
    ################################################################################################
    def activationsForLayers(self, layers, data):
        '''Get activations for a set of layers.

        .. attention:: Results should be cached.

        Parameters
        ----------
        layers  :   Layer or list
                    Layers to query
        data    :   np.ndarray
                    Data to pump through the net.
        '''
        if not isinstance(layers, list):
            raise ArgumentError(f'Input must be list, is {type(layers)}')

        activations = self._network.get_activations(
            list(layers), data)[0]

        if activations.ndim in {2, 4}:
            if activations.shape[0] != 1:
                raise RuntimeError('Attempting to visualise batch.')
            activations = np.squeeze(activations, axis=0)

        return activations

    def getNetworkName(self, network: Network) -> str:
        '''Get the name of the currently selected network.
        .. note:: This runs in O(n).

        Parameters
        ----------
        network     :   network.network.Network
                        The network to visualise.

        '''
        name = None
        for n, net in self._networks.items():
            if net == network:
                name = n
        return name

    def getInput(self, index: int) -> np.ndarray:
        '''Obtain input at index.

        Parameters
        ----------
        index : int
                Index into the current dataset

        Returns
        -------
        np.ndarray
        '''
        return self._current_source[index]
