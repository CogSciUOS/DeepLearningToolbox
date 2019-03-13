
#
# Changing global logging Handler
#

import os
import sys

print("!!!!!!!!!!!!!!!! Changing global logging Handler !!!!!!!!!!!!!!!!!!!!")
import logging
logging.basicConfig(level=logging.DEBUG)

import util
root_logger = logging.getLogger()
root_logger.handlers = []
logRecorder = util.RecorderHandler()
root_logger.addHandler(logRecorder)

# local loggger
logger = logging.getLogger(__name__)
logger.debug(f"Effective debug level: {logger.getEffectiveLevel()}")

import numpy as np

#from asyncio import Semaphore
from threading import Semaphore

from base import Observable, change, Runner, Controller as BaseController
from util import addons

# FIXME[todo]: speed up initialization by only loading frameworks
# that actually needed

from network import Network
from network.examples import keras, torch
from datasources import Datasource, Controller as DatasourceController


class Toolbox(Semaphore, Observable, Datasource.Observer,
              method='toolbox_changed',
              changes=['lock_changed', 'networks_changed',
                       'datasources_changed', 'datasource_changed',
                       'input_changed'],
              changeables={
                  'datasource': 'datasource_changed'
              }):
    """

    Changes
    -------
    lock_changed:

    networks_changed:
        The networks managed by this Toolbox were changed: this can
        mean that a network was added or removed from the Toolbox,
        of the current network was changed.
    datasources_changed:
        The list of datasources managed by this Toolbox was changed:
    datasource_changed:
        The currently selected datasource was changed.
    input_changed:
        The current input has changed. The new value can be read out
        from the :py:meth:`input` property.
    """
    _networks: list = None
    _datasources: list = None
    _toolbox_controller: BaseController = None  # 'ToolboxController'
    _datasource_controller: DatasourceController = None
    _runner: Runner = None
    _model: 'Model' = None
    _input_data: np.ndarray = None
    _input_label = None
    _input_description = None

    def __init__(self, args):
        Semaphore.__init__(self, 1)
        Observable.__init__(self)
        self._args = args
        self._toolbox_controller = ToolboxController(self)

        # FIXME[old] ...
        from model import Model
        self._model = Model()

        self._initialize_datasources()
        self._initialize_networks()
        self._initialize_gui()       
        
    def _initialize_gui(self):
        #
        # create the actual application
        #

        # FIXME[hack]: do not use local imports
        # FIXME[hack]: do not import Qt here!
        from PyQt5.QtWidgets import QApplication
        self._app = QApplication(sys.argv)

        # FIXME[hack]: this needs to be local to avoid circular imports
        # (qtgui.mainwindow imports toolbox.ToolboxController)
        from qtgui.mainwindow import DeepVisMainWindow
        self._mainWindow = DeepVisMainWindow(self._toolbox_controller)
        self._runner: Runner = self._mainWindow.getRunner()
        self._mainWindow.show()

        #
        # redirect logging
        #
        self._mainWindow.activateLogging(root_logger, logRecorder, True)      

        import util
        util.runner = self._runner  # FIXME[hack]: util.runner seems only be used by qtgui/panels/advexample.py

        #
        # Initialise the panels.
        #
        if addons.use('autoencoder'):
            self._mainWindow.panel('autoencoder', create=True)

        # Initialise the "Activation Maximization" panel.
        #if addons.use('maximization'):
        self._mainWindow.panel('maximization', create=True)

        # Initialise the "Resources" panel.
        self._mainWindow.panel('resources', create=True, show=True)

        # FIXME[old]
        self._mainWindow.setModel(self._model)


    def acquire(self):
        result = super().acquire()
        self.change('lock_changed')
        return result

    def release(self):
        super().release()
        self.change('lock_changed')

    def locked(self):
        return (self._value == 0)

    def run(self):
        return self._run_gui()

    def _run_gui(self):
        # FIXME[hack]
        self._mainWindow._runner.runTask(self.initializeToolbox,
                                         self._args, self._mainWindow)
        #initializeToolbox(args, mainWindow)

        util.start_timer(self._mainWindow.showStatusResources)
        try:
            return self._app.exec_()
        finally:
            util.stop_timer()

    def initializeToolbox(self, args, gui):
        try:
            from datasources import DataDirectory

            if addons.use('lucid'):
                from tools.lucid import Engine as LucidEngine

                lucid_engine = LucidEngine()
                # FIXME[hack]
                lucid_engine.load_model('InceptionV1')
                lucid_engine.set_layer('mixed4a', 476)
                gui.setLucidEngine(lucid_engine)

            from datasources import Predefined
            for id in Predefined.get_data_source_ids():
                datasource = Predefined.get_data_source(id)
                self.add_datasource(datasource)

            if args.data:
                source = Predefined.get_data_source(args.data)
            elif args.dataset:
                source = Predefined.get_data_source(args.dataset)
            elif args.datadir:
                source = DataDirectory(args.datadir)

            gui.setDatasource(source)

            #
            # network: dependes on the selected framework
            #
            # FIXME[hack]: two networks/models seem to cause problems!
            if args.alexnet:
                network = hack_load_alexnet(self)

            elif args.framework.startswith('keras'):
                # "keras-tensorflow" or "keras-theaono""
                dash_idx = args.framework.find('-')
                backend = args.framework[dash_idx + 1:]
                network = keras(backend, args.cpu, model_file=args.model)

            elif args.framework == 'torch':
                # FIXME[hack]: provide these parameters on the command line ...
                net_file = 'models/example_torch_mnist_net.py'
                net_class = 'Net'
                parameter_file = 'models/example_torch_mnist_model.pth'
                input_shape = (28, 28)
                network = torch(args.cpu, net_file, net_class,
                                parameter_file, input_shape)
            else:
                network = None

            # FIXME[hack]: the @change decorator does not work in different thread
            #self._model.add_network(network)
            m,change = self._model.add_network(network)
            m.notifyObservers(change)

        except Exception as e:
            # FIXME[hack]: rethink error handling in threads!
            import traceback
            print(e)
            traceback.print_tb(e.__traceback__)


    ###########################################################################
    ###                            Networks                                 ###
    ###########################################################################

    def _initialize_networks(self):
        self._networks = []

    def add_network(self, network: Network):
        self._networks.append(network)
        self.change('networks_changed')

    def remove_network(self, network: Network):
        self._networks.remove(network)
        self.change('networks_changed')

    def hack_load_alexnet(self):
        #
        # AlexNet trained on ImageNet data (TensorFlow)
        #
        logger.debug("alexnet: import tensorflow")
        from network.tensorflow import Network as TensorFlowNetwork
        checkpoint = os.path.join('models', 'example_tf_alexnet',
                                  'bvlc_alexnet.ckpt')
        logger.debug("alexnet: TensorFlowNetwork")
        network = TensorFlowNetwork(checkpoint=checkpoint, id='AlexNet')
        logger.debug("alexnet: prepare")
        network._online()
        logger.debug("alexnet: Load Class Names")
        from datasources.imagenet_classes import class_names2
        network.set_output_labels(class_names2)
        logger.debug("alexnet: Done")
        return network

    ###########################################################################
    ###                            Datasources                              ###
    ###########################################################################

    def _initialize_datasources(self):
        self._datasources = []
        self._datasource_controller = DatasourceController(self._model)
        # observe the new DatasourceController and add new datasources
        # reported by that controller to the list of known datasources
        my_interests = Datasource.Change('observable_changed', 'data_changed')
        self._datasource_controller.add_observer(self, interests=my_interests)

        # FIXME[hack]: training - we need a better concept ...
        self.dataset = None
        self.data = None
        self._toolbox_controller.hack_load_mnist()

    def datasource_changed(self, datasource: Datasource,
                           change: Datasource.Change) -> None:
        if change.observable_changed:
            self.add_datasource(datasource)
        if change.data_changed:
            if self._datasource_controller:
                data, label = self._datasource_controller.data_and_label
                description = self._datasource_controller.description
                self.set_input(data=data, label=label, description=description)
            else:
                self.set_input(data=None, label=None, description="No input")

    @property
    def datasource(self) -> Datasource:
        return (self._datasource_controller._datasource  # FIXME[hack]: private
                if self._datasource_controller else None)

    def add_datasource(self, datasource: Datasource):
        if datasource is not None and datasource not in self._datasources:
            self._datasources.append(datasource)
            self.change('datasources_changed')

    def remove_datasource(self, datasource: Datasource):
        if datasource not in self._datasources:
            self._datasources.remove(datasource)
            self.change('datasources_changed')

    def hack_load_mnist(self):

        """Initialize the dataset.
        This will set the self._x_train, self._y_train, self._x_test, and
        self._y_test variables. Although the actual autoencoder only
        requires the x values, for visualization the y values (labels)
        may be interesting as well.

        The data will be flattened (stored in 1D arrays), converted to
        float32 and scaled to the range 0 to 1. 
        """
        if self.dataset is not None:
            return  # already loaded
        # load the MNIST dataset
        from keras.datasets import mnist
        mnist = mnist.load_data()
        #self.x_train, self.y_train = mnist[0]
        #self.x_test, self.y_test = mnist[1]
        self.dataset = mnist
        self.data = mnist[1]  # FIXME[hack]

        # FIXME[hack]: we need a better training concept ...
        from tools.train import Training
        self.training = Training()
        #self.training.\
        #    set_data(self.get_inputs(dtype=np.float32, flat=True, test=False),
        #             self.get_labels(dtype=np.float32, test=False),
        #             self.get_inputs(dtype=np.float32, flat=True, test=True),
        #             self.get_labels(dtype=np.float32, test=True))

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = (None, None) if data is None else data 

    @property
    def inputs(self):
        return self._data[0]

    @property
    def labels(self):
        return self._data[1]

    ###########################################################################
    ###                               Input                                 ###
    ###########################################################################

    @property
    def input_data(self) -> np.ndarray:
        return self._input_data

    @property
    def input_label(self) -> np.ndarray:
        return self._input_label

    @property
    def input_description(self) -> np.ndarray:
        return self._input_description

    def set_input(self, data: np.ndarray, label=None,
                  description: str=None):
        self._input_data = data
        self._input_label = label
        self._input_description = description
        self.change('input_changed')


###########################################################################
### FIXME[old]: model code
###########################################################################

from typing import Dict, Iterable

import numpy as np

from network import Network, ShapeAdaptor, ResizePolicy
from network.layers import Layer
from util import ArgumentError
from base.observer import Observable, change
from util import async


class OldModel:
    """.. :py:class:: Model

    Model class encompassing network, current activations, and the like.

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

    def __init__model(self, network: Network=None):
        """Create a new ``Model`` instance.

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
        self._networks = {}
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
            self.add_network(network)

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
        :py:class::ShapeAdaptor for this Model.

        Parameters
        ----------
        data: np.ndarray
            The data array
        target: int
            The data label. None if no label is available.
        description: str
            A description of the input data.
        """
        logger.info(f"Model.set_input_data({data.shape},{target},{description})")
        #
        # do some sanity checks and corrections
        #
        if data is None or not data.ndim:
            raise ArgumentError('Data cannot be None.')

        if data.ndim > 4 or data.ndim < 2:
            raise ArgumentError(f'Data must have between 2 '
                                'and 4 dimensions (has {data.ndim}).')

        if data.ndim == 4:
            if data.shape[0] == 1:
                # first dimension has size of 1 -> remove
                data = data.squeeze(0)
            else:
                raise ArgumentError('Cannot visualize batch of images')

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
        if isinstance(network, str):
            network = self._networks.get(network)

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

    @async
    @change
    def _update_activation(self):
        """Set the :py:attr:`_current_activation` property by loading
        activations for :py:attr:`_layer` and :py:attr:`_data`.
        This is a noop if no layers are selected or no data is
        set."""

        self._update_layer_list()
        logger.info(f"Model._update_activation: LAYERS={self._layers}")
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
                logger.info(f"Model.update_activation(): "
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

###############################################################################


from .controller import Controller as ToolboxController
