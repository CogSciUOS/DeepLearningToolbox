
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
from datasources import (Datasource, Labeled as LabeledDatasource,
                         Controller as DatasourceController)


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
    _activation_engine: Observable = None # 'ActivationEngine'
    _activation_controller: BaseController = None # 'ActivationController'
    _input_data: np.ndarray = None
    _input_label = None
    _input_label_text: str = ''
    _input_description = None

    def __init__(self, args):
        Semaphore.__init__(self, 1)
        Observable.__init__(self)
        self._args = args
        self._toolbox_controller = ToolboxController(self)

        # FIXME[old] ...
        from tools.activation import (Engine as ActivationEngine,
                                      Controller as ActivationController)
        self._activation_engine = ActivationEngine(toolbox=self)
        self._activation_controller = \
            ActivationController(activation=self._activation_engine)
        network_controller = self._toolbox_controller.autoencoder_controller  # FIXME[hack]
        self._activation_controller.set_network_controller(network_controller)
        
        # we need the GUI first to get the runner ...
        self._initialize_gui1()

        self._initialize_datasources()
        self._initialize_networks()

        self._initialize_gui2()

    def _initialize_gui1(self):
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
        # FIXME[hack]: we need a better solution here!
        self._runner = self._mainWindow.getRunner()
        self._toolbox_controller.runner = self._runner
        self._activation_controller.runner = self._runner
        self._mainWindow.show()

    def _initialize_gui2(self):
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
        #self._mainWindow.panel('maximization', create=True)

        # Initialise the "Resources" panel.
        #self._mainWindow.panel('resources', create=True, show=True)

        # FIXME[old]
        self._mainWindow.setModel(self._activation_engine)


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
        self._runner.runTask(self.initializeToolbox,
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

            if addons.use('lucid') and False:  # FIXME[todo]
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

            self._datasource_controller(source)
            #gui.setDatasource(source)

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
                
            if False and network is not None:
                self.add_network(network)

            self._activation_engine.set_network(network)

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
        # FIXME[todo]: instead of providing labels here, we should
        # refer the network directly to the datasource.
        from datasources.imagenet_classes import class_names_short
        network.set_output_labels(class_names_short)
        logger.debug("alexnet: Done")
        return network

    ###########################################################################
    ###                            Datasources                              ###
    ###########################################################################

    def _initialize_datasources(self):
        self._datasources = []
        self._datasource_controller = DatasourceController(runner=self._runner)
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
        print(f"Toolbox.datasource_changed({datasource}, {change})")
        if change.observable_changed:
            if datasource is not None and datasource not in self._datasources:
                self.add_datasource(datasource)

        if change.data_changed:
            if (self._datasource_controller and
                self._datasource_controller.prepared):
                # we have a datasource that can provide data
                data = self._datasource_controller.data
                if (self._datasource_controller.isinstance(LabeledDatasource)
                    and self._datasource_controller.labels_prepared):
                    # we can also provide labels
                    label = self._datasource_controller.label
                else:
                    label = None
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
    def input_label(self) -> int:
        return self._input_label

    @property
    def input_label_text(self) -> str:
        return self._input_label_text

    @property
    def input_description(self) -> np.ndarray:
        return self._input_description

    def set_input(self, data: np.ndarray, label=None,
                  description: str=None):
        print(f"Toolbox.set_input({data is not None and data.shape}, {label}, {description})")
        self._input_data = data
        self._input_label = label
        if (label is not None and self._datasource_controller and
            self._datasource_controller.isinstance(LabeledDatasource) and
            self._datasource_controller.has_text_for_labels()):
            # We can provide some text for the label
            self._input_label_text = \
                self._datasource_controller.text_for_label(label)
        else:
            self._input_label_text
        self._input_description = description
        self.change('input_changed')


from .controller import Controller as ToolboxController
