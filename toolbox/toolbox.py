
"""

FIXME[concept]: we need some concept to run the Toolbox

* The Toolbox can run in a single thread or multi-threaded
* The Toolbay can run with or without graphical user interface

In all cases, we want some consistent behavior:
* it should be possible to stop the Toolbox by typing Ctrl+C
* it should be possible to stop the Toolbox from the GUI
* we want to exit gracefully (no errors)

Concepts:
* initialize the toolbox by calling its constructor
* start the toolbox by calling toolbox.run()
   -> start background threads
   -> if running with GUI, start the main event loop
* stop the toolbox by calling toolbox.stop()
   -> if running with GUI, stop the main event loop
   -> stop all background threads (FIXME[todo])
* quit the toolbox by calling toolbox.quit()
   -> stop the toolbox (call toolbox.stop())
   -> close the main window
"""

# FIXME[hack]: There should be no direct access to the mainWindow.
#  All interaction should be initiated via events

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

from base import (Observable, BusyObservable, busy, change,
                  Runner, Controller as BaseController)
from util import addons

# FIXME[todo]: speed up initialization by only loading frameworks
# that actually needed

from network import Network
from network.examples import keras, torch
from datasources import (Datasource, Labeled as LabeledDatasource,
                         Controller as DatasourceController)


class Toolbox(Semaphore, BusyObservable, Datasource.Observer,
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
    _input_datasource = None
    _input_description = None

    def __init__(self, args):
        Semaphore.__init__(self, 1)
        BusyObservable.__init__(self)
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
        # We use the local import here to avoid circular imports
        # (qtgui.mainwindow imports toolbox.ToolboxController)
        from qtgui import create_gui
        self._mainWindow = create_gui(sys.argv, self._toolbox_controller)
        # FIXME[hack]: we need a better solution here!
        self._runner = self._mainWindow.getRunner()
        self._toolbox_controller.runner = self._runner
        self._activation_controller.runner = self._runner

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
        #
        # Setup handling of KeyboardInterrupt (Ctrl-C)
        #
        import signal
        signal.signal(signal.SIGINT, self._interrupt_handler)
        # There may be a problem when this is done in the context of
        # PyQT: the main event loop is run in C++ core of Qt and is
        # not aware of the Python interpreter. The interrupt handler
        # will only be called after a (Python) event is emitted by the
        # GUI that is handled by the (Python) interpreter.
        # An ad hoc solutino is to use a QTimer to periodically run some
        # (dummy) Python code to make sure the signal handler gets a
        # chance to be executed:
        # safe_timer(50, lambda: None)

        return self._run_gui()

    def quit(self):
        """Quit this toolbox.
        """
        if self._mainWindow is not None:
            # This will stop the main event loop.
            print("Toolbox: Now stopping GUI main event loop.")
            self._mainWindow.stop()
            # Once the GUI main event loop was stopped, also the finally
            # block of the run method is executed ...
            print("Toolbox: Quitting now ...")
        else:
            print("Toolbox: Quitting the toolbox.")
            sys.exit(0)
    
    # FIXME[concept]: It may be better to define the interrupt handler as
    # a global function (not a method) to make sure it is not garbage
    # collected when going out of scope ...
    def _interrupt_handler(self, signum, frame):
        """Handle KeyboardInterrupt: quit application."""
        print(f"Toolbox: Keyboard interrupt: signum={signum}, frame={frame}")
        self.quit()

    def _run_gui(self):
        # Initialize the toolbox in the background, while
        # the (event loop of the) GUI is already started
        # in the foreground. 
        # FIXME[hack]:
        self._runner.runTask(self.initialize_toolbox,
                             self._args, self._mainWindow)
        # self.initialize_toolbox(self._args, self._mainWindow)

        # we start a background timer to periodically updated the user
        # interface.
        # util.start_timer(self._mainWindow.showStatusResources)
        self._mainWindow.safe_timer(1000, self._mainWindow.showStatusResources)

        try:
            # This will enter the main event loop.
            # It will only return once the main event loop exits.
            return self._mainWindow.run()
        finally:
            print("Toolbox: finally stopping the timer ...")
            self._mainWindow.stop_timer()

    def initialize_toolbox(self, args, gui):
        """Initialize the Toolbox, by importing required classes,
        tools, datasets and models.

        The initialization process may take some while and may be
        executed in some background thread.

        FIXME[todo]: There may be some conceptual improvements like:
        * incremental initialization: only load stuff that is currently
          needed. This may reduce the memory footprint and speed up
          initialization.
        * provide some feedback on the progress of initialization
        """
        try:
            from datasources import DataDirectory

            if addons.use('lucid') and False:  # FIXME[todo]
                from tools.lucid import Engine as LucidEngine

                lucid_engine = LucidEngine()
                # FIXME[hack]
                lucid_engine.load_model('InceptionV1')
                lucid_engine.set_layer('mixed4a', 476)
                if self._mainWindow is not None:
                    self._mainWindow.setLucidEngine(lucid_engine)

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
            # if self._mainWindow is not None:
            #     self._mainWindow.setDatasource(source)

            #
            # network: dependes on the selected framework
            #
            # FIXME[hack]: two networks/models seem to cause problems!
            if args.alexnet:
                network = hack_load_alexnet(self)

            elif args.framework.startswith('keras'):
                # "keras-tensorflow" or "keras-theano"
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
                
            if network is not None:
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

    def network_loaded(self, name: str) -> bool:
        for network in self._networks:
            if str(network) == name:
                return True
        return False

    @busy
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
        
        # FIXME[hack]: we need a better mechanism to refer to Datasources
        from datasources.imagenet import Predefined
        imagenet = Predefined.get_data_source('imagenet-val')
        network.set_labels(imagenet, format='caffe')
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
                self.set_input(data=data, label=label, datasource=datasource,
                               description=description)
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
    def input_datasource(self) -> Datasource:
        return self._input_datasource

    @property
    def input_description(self) -> np.ndarray:
        return self._input_description

    def set_input(self, data: np.ndarray, label=None,
                  datasource: Datasource=None, description: str=None):
        self._input_data = data
        self._input_label = label
        self._input_datasource = datasource
        self._input_description = description
        self.change('input_changed')

from .controller import Controller as ToolboxController
