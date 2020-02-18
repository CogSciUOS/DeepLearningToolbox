
"""

FIXME[concept]: we need some concept to run the Toolbox

* The Toolbox can run in a single thread or multi-threaded
* The Toolbox can run with or without graphical user interface

In all cases, we want some consistent behavior:
* it should be possible to stop the Toolbox by typing Ctrl+C [ok]
* it should be possible to stop the Toolbox from the GUI [ok]
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


#
# FIXME[hack]: There should be no direct access to the GUI (MainWindow).
#  All interaction should be initiated via events
#
# Current methods we access in the GUI:
#   - self._gui.getRunner()
#   - self._gui.activateLogging
#   - self._gui.showStatusResources
#   - self._gui.safe_timer
#   - self._gui.stop_timer()
#   - self._gui.run()
#   - self._gui.stop()
#   - self._gui.panel()
#   - self._gui.setLucidEngine()
#   - self._gui.setActivationEngine()
#
# Changing global logging Handler
#

import os
import sys

import importlib.util

import logging
#logging.basicConfig(level=logging.DEBUG)

# local loggger
logger = logging.getLogger(__name__)
logger.info(f"Effective debug level: {logger.getEffectiveLevel()}")

# silencing the matplotlib logger
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import util
from util import addons


logger.info("!!!!!!!!!!!!!!!! Changing global logging Handler !!!!!!!!!!!!!!!!!!!!")
root_logger = logging.getLogger()
root_logger.handlers = []
logRecorder = util.RecorderHandler()
root_logger.addHandler(logRecorder)

import numpy as np

from base import (Observable, BusyObservable, busy, change,
                  Runner, Controller as BaseController)

# FIXME[todo]: speed up initialization by only loading frameworks
# that actually needed

# FIXME[todo]: this will load tensorflow!
from network import Network
#from network.examples import keras, torch

from datasources import (Datasource, Labeled as LabeledDatasource,
                         Controller as DatasourceController)

class Toolbox(BusyObservable, Datasource.Observer,
              method='toolbox_changed',
              changes=['networks_changed',
                       'tools_changed',
                       'datasources_changed', 'datasource_changed',
                       'input_changed'],
              changeables={
                  'datasource': 'datasource_changed'
              }):
    """The :py:class:`Toolbox` is the central instance for
    coordinating :py:class:`Datasource`s, :py:class:`Network`s and
    different tools. It also allows for attaching different kinds of
    user interfaces.

    Datasources
    -----------
    
    The :py:class:`Toolbox` maintains a list of :py:class:`Datasource`s.

    The :py:class:`Toolbox` also has a current :py:class:`Datasource`
    which will be the default :py:class:`Datasource` for all
    components that support such a functionality. Changing the
    :py:class:`Toolbox`es :py:class:`Datasource` will also change the
    :py:class:`Datasource` for these components.


    Input
    -----

    The :py:class:`Toolbox` provides input data for components
    interested in such data. These compoenent will be informed when
    the input data changed via an "input_changed" message.


    Networks
    --------
    

    The :py:class:`Toolbox` is an 

    Changes
    -------
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

    Usage
    -----
    Upon construction, a :py:class:`Toolbox` will be basically empty.
    No resources are loaded, not tools are prepared and no user
    interface is attachted to the new :py:class:`Toolbox`. Those can
    be added by calling specific methods.


    Attributes
    ----------
    _toolbox_controller:
        A :py:class:`ToolboxController` controlling this
        :py:class:`Toolbox`.
    _runner:
        A :py:class:`Runner` that is used to run tools in the background.
        If None, all tasks will be run in the main :py:class:`Thread`.
    _networks:
        A list of :py:class:`Network`s managed by this :py:class:`Toolbox`.
    _tools:
        A list of tools provided by this :py:class:`Toolbox`.
    _gui:
        The GUI associated with the :py:class:`Toolbox` or None
        if no GUI is used.
    """
    _toolbox_controller: BaseController = None  # 'ToolboxController'
    _runner: Runner = None
    _networks: list = None
    _datasources: list = None
    _datasource_controller: DatasourceController = None

    _tools: dict = None

    _input_data: np.ndarray = None
    _input_label = None
    _input_datasource = None
    _input_description = None

    _gui = None


    def __init__(self):
        """Initialize the toolbox.

    
        When running with a GUI, certain concurrency issues arise: The
        GUI should be started quick, showing the user that something
        is happening. It may display some progress indicator, allowing
        to see what is done in the background.  As the GUI should be
        responsive in that situation, this means that the actual
        initialization should be run in a background thread.

        Hence the actual initialization is split into several parts:
        The constructor checks just checks if a GUI should be used.
        The actual initialization is done in the method
        _initialize_toolbox().
        """
        super().__init__()

        # provide access to this Toolbox via a ToolboxController
        self._toolbox_controller = ToolboxController(self)

        self._initialize_tools()
        self._initialize_datasources()
        self._initialize_networks()

    def set_runner(self, runner: Runner) -> None:
        """Set the :py:class:`Runner` to be used by this
        :py:class:`Toolbox`.

        Parameter
        ---------

        runner: Runner
            The :py:class:`Runner` to be used. None means that no
            :py:class:`Runner` should be used.
        """
        # FIXME[concept]: either store it in Toolbox or in the Controller,
        # but not in both. Probably the more accurate way is to store
        # it in the Controller, as that is the object that should
        # actually run threads ...
        self._runner = runner
        self._toolbox_controller.runner = self._runner
        if self._tools is not None:
            for tool in self._tools.values():
                tool.runner = runner
        if self._datasource_controller is not None:
            self._datasource_controller.runner = runner

    # FIXME[concept]: It may be better to define the interrupt handler as
    # a global function (not a method) to make sure it is not garbage
    # collected when going out of scope ...
    def _interrupt_handler(self, signum, frame):
        """Handle KeyboardInterrupt: quit application."""
        print(f"Toolbox: Keyboard interrupt: signum={signum}, frame={frame}")
        self.quit()

    def quit(self):
        """Quit this toolbox.
        """
        if self._gui is not None:
            # This will stop the main event loop.
            print("Toolbox: Now stopping GUI main event loop.")
            self._gui.stop()
            # Once the GUI main event loop was stopped, also the finally
            # block of the run method is executed ...
            print("Toolbox: Quitting now ...")
        else:
            print("Toolbox: Quitting the toolbox.")
            sys.exit(0)
    
    def _run_gui(self, gui="qt", panels=[], **kwargs):
        """Create a graphical user interface for this :py:class:`Toolbox`
        and run the main event loop.
        """
        #
        # create the actual application
        #

        # FIXME[hack]: do not use local imports
        # We use the local import here to avoid circular imports
        # (qtgui.mainwindow imports toolbox.ToolboxController)
        from qtgui import create_gui
        self._gui = create_gui(sys.argv, self._toolbox_controller)

        # FIXME[hack]: we need a better solution here!
        self.set_runner(self._gui.getRunner())

        #
        # Initialise the panels.
        #
        for panel in panels:
            self._gui.panel(panel, create=True)

        # FIXME[old]
        if self.contains_tool('activation'):
            self._gui.setActivationEngine()

        #
        # redirect logging
        #
        self._gui.activateLogging(root_logger, logRecorder, True)      

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
        # An ad hoc solution is to use a QTimer to periodically run some
        # (dummy) Python code to make sure the signal handler gets a
        # chance to be executed. 
        #
        # This should not be a problem for us, as the GUI should start a
        # background timer to periodically update the user interface.

        try:
            # This will enter the main event loop.
            # It will only return once the main event loop exits.
            logger.info("Toolbox: running the GUI main event loop")
            return self._gui.run(**kwargs)
        finally:
            print("Toolbox: finally stopping the timer ...")
            self._gui.stop_timer()

    def setup(self, tools=[], networks=[], datasources=[]):
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
            for tool in tools:
                self.add_tool(tool)

            self._toolbox_controller.hack_load_mnist()
            for datasource in datasources:
                datasource.prepare()
            if len(datasources) > 0:
                self._datasource_controller(datasources[-1])

            for network in networks:
                self.add_network(network)

        except Exception as exception:
            util.error.handle_exception(exception)

    ###########################################################################
    ###                            Networks                                 ###
    ###########################################################################

    def _initialize_networks(self):
        self._networks = []

    def add_network(self, network: Network):
        if isinstance(network, str):
            name = network
            if name == 'alexnet':
                network = self.hack_load_alexnet()
            elif name == 'keras-network':
                # FIXME[concept]: here we really need the command line arguments!
                #dash_idx = args.framework.find('-')
                #backend = args.framework[dash_idx + 1:]
                #network = keras(backend, args.cpu, model_file=args.model)
                network = None
            elif name == 'torch-network':
                # FIXME[hack]: provide these parameters on the command line ...
                #net_file = 'models/example_torch_mnist_net.py'
                #net_class = 'Net'
                #parameter_file = 'models/example_torch_mnist_model.pth'
                #input_shape = (28, 28)
                #network = torch(args.cpu, net_file, net_class,
                #                parameter_file, input_shape)
                network = None
            else:
                raise ValueError(f"Unknown network: '{name}'")

        if network is None:
            return
        if not isinstance(network, Network):
            raise TypeError(f"Invalid type for Network {network}: "
                            f"{type(network)}")
        self._networks.append(network)
        self.change('networks_changed')
        return network

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

        if self.contains_tool('activation'):
            tool = self.get_tool('activation')
            #tool.set_toolbox(self)
            tool.set_network(network)

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
        # FIXME[hack]: Suppress messages from keras/tensorflow
        stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        from keras.datasets import mnist
        sys.stderr = stderr
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
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

    def hack_load_imagenet(self):
        from datasources import Predefined
        imagenet = Predefined.get_data_source('imagenet-val')
        imagenet.prepare()
        return imagenet

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

    ###########################################################################
    ###                               Tools                                 ###
    ###########################################################################

    def _initialize_tools(self) -> None:
        """Initialize the tools of this :py:class:`Toolbox`.
        """
        self._tools = {}

    def add_tool(self, tool) -> None:
        """Add a tool to this :py:class:`Toolbox`. Adding a tool to
        the :py:class:`Toolbox` will have the following effects:
        (1) The tool will use the :py:class:`Runner` of the
            :py:class:`Toolbox` to execute its tasks.
        (2) The tool will become an observer of the Toolbox, that is,
            it will be informed when new data, networks, etc.
            are available in the Toolbox.
        (3) Toolbox observers will be informed that a new tools is
            available (via the "tools_changed" message).

        Parameter
        ---------
        tool:
            The new tool. Eeither an object or a string naming the
            tool. Valid tool names are "activation".

        Result
        ------
        contains: bool
            True, if the tool is in this :py:class:`Toolbox`,
            False otherwise.
        """

        if isinstance(tool, str):
            name = tool
            if name in self._tools:
                return self._tools[name] # tool is already in this Toolbox
                # FIXME[concept]: in some situations it may be desirable
                # to have multiple instances of a tool in the Toolbox,
                # e.g. multiple activation tools if we want to compare
                # activations of multiple networks ...
            
            if name == 'activation':
                # FIXME[hack] ...
                from tools.activation import (Engine as ActivationEngine,
                                              Controller as ActivationController)
                engine = ActivationEngine(toolbox=self)
                tool = ActivationController(activation=engine)
                network_controller = \
                    self._toolbox_controller.autoencoder_controller  # FIXME[hack]
                tool.set_network_controller(network_controller)

            elif name == 'lucid':
                if addons.use('lucid') and False:  # FIXME[todo]
                    from tools.lucid import Engine as LucidEngine

                    lucid_engine = LucidEngine()
                    # FIXME[hack]
                    lucid_engine.load_model('InceptionV1')
                    lucid_engine.set_layer('mixed4a', 476)
                    if self._gui is not None:
                        self._gui.setLucidEngine(lucid_engine)  # FIXME[hack]
                    tool = lucid_engine
            else:
                raise ValueError(f"Unknown tool: '{name}'")
        else:
            name = tool.name
        if isinstance(tool, str):
            # print(f"FIXME[error]: Tool '{tool}' is a string!")
            pass
        elif tool is not None:
            self._tools[name] = tool
            tool.runner = self._runner
            self.change('tools_changed')
        return tool

    def contains_tool(self, tool) -> bool:
        """Check whether this :py:class:`Toolbox` contains the given
        tool.

        Parameter
        ---------
        tool
            The tool to check for.

        Result
        ------
        contains: bool
            True, if the tool is in this :py:class:`Toolbox`,
            False otherwise.
        """
        name = tool if isinstance(tool, str) else tool.name
        return self._tools is not None and name in self._tools

    def get_tool(self, name: str):
        """Get a tool with the given name from this :py:class:`Toolbox`.
        tool.

        Parameter
        ---------
        name: str
            The name of the tool to fetch.

        Result
        ------
        tool: 
            The tool or None, if no tool with this name is contained in
            this :py:class:`Toolbox`.
        """
        return self._tools[name] if self.contains_tool(name) else None


    ###########################################################################
    ###                       Command line options                          ###
    ###########################################################################


    def add_command_line_arguments(self, parser):
        parser.add_argument('--model', help='Filename of model to use',
                            default='models/example_keras_mnist_model.h5')
        parser.add_argument('--data', help='filename of dataset to visualize')
        parser.add_argument('--datadir', help='directory containing input images')

        #
        # Datasources
        #
        logging.debug("importing datasources")
        from datasources import Predefined
        datasets = Predefined.get_data_source_ids()
        logging.debug(f"got datesets: {datasets}")

        if (len(datasets) > 0):
            parser.add_argument('--dataset', help='name of a dataset',
                                choices=datasets, default=datasets[0])
        parser.add_argument('--imagenet', help='Load the ImageNet dataset',
                            action='store_true', default=False)

        parser.add_argument('--framework', help='The framework to use.',
                            choices=['keras-tensorflow', 'keras-theano',
                                     'torch'],
                            default='keras-tensorflow')
        parser.add_argument('--cpu', help='Do not attempt to use GPUs',
                            action='store_true', default=False)
        parser.add_argument('--alexnet', help='Load the AlexNet model',
                            action='store_true', default=False)
        parser.add_argument('--autoencoder',
                            help='Load the autoencoder module (experimental!)',
                            action=addons.UseAddon, default=False)
        parser.add_argument('--advexample',
                            help='Load the adversarial example module'
                            ' (experimental!)',
                            action=addons.UseAddon, default=False)
        parser.add_argument('--internals',
                            help='Open the internals panel'
                            ' (experimental!)',
                            action='store_true', default=False)

    def process_command_line_arguments(self, args):
        util.use_cpu = args.cpu


        #
        # Tools
        #
        tools = []
        tools.append('activation')
        tools.append('lucid')

        #
        # Datasources
        #
        datasources = []

        from datasources import DataDirectory, Predefined
        for id in Predefined.get_data_source_ids():
            datasource = Predefined.get_data_source(id)
            self.add_datasource(datasource)

        if args is not None:
            datasource = None
            if args.data:
                datasource = Predefined.get_data_source(args.data)
            elif args.dataset:
                datasource = Predefined.get_data_source(args.dataset)
            elif args.datadir:
                datasource = DataDirectory(args.datadir)
            if datasource is not None:
                self.add_datasource(datasource)
                datasources.append(datasource)

            if args.imagenet:
                datasources.append(Predefined.get_data_source('imagenet-val'))

        #
        # Networks
        #
        networks = []
        
        if args is not None:
            if args.alexnet:
                networks.append('alexnet')

            if args.framework.startswith('keras'):
                # "keras-tensorflow" or "keras-theano"
                networks.append('keras-network')
            elif args.framework == 'torch':
                networks.append('torch-network')

        # we need the GUI first to get the runner ...
        spec = importlib.util.find_spec('PyQt5')
        if spec is not None:
            gui = 'qt'
        else:
            logging.fatal("No GUI library (PyQt5) was found.")
            gui = None

        if gui is not None:
            panels = []
            if addons.use('autoencoder'):
                panels.append('autoencoder')

            if args.internals:
                panels.append('internals')
            #if addons.internals('internals')
            # Initialise the "Activation Maximization" panel.
            #if addons.use('maximization'):
            #    panels.append('maximization')

            #panels.append('resources')
            #panels.append('activations')

            rc = self._run_gui(gui=gui, panels=panels, tools=tools,
                               networks=networks, datasources=datasources)
        else:
            self.setup(tools=tools, networks=networks, datasources=datasources)
            rc = 0
        return rc


from .controller import Controller as ToolboxController
