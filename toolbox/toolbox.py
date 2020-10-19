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
# FIXME[todo]: Direct access to the GUI (MainWindow) should be reduced
# to a minimum. Most interaction should be initiated via events
#
# Current methods we access in the GUI:
#   - self._gui = gui_module.create_gui()
#   - self._gui.run()
#   - self._gui.stop()
#   - self._gui.panel()
#
# We also call the following, which should be avoided:
#   - self._gui.getRunner()
#   - self._gui.setLucidEngine()
#
# Changing global logging Handler
#

# standard imports
from typing import Iterable, Iterator, Union
import os
import sys
import signal
import threading
import importlib.util
import logging
from argparse import ArgumentParser

# third party imports
import numpy as np

# toolbox imports
import util
from util import addons
from base import BusyObservable, Runner, Controller as BaseController
from dltb.util.image import imread
from dltb.base import run
from dltb.tool import Tool
from dltb.base.data import Data

# FIXME[hack]: provide predefined Datasources
import datasource.predefined

# FIXME[todo]: speed up initialization by only loading frameworks
# that actually needed

# FIXME[todo]: this will load tensorflow!
from dltb.network import Network, argparse as NetworkArgparse
# FIXME[old]: make AutoencoderController a tool
#from network import AutoencoderController
# from network.examples import keras, torch
from dltb.datasource import Datasource, Datafetcher, DataDirectory
from tools.train import TrainingController
from .process import Process

# logging

# The local Toolbox loggger
LOG = logging.getLogger(__name__)
LOG.info("Effective log level for logger '%s': %d",
         __name__, LOG.getEffectiveLevel())

# FIXME[hack]: silencing the matplotlib logger.
# The 'matplotlib' plotter seems to be set on `logging.DEBUG`, causing
# the `matplotlib` module upon import to emit a lot of messages.
logging.getLogger('matplotlib').setLevel(logging.WARNING)


class Toolbox(BusyObservable, Datafetcher.Observer,
              method='toolbox_changed',
              changes={'networks_changed',
                       'tools_changed',
                       'datasources_changed', 'datasource_changed',
                       'input_changed', 'server_changed', 'shell_changed',
                       'processes_changed'},
              changeables={
                  'datasource': 'datasource_changed'
              }):
    """The :py:class:`Toolbox` is the central instance for
    coordinating :py:class:`Datasource`s, :py:class:`Network`s and
    different tools. It also allows for attaching different kinds of
    user interfaces.

    Upon construction, a :py:class:`Toolbox` will be basically empty.
    No resources are loaded, no tools are prepared and no user
    interface is attachted to the new :py:class:`Toolbox`. Those can
    be added by calling specific methods.

    **Datasources**

    The :py:class:`Toolbox` maintains a list of :py:class:`Datasource`s.

    The :py:class:`Toolbox` also has a current :py:class:`Datasource`
    which will be the default :py:class:`Datasource` for all
    components that support such a functionality. Changing the
    :py:class:`Toolbox`es :py:class:`Datasource` will also change the
    :py:class:`Datasource` for these components.


    **Input**

    The :py:class:`Toolbox` provides input data for components
    interested in such data. These compoenent will be informed when
    the input data changed via an "input_changed" message.


    **Networks**

    The :py:class:`Toolbox` is an 


    **Tools**

    **Changes**

    state_changed:
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

    **Logging and error handling**

    The logging behaviour of the :py:class:`Toolbox` can be altered
    by setting :py:class:`logging.Handler`. The Toolbox basically
    provides two Handlers which can be activated or deactivated
    independently: The console handler and the GUI handler.
    The GUI handler will be used when a GUI is started that provides
    a handler. The console handler is used, when no GUI handler
    is active, or when a shell is running.

    Both can be altered by setting gui_logging and/or console_logging.
    The handler can be accessed and configured via the properties
    gui_logging_handler and console_logging_handler.




    Attributes
    ----------
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
    
    _datasources: list
        A list of :py:class`Datasource`s available to this :py:class:`Toolbox`.
    _input_data: np.ndarray
        Current input data of this :py:class:`Toolbox`. Tools can observe
        the Toolbox and get informed whenever the input data change.
    _input_meta: dict
        Metadata describing the current input data.

    _command_line_arguments
    """
    _runner: Runner = None
    _networks: list = None
    _datasources: list = None
    _datasource: Datasource = None

    _tools: dict = None

    _input_data: Data = None

    _gui = None

    def __init__(self) -> None:
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

        self._initialize_logging()
        self._initialize_exception_handler()
        self._initialize_processes()
        self._initialize_tools()
        self._initialize_datasources()
        self._initialize_networks()

    def _uninitialize(self) -> None:
        """Free resources acquired by the toolbox.
        Most important: stop running threads and background processes.
        """
        # self._uninitialize_logging()
        # self._uninitialize_exception_handler()
        # self._uninitialize_processes()
        # self._uninitialize_tools()
        self._uninitialize_datasources()
        # self._uninitialize_networks()

    @property
    def runner(self) -> Runner:
        return self._runner

    def set_runner(self, runner: Runner) -> None:
        """Set the :py:class:`Runner` to be used by this
        :py:class:`Toolbox`.

        Parameters
        ----------

        runner: Runner
            The :py:class:`Runner` to be used. None means that no
            :py:class:`Runner` should be used.
        """
        # FIXME[concept]: either store it in Toolbox or in the Controller,
        # but not in both. Probably the more accurate way is to store
        # it in the Controller, as that is the object that should
        # actually run threads ...
        self._runner = runner
        if self._tools is not None:
            for tool in self._tools.values():
                tool.runner = runner
        if self.datasource is not None:
            self.datasource.runner = runner

    # FIXME[concept]: It may be better to define the interrupt handler as
    # a global function (not a method) to make sure it is not garbage
    # collected when going out of scope ...
    def _interrupt_handler(self, signum, frame):
        """Handle KeyboardInterrupt: quit application."""
        print(f"Toolbox: Keyboard interrupt: signum={signum}, frame={frame}")
        if signum == signal.SIGINT:
            if self.shell:
                print("Toolbox: Shell is already running.")
            else:
                print("Toolbox: Starting a shell")
                self.run_shell()
            print("Type 'quit' or press Ctrl+\\ to quit the program; "
                  "type 'bye' or press Ctrl+D to leave the shell")
        elif signum == signal.SIGQUIT:
            self.quit()

    def quit(self, sys_exit=False):
        """Quit this toolbox.
        """
        # FIXME[problem]: some things seem to block the exit process.
        # - if the webcam is looping, the program will not exit
        # FIXME[todo]: we need a more general solution to abort all
        # operations running in the background

        # unset the datasource - this will also stop loops
        self.datasource = None

        if self._runner is not None:
            self._runner.quit()

        self._finish_processes()

        # Stop the web server if running
        self.server = False

        if self._gui is not None:
            # This will stop the main event loop.
            print("Toolbox: Now stopping GUI main event loop.")
            self._gui.stop()
            # Once the GUI main event loop was stopped, also the finally
            # block of the run method is executed ...
            print("Toolbox: Quitting now ...")
        else:
            print("Toolbox: Quitting the toolbox.")

        if sys_exit:
            sys.exit(0)

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

            for datasource in datasources:
                datasource.prepare()
            if len(datasources) > 0:
                self.datasource = datasources[-1]

            for network in networks:
                self.add_network(network)

        except Exception as exception:
            util.error.handle_exception(exception)

    #
    # Networks
    #

    def _initialize_networks(self):
        self._networks = []

    def add_network(self, network: Union[str, Network],
                    prepare: bool = False) -> None:
        """Add a network to the toolbox.

        Arguments
        ---------
        network:
            Either the network or a registered network name.
        prepare: bool
            If `True`, the network will be prepared. This may
            take some time.
        """
        if isinstance(network, str):
            try:
                network = Network[network]
            except KeyError:
                raise ValueError(f"Unknown network: '{network}'")

        if not isinstance(network, Network):
            raise TypeError(f"Invalid type for Network {network}: "
                            f"{type(network)}")
        self._networks.append(network)
        self.change('networks_changed')
        if prepare:
            network.prepare()
        return network

    def remove_network(self, network: Network):
        self._networks.remove(network)
        self.change('networks_changed')

    def network_loaded(self, name: str) -> bool:
        for network in self._networks:
            if str(network) == name:
                return True
        return False

    @property
    def networks(self) -> Iterable[Network]:
        """Networks registered with this Toolbox."""
        return iter(self._networks)

    #
    # Datasources
    #

    def _initialize_datasources(self) -> None:
        """Initialized the datasources managed by this :py:class:`Toolbox`.
        """
        self._datasources = []

        # a datafetcher for the currently selected datasource.
        self._datafetcher = Datafetcher()
        interests = Datafetcher.Change('data_changed')
        self.observe(self._datafetcher, interests=interests)

        # FIXME[hack]: training - we need a better concept ...
        # self.dataset = None
        # self.data = None

    def _uninitialize_datasources(self) -> None:
        """Free (most) datasource related resources acquired by the toolbox.
        Most important: stop running threads.
        """
        self._datafetcher.datasource = None  # will stop running loops

    def add_datasource(self, datasource: Union[str, Datasource],
                       select: bool = False) -> None:
        """Add a new Datasource to this Toolbox.

        Arguments
        ---------
        datasource:
            Either a :py:class:`Datasource` or a key (str) identifying
            a datasource.  In the latter case the datasource will
            be instantiated if it doesn't exist yet'.
        select:
            A flag indicating if the newly add Datasource should become
            the active datasource.
        """
        if isinstance(datasource, str):
            datasource = Datasource[datasource]

        if datasource not in self._datasources:
            self._datasources.append(datasource)
            self.change('datasources_changed')
        if select:
            self.datasource = datasource

    def remove_datasource(self, datasource: Union[str, Datasource]) -> None:
        # unpythonic first-check-then-removal style:
        if isinstance(datasource, str):
            datasource = next((d for d in self._datasources
                               if d.key == datasource), None)
        elif datasource not in self._datasources:
            datasource = None
        if datasource is not None:
            self._toolbox.remove_datasource(datasource)
            self.change('datasources_changed')
            if datasource is datasource:
                self.datasource = None

    @property
    def datasources(self) -> Iterable[Datasource]:
        """Datasources registered with this Toolbox."""
        return iter(self._datasources)

    @property
    def datafetcher(self) -> Datasource:
        return self._datafetcher

    @property
    def datasource(self) -> Datasource:
        return self._datafetcher.datasource

    @datasource.setter
    def datasource(self, datasource: Datasource) -> None:

        # Add new datasources to the list of known datasources
        if datasource is not None:
            self.add_datasource(datasource, select=False)

        # set the datasource to be used for fetching data
        self._datafetcher.datasource = datasource

        # Inform observers that we have a new datasource
        self.change('datasource_changed')

    def datafetcher_changed(self, datafetcher: Datafetcher,
                            change: Datafetcher.Change) -> None:
        """React to a change of the observed :py:class:`Datasource`.

        Arguments
        ---------
        datafetcher: Datafetcher
            The datafetcher that has changed. This should be the
            datafetcher of this :py:class:`Toolbox`.
        change: Datafetcher.Change
            The change that occured. We are interested in a change
            of the :py:class:`Data`.
        """
        if change.data_changed:
            self.set_input(data=datafetcher.data)  # data may be None

    #
    # Input
    #

    @property
    def input_data(self) -> Data:
        """The current input data of this :py:class:`Toolbox`.
        """
        return self._input_data

    def set_input(self, data: Data):
        """Set a new input :py:data:`Data` object for this
        :py:class:`Toolbox`. Observers will be notified.
        """
        self._input_data = data
        self.change('input_changed')

    # FIXME[todo]: we should set input_data busy
    @run
    def set_input_from_file(self, filename: str,
                            description: str = None) -> None:
        """Read data from a file and set a new input
        :py:data:`Data` object for this :py:class:`Toolbox`.
        Observers will be notified.

        Currently only image files are supported.
        """
        # FIXME[todo]: we should allow for other data than images
        image = imread(filename)
        data = Data(data=image)
        data.type = Data.TYPE_IMAGE
        data.add_attribute('shape', image.shape)
        data.add_attribute('filename', filename)
        data.add_attribute('description',
                           description or "Image loaded from file")
        self.set_input(data)

    #
    # Tools
    #

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

        Parameters
        ----------
        tool:
            The new tool. Eeither an object or a string naming the
            tool. Valid tool names are "activation".

        Returns
        ------
        contains: bool
            True, if the tool is in this :py:class:`Toolbox`,
            False otherwise.
        """

        if isinstance(tool, str):
            key = tool
            if key in self._tools:
                return self._tools[key]  # tool is already in this Toolbox
                # FIXME[concept]: in some situations it may be desirable
                # to have multiple instances of a tool in the Toolbox,
                # e.g. multiple activation tools if we want to compare
                # activations of multiple networks ...

            if key == 'activation':
                # FIXME[old/todo]: should the Toolbox have an
                # activation tool?
                pass 

            elif key == 'lucid':
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
                raise ValueError(f"Unknown tool: '{key}'")
        else:
            key = tool.key
        if isinstance(tool, str):
            # print(f"Toolbox: FIXME[error]: Tool '{tool}' is a string!")
            pass
        elif tool is not None:
            self._tools[key] = tool
            tool.runner = self._runner
            self.change('tools_changed')
        return tool

    def contains_tool(self, tool) -> bool:
        """Check whether this :py:class:`Toolbox` contains the given
        tool.

        Parameters
        ----------
        tool
            The tool to check for.

        Returns
        ------
        contains: bool
            True, if the tool is in this :py:class:`Toolbox`,
            False otherwise.
        """
        key = tool if isinstance(tool, str) else tool.key
        return self._tools is not None and key in self._tools

    def get_tool(self, key: str):
        """Get a tool with the given name from this :py:class:`Toolbox`.
        tool.

        Parameters
        ----------
        key: str
            The name of the tool to fetch.

        Returns
        ------
        tool:
            The tool or None, if no tool with this name is contained in
            this :py:class:`Toolbox`.
        """
        return self._tools[key] if self.contains_tool(key) else None

    @property
    def tools(self) -> Iterator[Tool]:
        """Iterate over the tools registered in this :py:class:`Toolbox`.
        """
        return self._tools.values()

    def tools_of_type(self, cls: type) -> Iterator[Tool]:
        """Iterate over the tools registered in this :py:class:`Toolbox`
        of a given type.

        Arguments
        ---------
        cls: type
            The class of which the tool should be an instance.
        """
        for tool in self.tools:
            if isinstance(tool, cls):
                yield tool

    #
    # Command line options
    #

    def option(self, name, default: bool = False) -> bool:
        if not hasattr(self, '_command_line_arguments'):
            return default
        return getattr(self._command_line_arguments, name, default)

    def _prepare_argument_parser(self) -> ArgumentParser:
        """Prepare a Toolbox ArgumentParser.
        """

        parser = ArgumentParser(description='Deep Learning Toolbox')

        #
        # General options
        #
        parser.add_argument('--cpu', help='Do not attempt to use GPUs',
                            action='store_true', default=False)
        # FIXME[old]: this option has been integrated into
        # NetworkArgparse
        # parser.add_argument('--framework', help='The framework to use.',
        #                     choices=['keras-tensorflow', 'keras-theano',
        #                              'torch'],
        #                     default='keras-tensorflow')
        parser.add_argument('--shell', help='Run the toolbox shell',
                            action='store_true', default=False)

        #
        # Models
        #
        NetworkArgparse.prepare(parser)
        parser.add_argument('--autoencoder',
                            help='Load the autoencoder module (experimental!)',
                            action=addons.UseAddon, default=False)

        #
        # Data
        #
        parser.add_argument('--data', help='filename of dataset to visualize')
        parser.add_argument('--datadir',
                            help='directory containing input images')

        #
        # Datasources
        #
        logging.debug("importing datasources")
        datasources = list(Datasource.instance_register.keys())
        logging.debug(f"got datasources: {datasources}")

        if (len(datasources) > 0):
            parser.add_argument('--datasource', help='name of a datasources',
                                choices=datasources)
        parser.add_argument('--imagenet', help='Load the ImageNet datasource',
                            action='store_true', default=False)
        parser.add_argument('--helen', help='Load the Helen datasource',
                            action='store_true', default=False)

        #
        # Modules
        #
        parser.add_argument('--internals',
                            help='Open the internals panel (experimental!)',
                            action='store_true', default=False)
        parser.add_argument('--face',
                            help='Open the face module',
                            action='store_true', default=False)
        parser.add_argument('--face-all',
                            help='Load additional detectors with the '
                            'face module',
                            action='store_true', default=False)
        parser.add_argument('--resources',
                            help='Open the resources panel',
                            action='store_true', default=False)
        parser.add_argument('--activations',
                            help='Open the activations panel',
                            action='store_true', default=False)
        parser.add_argument('--adversarial',
                            help='Open the adversarial examples panel',
                            action='store_true', default=False)
        parser.add_argument('--styletransfer',
                            help='Open the style transfer module'
                            ' (experimental!)',
                            action='store_true', default=False)
        parser.add_argument('--advexample',
                            help='Load the adversarial example module'
                            ' (experimental!)',
                            action=addons.UseAddon, default=False)

        #
        # Debugging
        #
        parser.add_argument('--info', default=[], action='append',
                            metavar='MODULE',
                            help='Show info messages from MODULE')
        parser.add_argument('--debug', default=[], action='append',
                            metavar='MODULE',
                            help='Show debug messages from MODLE')

        #
        # Bugs
        #
        parser.add_argument('--firefox-bug', help='avoid the firefox bug',
                            action='store_true', default=False)

        return parser

    def process_command_line_arguments(self) -> None:
        """Process the arguments given to the :py:class:`ArgumentParser`.
        """

        parser = self._prepare_argument_parser()
        args = parser.parse_args()
        self._command_line_arguments = args

        #
        # Global flags
        #
        util.use_cpu = args.cpu

        #
        # Debugging
        #
        if args.info:
            info_handler = logging.StreamHandler(sys.stderr)
            info_handler.setLevel(logging.INFO)
            info_handler.setFormatter(self.logging_formatter)
            print(f"Toolbox: outputing info messages from '{args.info}' "
                  "on {info_handler}")
            for module in args.info:
                logger = logging.getLogger(module)
                logger.addHandler(info_handler)
                logger.info(f"Outputing debug messages from module {module}")

        if args.debug:
            debug_handler = logging.StreamHandler(sys.stderr)
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(self.logging_formatter)
            print(f"Toolbox: outputing debug messages from '{args.debug}' "
                  "on {debug_handler}")
            for module in args.debug:
                logger = logging.getLogger(module)
                logger.addHandler(debug_handler)
                logger.debug(f"Outputing debug messages from module {module}")

        #
        # Tools
        #
        tools = []
        tools.append('activation')
        tools.append('lucid')
        if args is not None:
            if args.face or args.face_all:
                for key in ('widerface', '5celeb'):
                    self.add_datasource(key)
                # 'haar': opencv
                # 'cnn': opencv
                # 'ssd': dlib
                # 'hog': dlib
                # 'mtcnn': mtcnn (keras)
                face_detectors = ('haar', 'hog', 'ssd')
                if args.face_all:
                    face_detectors += ('mtcnn')  # 'cnn'
                    # FIXME[todo]: 'cnn' runs really slow on CPU and
                    # blocks the GUI! - we may think of doing
                    # real multiprocessing!
                    # https://stackoverflow.com/questions/7542957/is-python-capable-of-running-on-multiple-cores
                    # https://stackoverflow.com/questions/47368904/control-python-code-to-run-on-different-core
                    # https://docs.python.org/3/library/multiprocessing.html
                    # https://stackoverflow.com/questions/10721915/shared-memory-objects-in-multiprocessing
                    # detector = FaceDetector[key] # .create(name, prepare=False)
                for key in face_detectors: 
                    LOG.info("Toolbox: Initializing detector '%s'", key)
                    detector = Tool[key]
                    detector.runner = self.runner  # FIXME[hack]
                    # LOG.info("Toolbox: Preparing detector '%s'", key)
                    # detector.prepare(busy_async=False)
                    self.add_tool(detector)

        #
        # Datasources
        #
        for id in 'Webcam', 'Noise', 'Helen', 'Movie':
            self.add_datasource(Datasource[id])

        if args is not None:
            if args.data:
                self.add_datasource(args.data)
            elif args.datasource:
                self.add_datasource(args.datasource)
            elif args.datadir:
                self.add_datasource(DataDirectory(args.datadir))

            if args.imagenet:
                self.add_datasource('imagenet-val', select=True)
            if args.helen:
                self.add_datasource('Helen', select=True)

        #
        # Networks
        #
        if args is not None:
            for network in NetworkArgparse.networks(args):
                self.add_network(network)  # prepare=True

    def run(self) -> int:
        #
        # Signal handling
        #

        # Note 1: This has to be done in the main thread.
        #
        # Note 2: There may be a problem when this is done in the
        # context of PyQT: the main event loop is run in the C++ core
        # of Qt and is not aware of the Python interpreter. The
        # interrupt handler will only be called after a (Python) event
        # is emitted by the GUI that is handled by the (Python)
        # interpreter.  An ad hoc solution is to use a QTimer to
        # periodically run some (dummy) Python code to make sure the
        # signal handler gets a chance to be executed.
        # However, this should not be a problem for us, as the GUI
        # should start a background timer to periodically update the
        # user interface.
        # FIXME[question]: why is this done here?

        # Setup handling of KeyboardInterrupt (Ctrl-C)
        signal.signal(signal.SIGINT, self._interrupt_handler)
        # Setup handling of Quit (Ctrl-\)
        signal.signal(signal.SIGQUIT, self._interrupt_handler)

        #
        # User Interface
        #
        if self.option('shell'):
            gui = None
        else:
            gui = 'qtgui'

        if gui is not None:
            rc = self.start_gui(gui=gui)
        else:
            self.setup()
            # self.setup(tools=tools, networks=networks, datasources=datasources)
            self.run_shell(asynchronous=False)
            self.quit(sys_exit=False)
            rc = 0
        return rc

    #
    # Shell
    #

    @property
    def shell(self):
        return hasattr(self, '_shell')

    def run_shell(self, asynchronous: bool = True):
        """Run the toolbox shell.
        """
        if self.shell:
            return  # shell is already running - nothing to do
        from .shell import ToolboxShell
        self._shell = ToolboxShell(self)
        self.change('shell_changed')

        def run_shell_loop():
            self._shell.cmdloop()
            self.stop_shell()

        if asynchronous:
            self._shell_thread = threading.Thread(target=run_shell_loop)
            # We make the shell thread a daemon thread, meaning that
            # the program may exit even if the shell thread is still
            # running. This is currently necessary, as we have no way
            # to stop the shell programatically (see comment in stop_shell).
            self._shell_thread.setDaemon(True)
            self._shell_thread.start()
        else:
            run_shell_loop()

    def stop_shell(self):
        if not self.shell:
            return  # no shell is running - nothing to do

        if hasattr(self, '_shell_thread'):
            # FIXME[problem]: don't know how to terminate Cmd.cmdloop.
            # That loop blocks on input() or readline() and does not
            # seem to provide a way to stop it from the outside.
            # Maybe we should subclass Cmd and reimplement that method.
            if self._shell_thread is not threading.current_thread:
                print("Toolbox: Cannot terminate Cmd.cmdloop"
                      " - please quit the shell manually")
                # self._shell_thread.join()
                # del self._shell_thread
            else:
                self._shell_thread = None
                del self._shell_thread

        del self._shell
        self.change('shell_changed')

    #
    # GUI
    #

    @property
    def gui(self):
        return self._gui

    def start_gui(self, gui: str = 'qtgui', threaded: bool = None, **kwargs):
        """Start the graphical user interface.

        Arguments
        ---------
        gui: str
            The graphical user interface to use. Currently supported
            are 'qtgui' and 'gtkgui'.
        threaded: bool
            Run the GUI in its own thread.
        """

        # we need the GUI first to get the runner ...

        #
        # Step 1: determine the GUI to use
        #
        valid_guis = {
            'qtgui': 'PyQt5',
            'gtkgui': 'gi'
        }

        if gui not in valid_guis:
            raise ValueError(f"Invalid gui module '{gui}'. "
                             f"Valid values are {valid_guis.keys()}")
        for name, module in valid_guis.items():
            if gui is not None and gui != name:
                continue
            spec = importlib.util.find_spec(module)
            if spec is not None:
                gui = name
                break
            elif gui is not None:
                message = f"GUI library ({module}) was not found."
                logging.fatal(message)
                raise RuntimeError(message)
        if gui is None:
            raise RuntimeError("No GUI is supported by your environment.")

        #
        # Step 2: select the panels to display
        #
        panels = kwargs.get('panels', [])
        if addons.use('autoencoder'):
            panels.append('autoencoder')

        if self.option('internals'):
            panels.append('internals')
        # if addons.internals('internals')
        # Initialise the "Activation Maximization" panel.
        # if addons.use('maximization'):
        #    panels.append('maximization')
        if self.option('face'):
            panels.append('face')
            # wider_faces = Datasource['wider-faces-train']
            # datasources.append(wider_faces)
            # self._datasource(wider_faces)

        if self.option('resources'):
            panels.append('resources')

        if self.option('activations'):
            panels.append('activations')

        if self.option('adversarial') or self.option('advexample'):
            panels.append('advexample')

        if self.option('styletransfer'):
            panels.append('styletransfer')

        kwargs['panels'] = panels

        #
        # Step 3: run the GUI
        #
        if threaded is None:
            # Automatically determine if to run GUI in a separate Thread:
            # In an interactive interpreter, we want to return to
            # input mode an hence rund the GUI in its own Thread.
            # If not running interactively, we can use the current Thread
            # as GUI event loop.
            #
            # To test if we run in an interactive interpreter we can
            # check the existence of sys.ps1 and sys.ps2, which are
            # only defined in interactive mode. (An alternative would
            # be to check if the module __main__ has the attribute
            # __file__, which shouldn't exist in interactive mode:
            #   import __main__ as main
            #   threaded = not hasattr(main, '__file__')
            threaded = hasattr(sys, 'ps1')

        if threaded:
            thread = threading.Thread(target=self._run_gui, name="gui-thread",
                                      args=(gui,), kwargs=kwargs)
            thread.start()
            self._gui_thread = thread
        else:
            rc = self._run_gui(gui=gui, **kwargs)

    def _run_gui(self, gui, panels=[], tools=[], networks=[],
                 datasources=[], **kwargs):
        """Create a graphical user interface for this :py:class:`Toolbox`
        and run the main event loop.
        """

        #
        # create the actual application
        #

        # We use the local import here to avoid circular imports
        # (the gui_module may import toolbox.Toolbox)
        gui_module = importlib.import_module(gui)
        self._gui = gui_module.create_gui(sys.argv, self, **kwargs)

        # FIXME[hack]: we need a better solution here!
        self.set_runner(self._gui.getRunner())

        #
        # Initialize the panels.
        #
        panel = None
        for panel in panels:
            self._gui.panel(panel, create=True)
        if panel is not None:  # show the last panel
            self._gui.panel(panel, show=True)

        # Show the logging panel.
        #
        # FIXME[concept]: think about initialization of panels
        #  - panels should only be initialized on demand
        #    -> no explicit instantiation here
        #  - toolbox should have no knowledge on panel internals
        #    -> the panel should request necessary information
        #       (logger, logging_recorder, ...) from the Toolbox
        #
        logging_panel = self._gui.panel('logging', create=True, show=False)
        if logging_panel is not None:  # can only fail if create=False
            if self.logging_recorder is not None:
                logging_panel.setLoggingRecorder(self.logging_recorder)
            # add the root logger to the logging panels
            logging_panel.addLogger(logging.getLogger())

        try:
            # This will enter the main event loop and will only return
            # once the main event loop exits.
            # Before entering the main event loop, it will start the
            # execution of self.setup() in a QtAsyncRunner background
            # thread, passing the arguments provided to gui.run().
            # -> This will ensure that the GUI can show up quickly,
            # even if completing the setup procedure may take some time.
            LOG.info("Toolbox: running the GUI main event loop")
            # FIXME[hack]: we need a better way to determine if a thread
            # is a GUI event loop
            import threading
            threading.current_thread().GUI_event_loop = True
            rc = self._gui.run(tools=tools, networks=networks,
                               datasources=datasources)
            LOG.info("Toolbox: GUI main event loop finished (rc=%d)", rc)
            if self._gui.isVisible():
                self._gui.close()
            print("Toolbox: GUI main event loop finished"
                  f" (rc={rc}, visible={self._gui.isVisible()})")
        finally:
            # FIXME[old] ...
            # print("Toolbox: finally stopping the timer ...")
            # self._gui.stopTimer()
            self._gui = None
        return rc

    #
    # Logging and error handling
    #

    def _initialize_logging(self, record: bool = True,
                            level: int = logging.DEBUG) -> None:
        """Initialize logging by this :py:class:`Toolbox`.  This is based on
        the infrastructure provided by Python's `logging` module,
        consisting of :py:class:`logging.Logger`s and
        :py:class:`logging.Handler`s.

        The :py:class:`Toolbox` class

        Each module of the toolbox should define its own
        :py:class:`logging.Logger` by calling `logger =
        logging.get_logger(__name__)` to allow for individual
        configuration.

        Arguments
        ---------
        record: bool
            A flag indicating if :py:class:`util.RecorderHandler` should
            be added to this :py:class:`Toolbox`.
        level: int
            The basic log level to be set.
        """

        # The the basic logging level
        logging.basicConfig(level=level)

        if record:
            LOG.info("Changing global logging Handler to RecorderHandler.")

            # remove all current handlers from the root logger.
            original_handlers = logging.getLogger().handlers.copy()
            for handler in original_handlers:
                self.remove_logging_handler(handler)

            # use a util.RecorderHandler
            if record:
                self.record_logging()

    def add_logging_handler(self, handler: logging.Handler) -> None:
        """Add a new logging handler to this :py:class:`Toolbox`.

        Arguments
        ---------
        handler:
            The :py:class:`logging.Handler` to be added.
        """
        LOG.info("Adding log handler: %r", handler)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

    def remove_logging_handler(self, handler: logging.Handler) -> None:
        """Remove a logging handler to this :py:class:`Toolbox`.

        Arguments
        ---------
        handler:
            The :py:class:`logging.Handler` to be removed.
        """
        LOG.info("Removing log handler: %r", handler)
        root_logger = logging.getLogger()
        root_logger.removeHandler(handler)

    @property
    def logging_recorder(self) -> util.RecorderHandler:
        """The logging RecorderHandler. May be None if the
        :py:class:`Toolbox` was not instructed to record
        :py:class:`LogRecord`s.
        """
        return getattr(self, '_logging_recorder', None)

    def record_logging(self, flag: bool = True) -> None:
        """Determine if a logging recorder should be used in this
        :py:class:`Toolbox`. A logging record will record all
        log records emitted, so they can be inspected later on.
        """
        if flag == hasattr(self, '_logging_recorder'):
            return  # nothing to do
        if flag:
            self._logging_recorder = util.RecorderHandler()
            self.add_logging_handler(self._logging_recorder)
        else:
            self.remove_logging_handler(self._logging_recorder)
            del self._logging_recorder

    @property
    def logging_formatter(self) -> logging.Formatter:
        if not hasattr(self, '_logging_formatter'):
            # format='%(relativeCreated)6d %(threadName)s %(message)s'
            # format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
            # format='%(relativeCreated)6d %(threadName)s %(message)s'
            # format='%(relativeCreated)5d %(name)-15s %(levelname)-8s %(message)s')
            # format='%(levelname)s %(name)s %(message)s'
            # format='%(asctime)-15s %(name)-5s %(levelname)-8s IP: %(ip)-15s User: %(user)-8s %(message)s')
            #format = f"{args.info}[%(threadName)s]: %(levelname)-8s %(message)s"
            format = f"[%(threadName)s]: %(levelname)-8s %(message)s"
            self._logging_formatter = logging.Formatter(format)
        return self._logging_formatter

    def _initialize_exception_handler(self, record: bool = True) -> None:
        """Initialize exception handling by this Toolbox.
        This will register als the global exception handler, that is
        calls to :py:func:`util.error.handle_exception` will be
        consumed by this :py:class:`Toolbox`.

        Arguments
        ---------
        record: bool
            A flag indicating if exceptions should be recorded for
            later inspection.  If true, the list of exceptions reported
            can be obtained via the property exceptions.
        """
        self._exception_handlers = set()
        self.add_exception_handler(util.error.print_exception)
        if record:  # record exceptions
            self._exceptions = []
            self.add_exception_handler(self._record_exception)
        util.error.set_exception_handler(self.handle_exception)

    def handle_exception(self, exception: BaseException) -> None:
        """Handle an exceptoin
        """
        for handler in self._exception_handlers:
            handler(exception)

    def add_exception_handler(self, handler):
        """Add an exception handler to this Toolbox.
        """
        self._exception_handlers.add(handler)

    def remove_exception_handler(self, handler):
        """Remove an exception handler from this Toolbox.
        """
        self._exception_handlers.remove(handler)

    def _record_exception(self, exception: BaseException) -> None:
        """An exception handler to record exceptions. Exceptions recorded will
        be accessible via the property :py:meth:`exceptions`.

        Arguments
        ---------
        exception:
            The exception to record.
        """
        self._exceptions.append(exception)

    @property
    def exceptions(self):
        """A list of exceptions recorded by this Toolbox.
        """
        if not hasattr(self, '_exceptions'):
            RuntimeError("Recording of exceptions has not been activated "
                         "at this Toolbox.")
        return self._exceptions

    #
    # Server
    #

    @property
    def server(self) -> bool:
        return hasattr(self, '_server')

    @server.setter
    def server(self, state: bool):
        if state == self.server:
            return
        if state:
            from .http import Server
            self._server = Server()
            self._server.serve()
        else:
            self._server.stop()
            del self._server
        self.change('server_changed')

    @property
    def server_url(self) -> str:
        return self._server.url() if self.server else None

    @property
    def server_status_message(self) -> str:
        """A string describing the current server status.
        """
        return (f"Serving at {self.server_url}" if self.server else
                "Server is down")

    def server_open(self) -> None:
        if self.server:
            import webbrowser
            webbrowser.open(self.server_url)

    #
    # Processes
    #

    # FIXME[experimental]: this is experimental code that should not
    # be part of the Toolbox

    # FIXME[bug]: a crash of the main python program leaves background
    # processes alive!
    # [such a crash can be caused by sounddevice]
    # Killing such a leftover process from the command line may even
    # crash the X-Server!
    #
    # We need some mechanism to avoid some unwanted background processes
    # when ending the program (regularly or irregularly)
    #
    # It may, however, be useful to start a background process and then
    # later reconnect to that process, e.g. to obtain results

    # In fact, the Deep Learning Toolbox currently starts quite a lot
    # processes:
    #
    #   python3(32251)-+--Deep Learning T(32253)
    #                  +--{QDBusConnection}(32266)
    #                  +--{QXcbEventQueue}(32265)
    #                  +--{python3}(32254)
    #                  +--{python3}(32268)
    #                  +--{python3}(32269)
    #
    # Notice that most of these are actually threads, not processes
    # (under linux, threads are merely processes that use the same
    # address space as another process. They will have the same PID
    # as the main process, but a different LWP
    # (light weight process = thread ID)).

    def _initialize_processes(self) -> None:
        """
        """
        if True:
            print("Toolbox: Not initializing processes")
            self._process = None
            return

        print("Toolbox: Initializing process")
        self._process = Process(name='test')
        print("Toolbox: Starting process")
        self._process.start()
        print("Toolbox: Process is running")
        self.notify_process("Hallo")

    def get_process(self, name: str) -> Process:
        return self._process

    def notify_process(self, message: str) -> None:
        if self._process is not None:
            self._process.send(message)

    def _finish_processes(self) -> None:
        if self._process is not None:
            print("Toolbox: Terminating process")
            self._process.terminate()
            print("Toolbox: Joining process")
            self._process.join()
            print("Toolbox: Finished process")
            self._process = None

    #
    # Miscallenous
    #

    def __repr__(self):
        return f"<Toolbox: {self._gui is not None}>"

    def __str__(self):
        return ("Toolbox with "
                f"{len(self._datasources or '')}/{len(Datasource)} "
                "datasources, "
                f"{len(self._networks or '')}/{len(Network)} networks, "
                f"{len(Tool)} tools.")

    def old__str__(self):
        """String representation of this :py:class:`Toolbox`.
        """
        result = f"GUI: {self._gui is not None}"
        result += "\nTools:"
        if self._tools is None:
            result += " None"
        else:
            for tool in self._tools:
                result += f"\n  - {tool}"

        result += "\nNetworks:"
        if self._networks is None:
            result += " None"
        else:
            for network in self._networks:
                result += f"\n  - {network}"

        result += "\nDatasources:"
        if self._datasources is None:
            result += " None"
        else:
            for datasource in self._datasources:
                if datasource == self.datasource:
                    mark = '*'
                elif datasource.prepared:
                    mark = '+'
                else:
                    mark = '-'
                result += f"\n  {mark} {datasource}"

        result += "\nPredefined data sources: "
        result += f"{list(Datasource.instance_register.keys())}"

        return result + "\n"

    def old__str__2(self):
        self = self._toolbox  # FIXME[hack]
        result = f"GUI: {self._gui is not None}"
        result += "\nTools:"
        if self._tools is None:
            result += " None"
        else:
            for tool in self._tools:
                result += f"\n  - {tool}"

        result += "\nNetworks:"
        if self._networks is None:
            result += " None"
        else:
            for network in self._networks:
                result += f"\n  - {network}"

        result += "\nDatasources:"
        if self._datasources is None:
            result += " None"
        else:
            for datasource in self._datasources:
                result += f"\n  - {datasource}"

        return result + "\n"

    @staticmethod
    def debug_register():
        Datasource.debug_register()
        Network.debug_register()
        Tool.debug_register()
        print("\n")

    #
    # FIXME[hack/old]
    #

    @property
    def autoencoder_controller(self) -> 'AutoencoderController':
        if True:
            raise NotImplementedError("The AutoencoderController is "
                                      "currently not working")
        controller = getattr(self, '_autoencoder_controller', None)
        # FIXME[update]: make the AutoencoderController a Tool/Network
        # if controller is None:
        #     controller = AutoencoderController(runner=self._runner)
        #     self._autoencoder_controller = controller
        return controller

    @property
    def training_controller(self) -> TrainingController:
        controller = getattr(self, '_training_controller', None)
        if controller is None:
            controller = TrainingController(runner=self._runner)
            self._training_controller = controller
        return controller

    @property
    def maximization_engine(self) -> BaseController:  # -> tools.am.Controller
        """Get the Controller for the activation maximization engine. May
        create a new Controller (and Engine) if none exists yet.
        """
        engine_controller = getattr(self, '_am_engine', None)
        if engine_controller is None:
            from tools.am import (Engine as MaximizationEngine,
                                  Config as MaximizationConfig,
                                  Controller as MaximizationController)
            engine = MaximizationEngine(config=MaximizationConfig())
            engine_controller = \
                MaximizationController(engine, runner=self._runner)
            self._am_engine = engine_controller
        return engine_controller

    def hack_load_mnist(self):
        # FIXME[old]: seems not to be used anymore
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
        # self.x_train, self.y_train = mnist[0]
        # self.x_test, self.y_test = mnist[1]
        self.dataset = mnist
        self.data = mnist[1]  # FIXME[hack]

        # FIXME[hack]: we need a better training concept ...
        from tools.train import Training
        self.training = Training()
        # self.training.\
        #    set_data(self.get_inputs(dtype=np.float32, flat=True, test=False),
        #             self.get_labels(dtype=np.float32, test=False),
        #             self.get_inputs(dtype=np.float32, flat=True, test=True),
        #             self.get_labels(dtype=np.float32, test=True))

    @property
    def data(self):
        # FIXME[old]: seems not to be used anymore
        print("FIXME[old]: Toolbox.data should not be used anymore")
        raise RuntimeError("FIXME[old]: Toolbox.data getter should "
                           "not be used anymore")
        return self._data

    @data.setter
    def data(self, data):
        # FIXME[old]: seems not to be used anymore
        print("FIXME[old]: Toolbox.data should not be used anymore")
        raise RuntimeError("FIXME[old]: Toolbox.data setter should not "
                           "be used anymore")
        self._data = (None, None) if data is None else data

    @property
    def inputs(self):
        # FIXME[old]: seems not to be used anymore
        print("FIXME[old]: Toolbox.inputs should not be used anymore")
        raise RuntimeError("FIXME[old]: Toolbox.inputs getter should not "
                           "be used anymore")
        return self._data[0]

    @property
    def labels(self):
        # FIXME[old]: used in ./qtgui/panels/autoencoder.py
        print("FIXME[old]: Toolbox.labels should not be used anymore")
        raise RuntimeError("FIXME[old]: Toolbox.labels getter should not "
                           "be used anymore")
        return self._data[1]

    #
    # FIXME[old] old stuff from the controller
    #

    # FIXME[old]: seems not to be used anymore ...
    def set_network(self, network: Network):
        for attribute in '_autoencoder_controller':
            if getattr(self, attribute, None) is not None:
                setattr(self, attribute, network)

    #
    # Datasources
    #

    def get_inputs(self, dtype=np.float32, flat=True, test=False):
        inputs = self.dataset[1 if test else 0][0]
        if (np.issubdtype(inputs.dtype, np.integer) and
                np.issubdtype(dtype, np.floating)):
            # conversion from int to float will also scale to the interval
            # [0,1].
            inputs = inputs.astype(dtype)/256
        if flat:
            inputs = np.reshape(inputs, (-1, inputs[0].size))
        return inputs

    def get_labels(self, dtype=np.float32, one_hot=True, test=False):
        labels = self.dataset[1 if test else 0][1]
        if not one_hot:
            labels = labels.argmax(axis=1)
        return labels

    def get_data_shape(self):
        return self.dataset[0][0][0].shape
