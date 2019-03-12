
#
# Changing global logging Handler
#

import os
import sys
import logging
print("!!!!!!!!!!!!!!!! Changing global logging Handler !!!!!!!!!!!!!!!!!!!!")
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

from base import Observable, change, Runner
from base import View as BaseView, Controller as BaseController
from util import addons

# FIXME[todo]: speed up initialization by only loading frameworks
# that actually needed

from network import Network
from network import AutoencoderController
from datasources import Datasource, Controller as DatasourceController
from tools.train import TrainingController



# from network.keras import Network as KerasNetwork
# from network.torch import Network as TorchNetwork

def keras(backend: str, cpu: bool,
          model_file: str='models/example_keras_mnist_model.h5') -> Network:
    # actually: KerasNetwork
    '''
    Visualise a Keras-based network

    Parameters
    ----------
    backend     :   str
                    Name of the Keras backend
                    (currently only "tensorflow" or "theano")
    cpu         :   bool
                    Whether to use only cpu, not gpu
    model_file       :   str
                    Filename where the model_file is located (in hdf5)

    Returns
    -------
    network.network.Network
        The concrete network instance to visualise

    Raises
    ------
    RuntimeError
        In case of unknown backend

    '''
    if backend == 'tensorflow':
        return Network.load('network.keras_tensorflow',
                            model_file=model_file)
    elif backend == 'theano':
        return Network.load('network.keras_theano',
                            model_file=model_file)
    else:
        raise RuntimeError('Unknown backend {backend}')


def torch(cpu: bool, model_file: str, net_class: str, parameter_file: str,
          input_shape: tuple) -> Network:  # actually: TorchNetwork
    '''
    Visualise a Torch-based network

    .. error:: Torch network currently does not work.

    Parameters
    ----------
    cpu : bool
        Whether to use only cpu, not gpu

    model_file : str
        Filename where the model is defined (a Python file with a
        :py:class:`torch.nn.Module` sublcass)

    net_class : str
        Name of the model_file class (see ``model_file``)

    parameter_file : str
        Name of the file storing the model weights (pickled torch weights)

    input_shape : tuple
        Shape of the input images

    Returns
    -------
    network: TorchNetwork
        The concrete network instance to visualize.

    '''
    # FIXME[todo]: Fix errors when running torch network
    return Network.load('network.torch',
                        model_file, parameter_file, net_class=net_class,
                        input_shape=input_shape, use_cuda=not cpu)




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

class View(BaseView, view_type=Toolbox):

    def __init__(self, toolbox=None, **kwargs):
        super().__init__(observable=toolbox, **kwargs)

    @property
    def networks(self):
        return () if self._toolbox is None else iter(self._toolbox._networks)

    @property
    def datasources(self):
        return (() if self._toolbox is None else
                iter(self._toolbox._datasources))

def debug(func):
    def closure(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            import traceback
            print(f"DEBUG: {type(e).__name__}: {e}")
            traceback.print_tb(e.__traceback__)
            raise e
    return closure


class Controller(View, BaseController):

    def __init__(self, toolbox=None, **kwargs):
        super().__init__(toolbox=toolbox, **kwargs)

    @property
    def autoencoder_controller(self) -> AutoencoderController:
        controller = getattr(self._toolbox, '_autoencoder_controller', None)
        if controller is None:
            controller = AutoencoderController(runner=self._toolbox._runner)
            self._toolbox._autoencoder_controller = controller
        return controller

    @property
    def training_controller(self) -> TrainingController:
        controller = getattr(self._toolbox, '_training_controller', None)
        if controller is None:
            controller = TrainingController(runner=self._toolbox._runner)
            self._toolbox._training_controller = controller
        return controller

    @property
    @debug
    def maximization_engine(self) -> BaseController:  # -> tools.am.Controller
        """Get the Controller for the action maximization engine. May
        create a new Controller (and Engine) if none exists yet.
        """
        engine_controller = getattr(self._toolbox, '_am_engine', None)
        if engine_controller is None:
            from tools.am import Engine as AMEngine, Config as AMConfig
            from controller import MaximizationController
            engine = AMEngine(self._toolbox._model, AMConfig())
            engine_controller = \
                MaximizationController(engine, runner=self._runner)
            self._toolbox._am_engine = engine_controller
        return engine_controller

    ###########################################################################
    ###                            Networks                                 ###
    ###########################################################################

    def add_network(self, network: Network) -> None:
        self._toolbox.add_network(network)
        self.set_network(network)

    def remove_network(self, network: Network) -> None:
        self._toolbox.remove_network(network)
        # FIXME[todo]: unset network from views!

    def set_network(self, network: Network):
        for attribute in '_autoencoder_controller':
            view = getattr(self._toolbox, attribute, None)
            if view is not None:
                view(network)

    # FIXME[hack]: setting the new model will add observers, may not be done asynchronously!
    #@run
    def hack_new_model(self):
        # FIXME[hack]:
        if self.data is None:
            self.hack_load_mnist()

        original_dim = self.data[0][0].size
        print(f"Hack 1: new model with original_dim={original_dim}")
        intermediate_dim = 512
        latent_dim = 2
        from models.example_keras_vae_mnist import KerasAutoencoder
        network = KerasAutoencoder(original_dim)
        self.add_network(network)
        return network

    #@run
    def hack_new_model2(self):
        # FIXME[hack]:
        if self.data is None:
            self.hack_load_mnist()

        original_dim = self.data[0][0].size
        print(f"Hack 2: new model with original_dim={original_dim}")
        from models.example_keras_vae_mnist import KerasAutoencoder
        network = KerasAutoencoder(original_dim)
        self.add_network(network)
        return network

    def hack_new_alexnet(self):
        alexnet = self._toolbox.hack_load_alexnet()
        self.add_network(alexnet)

    ###########################################################################
    ###                            Datasources                              ###
    ###########################################################################

    @property
    def datasource_controller(self) -> DatasourceController:
        controller = getattr(self._toolbox, '_datasource_controller', None)
        if controller is None:
            controller = DatasourceController(runner=self._toolbox._runner)
            self._toolbox._datasource_controller = controller
        return controller

    def add_datasource(self, datasource: Datasource) -> None:
        self._toolbox.add_datasource(datasource)
        self.set_datasource(datasource)

    def remove_network(self, datasource: Datasource) -> None:
        self._toolbox.remove_datasource(datasource)
        # FIXME[todo]: unset datasource from views!

    def set_datasource(self, datasource: Datasource):
        for attribute in '_datasource_controller':
            view = getattr(self._toolbox, attribute, None)
            if view is not None:
                view(datasource)

    def get_inputs(self, dtype=np.float32, flat=True, test=False):
        inputs = self.dataset[1 if test else 0][0]
        print(f"ToolboxController.get_inputs(): inputs: {inputs.shape}, {inputs.dtype}, {inputs.max()}")
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
        print(f"labels: {labels.shape}, {labels.dtype}")
        if not one_hot:
            labels = labels.argmax(axis=1)
        return labels

    def get_data_shape(self):
        return self.dataset[0][0][0].shape

# FIXME[hack]
ToolboxView = View
ToolboxController = Controller
