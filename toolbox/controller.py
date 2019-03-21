from base import View as BaseView
from .toolbox import Toolbox


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


import util
from .toolbox import Toolbox
from base import Controller as BaseController
from network import Network, AutoencoderController
from datasources import Datasource, Controller as DatasourceController
from tools.train import TrainingController

import numpy as np


class Controller(View, BaseController):

    def __init__(self, toolbox=None, **kwargs):
        super().__init__(toolbox=toolbox, **kwargs)

    @property
    def autoencoder_controller(self) -> AutoencoderController:
        controller = getattr(self._toolbox, '_autoencoder_controller', None)
        if controller is None:
            controller = AutoencoderController(runner=self._runner)
            self._toolbox._autoencoder_controller = controller
        return controller

    @property
    def training_controller(self) -> TrainingController:
        controller = getattr(self._toolbox, '_training_controller', None)
        if controller is None:
            controller = TrainingController(runner=self._runner)
            self._toolbox._training_controller = controller
        return controller

    @property
    def activation_controller(self) -> BaseController:
        # -> tools.activation.Controller
        """Get the Controller for the activation engine.
        """
        return getattr(self._toolbox, '_activation_controller', None)

    @property
    def maximization_engine(self) -> BaseController:  # -> tools.am.Controller
        """Get the Controller for the activation maximization engine. May
        create a new Controller (and Engine) if none exists yet.
        """
        engine_controller = getattr(self._toolbox, '_am_engine', None)
        if engine_controller is None:
            from tools.am import (Engine as MaximizationEngine,
                                  Config as MaximizationConfig,
                                  Controller as MaximizationController)
            engine = MaximizationEngine(config=MaximizationConfig())
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
        return getattr(self._toolbox, '_datasource_controller', None)

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


    ###########################################################################
    ###                            Input data                               ###
    ###########################################################################

    #@run
    def set_input_from_file(self, filename: str, label=None,
                            description: str=None):
        image = util.imread(filename)
        self.set_input(image, label=label,
                       description=description or
                       f"Image from file '{filename}'")
